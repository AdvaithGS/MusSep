"""Two-level stem taxonomy: parents, candidate sub-stem classes, the mapping
from dataset labels to classes, and the minimum-data rule that decides which
candidate classes survive.

Life cycle::

    taxonomy = Taxonomy.load("configs/taxonomy.yaml")          # candidates
    merges, collapsed, log = plan_merges(taxonomy, songs, 20)  # counted on real data
    frozen = taxonomy.with_merges(merges, collapsed)           # what training uses
    frozen.save("configs/taxonomy.frozen.yaml")

Dataset indexers always record the *candidate* ("raw") class of a file;
:meth:`Taxonomy.resolve` maps it to the class that is actually modelled.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

import yaml

from utils.config import load_yaml

NULL_CLASS = "<null>"


def normalize_label(label: Any) -> str:
    """Lower-case and drop everything except letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


class Assignment(NamedTuple):
    parent: Optional[str]       # None only if excluded
    raw_class: Optional[str]    # candidate class; None if the parent is not split
    matched: bool               # False if a default / wildcard rule decided
    excluded: bool = False


class Taxonomy:
    def __init__(self, spec: Mapping[str, Any]):
        self.spec: Dict[str, Any] = copy.deepcopy(dict(spec))
        self.frozen = bool(self.spec.get("frozen", False))
        min_data = self.spec.get("min_data") or {}
        self.min_songs = int(min_data.get("min_songs", 0))
        self.count_sources: Tuple[str, ...] = tuple(min_data.get("sources") or ("moisesdb",))
        self.count_split: str = str(min_data.get("split", "train"))

        self.candidates: Dict[str, List[str]] = {}
        self.residual: Dict[str, Optional[str]] = {}
        self.collapse_if_residual_rare: Dict[str, bool] = {}
        self.fallback: Dict[str, str] = {}
        self._parent_of: Dict[str, str] = {}
        for parent, body in (self.spec.get("parents") or {}).items():
            names: List[str] = []
            for entry in body.get("classes") or []:
                if isinstance(entry, Mapping):
                    name = str(entry["name"])
                    if entry.get("fallback"):
                        self.fallback[name] = str(entry["fallback"])
                else:
                    name = str(entry)
                if name in self._parent_of or name == NULL_CLASS:
                    raise ValueError(f"duplicate or reserved class name '{name}'")
                self._parent_of[name] = parent
                names.append(name)
            self.candidates[parent] = names
            residual = body.get("residual")
            if residual is not None and residual not in names:
                raise ValueError(f"residual '{residual}' is not a class of parent '{parent}'")
            self.residual[parent] = residual
            self.collapse_if_residual_rare[parent] = bool(body.get("collapse_if_residual_rare", False))
        for name, target in self.fallback.items():
            if self._parent_of.get(target) != self._parent_of[name] or target == name:
                raise ValueError(f"fallback of '{name}' must be another class of the same parent")

        self.merges: Dict[str, str] = {str(k): str(v) for k, v in (self.spec.get("merges") or {}).items()}
        for name, target in self.merges.items():
            if name not in self._parent_of or self._parent_of.get(target) != self._parent_of[name]:
                raise ValueError(f"invalid merge '{name}' -> '{target}'")
            if name == self.residual[self._parent_of[name]]:
                raise ValueError(f"the residual class '{name}' cannot be merged away")
        for name in self.merges:
            self._root(name)  # raises on cycles

        self.collapsed: Set[str] = set(self.spec.get("collapsed_parents") or [])
        for parent in self.candidates:
            if len(self._active(parent)) < 2:
                self.collapsed.add(parent)

        self.parents: List[str] = [p for p in self.candidates if p not in self.collapsed]
        self.class_names: List[str] = [NULL_CLASS]
        for parent in self.parents:
            self.class_names.extend(self._active(parent))
        self._ids = {name: index for index, name in enumerate(self.class_names)}

        self._init_moisesdb()
        self._init_slakh()

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        return cls(load_yaml(path))

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> "Taxonomy":
        return cls(spec)

    def to_dict(self) -> Dict[str, Any]:
        spec = copy.deepcopy(self.spec)
        spec["frozen"] = self.frozen
        spec["merges"] = dict(self.merges)
        spec["collapsed_parents"] = sorted(self.collapsed)
        return spec

    def with_merges(self, merges: Mapping[str, str], collapsed: Iterable[str] = (), frozen: bool = True) -> "Taxonomy":
        spec = self.to_dict()
        spec["merges"] = dict(merges)
        spec["collapsed_parents"] = sorted(set(collapsed))
        spec["frozen"] = frozen
        return Taxonomy(spec)

    def save(self, path: str | Path, header: Optional[str] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            if header:
                for line in header.rstrip("\n").split("\n"):
                    handle.write(f"# {line}".rstrip() + "\n")
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)

    # -- classes ----------------------------------------------------------

    def _root(self, name: str) -> str:
        seen = {name}
        while name in self.merges:
            name = self.merges[name]
            if name in seen:
                raise ValueError(f"merge cycle through '{name}'")
            seen.add(name)
        return name

    def _active(self, parent: str) -> List[str]:
        return [name for name in self.candidates[parent] if name not in self.merges]

    @property
    def null_id(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        """Size of the label embedding, including the null label."""
        return len(self.class_names)

    @property
    def max_slots(self) -> int:
        return max((len(self._active(parent)) for parent in self.parents), default=0)

    def class_id(self, name: str) -> int:
        return self._ids[name]

    def parent_classes(self, parent: str) -> List[str]:
        """Modelled classes of ``parent`` (after merges), in canonical order."""
        if parent not in self.parents:
            raise KeyError(f"'{parent}' is not a split parent; choose from {self.parents}")
        return self._active(parent)

    def parent_of(self, class_name: str) -> str:
        return self._parent_of[class_name]

    def resolve(self, raw_class: Optional[str]) -> Optional[str]:
        """Candidate class -> modelled class (``None`` if its parent is not split)."""
        if raw_class is None or self._parent_of[raw_class] in self.collapsed:
            return None
        return self._root(raw_class)

    # -- MoisesDB ---------------------------------------------------------

    def _init_moisesdb(self) -> None:
        spec = self.spec.get("moisesdb") or {}
        self._mdb_exclude = {
            (normalize_label(item.get("stem", "")), normalize_label(item.get("track_type", "")))
            for item in spec.get("exclude") or []
        }
        stem_to_parent = dict(spec.get("stem_to_parent") or {})
        self._mdb_default_parent = stem_to_parent.pop("*", None)
        self._mdb_parent = {normalize_label(k): v for k, v in stem_to_parent.items()}
        self._mdb_rules: Dict[str, Dict[str, Any]] = {}
        for parent, rule in (spec.get("rules") or {}).items():
            compiled = {
                "track_types": {normalize_label(k): v for k, v in (rule.get("track_types") or {}).items()},
                "stems": {normalize_label(k): v for k, v in (rule.get("stems") or {}).items()},
                "default": rule.get("default"),
            }
            for name in list(compiled["track_types"].values()) + list(compiled["stems"].values()) + [compiled["default"]]:
                if name is not None and self._parent_of.get(name) != parent:
                    raise ValueError(f"moisesdb rule for '{parent}' names unknown class '{name}'")
            self._mdb_rules[parent] = compiled

    def moisesdb_lookup(self, stem_name: str, track_type: str) -> Assignment:
        stem, kind = normalize_label(stem_name), normalize_label(track_type)
        if (stem, kind) in self._mdb_exclude or (stem, "") in self._mdb_exclude:
            return Assignment(None, None, True, True)
        matched = stem in self._mdb_parent
        parent = self._mdb_parent.get(stem, self._mdb_default_parent)
        if parent is None:
            return Assignment(None, None, False, True)
        if not self.candidates.get(parent):
            return Assignment(parent, None, matched)
        rule = self._mdb_rules.get(parent, {"track_types": {}, "stems": {}, "default": None})
        if kind in rule["track_types"]:
            return Assignment(parent, rule["track_types"][kind], matched)
        if stem in rule["stems"]:
            return Assignment(parent, rule["stems"][stem], matched)
        return Assignment(parent, rule["default"] or self.residual.get(parent), False)

    # -- Slakh ------------------------------------------------------------

    def _init_slakh(self) -> None:
        spec = self.spec.get("slakh") or {}
        self._slakh_drums = spec.get("drums_parent", "drums")
        self._slakh_default = dict(spec.get("default") or {})
        self._slakh_table: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        for item in spec.get("programs") or []:
            low, high = item["range"]
            raw_class = item.get("class")
            if raw_class is not None and raw_class not in self._parent_of:
                raise ValueError(f"slakh mapping names unknown class '{raw_class}'")
            parent = item.get("parent") or self._parent_of[raw_class]
            for program in range(int(low), int(high) + 1):
                self._slakh_table[program] = (parent, raw_class)

    def slakh_lookup(self, program_num: Optional[int], is_drum: bool) -> Assignment:
        if is_drum:
            return Assignment(self._slakh_drums, None, True)
        if program_num is not None and int(program_num) in self._slakh_table:
            parent, raw_class = self._slakh_table[int(program_num)]
            return Assignment(parent, raw_class, True)
        raw_class = self._slakh_default.get("class")
        parent = self._slakh_default.get("parent") or (self._parent_of.get(raw_class) if raw_class else None)
        if parent is None:
            return Assignment(None, None, False, True)
        return Assignment(parent, raw_class, False)


# ---------------------------------------------------------------------------
# Minimum-data rule
# ---------------------------------------------------------------------------

def plan_merges(
    taxonomy: Taxonomy,
    songs: Mapping[str, Set[str]],
    min_songs: Optional[int] = None,
) -> Tuple[Dict[str, str], Set[str], List[str]]:
    """Decides which candidate classes are merged.

    ``songs[candidate_class]`` is the set of song ids that contain the class
    (already restricted to the sources / split that count). Repeatedly, the
    smallest class below ``min_songs`` is merged into its ``fallback`` class
    or, failing that, its parent's residual; song sets are unioned, so a
    merge can lift its target over the threshold. Residual classes are never
    merged away. A parent is "collapsed" (not split at all) when it is left
    with fewer than two classes, when a rare class has nowhere to go, or when
    it is marked ``collapse_if_residual_rare`` and its residual, after all
    merges into it, is still below the threshold.

    Returns ``(merges, collapsed_parents, log_lines)``. Existing merges of
    ``taxonomy`` are kept.
    """
    threshold = taxonomy.min_songs if min_songs is None else int(min_songs)
    merges: Dict[str, str] = dict(taxonomy.merges)
    collapsed: Set[str] = set(taxonomy.spec.get("collapsed_parents") or [])
    log: List[str] = []
    order = {name: index for index, name in enumerate(n for names in taxonomy.candidates.values() for n in names)}

    def root(name: str) -> str:
        while name in merges:
            name = merges[name]
        return name

    def active(parent: str) -> List[str]:
        return [name for name in taxonomy.candidates[parent] if name not in merges]

    while True:
        for parent in taxonomy.candidates:
            if parent not in collapsed and len(active(parent)) < 2:
                collapsed.add(parent)
                log.append(f"parent '{parent}' has fewer than two classes left -> not split")

        counts: Dict[str, Set[str]] = {}
        for name in order:
            counts.setdefault(root(name), set()).update(songs.get(name, ()))
        low = [
            name
            for parent in taxonomy.candidates if parent not in collapsed
            for name in active(parent)
            if name != taxonomy.residual[parent] and len(counts.get(name, ())) < threshold
        ]
        if not low:
            for parent in taxonomy.candidates:
                residual = taxonomy.residual[parent]
                if (parent not in collapsed and residual is not None and taxonomy.collapse_if_residual_rare[parent]
                        and len(counts.get(residual, ())) < threshold):
                    collapsed.add(parent)
                    log.append(f"'{residual}' (residual of '{parent}'): {len(counts.get(residual, ()))} songs < "
                               f"{threshold} -> parent '{parent}' not split")
            break
        name = min(low, key=lambda item: (len(counts.get(item, ())), order[item]))
        parent = taxonomy.parent_of(name)
        target = taxonomy.fallback.get(name) or taxonomy.residual[parent]
        target = root(target) if target is not None else None
        have = len(counts.get(name, ()))
        if target is None or target == name:
            collapsed.add(parent)
            log.append(f"'{name}': {have} songs < {threshold} and no class to merge into -> parent '{parent}' not split")
        else:
            merges[name] = target
            log.append(f"'{name}': {have} songs < {threshold} -> merged into '{target}'")
    return merges, collapsed, log
