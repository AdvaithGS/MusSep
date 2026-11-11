import torch
import torch.nn as nn

class MaskedOTFlowMatcher(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) objective.
    Adapted for joint multi-stem generation with binary validity masking.
    """
    def __init__(self, sigma_min=1e-5):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(self, model, x1, parent_latent, stem_labels, valid_mask):
        """
        model: The JointSubStemDiT network
        x1: Target continuous latents from DAC-VAE (Batch, Num_Stems, Channels, Time)
        stem_labels: Integer IDs for the slots (Batch, Num_Stems)
        valid_mask: Boolean mask for present stems (Batch, Num_Stems)
        """
        B, S, C, T = x1.shape
        device = x1.device
        
        # 1. Sample standard normal noise for all slots
        x0 = torch.randn_like(x1)
        
        # 2. Sample timesteps t ~ U(0, 1) independently per batch item
        # Shape: (B,)
        t = torch.rand(B, device=device)
        
        # Reshape t for broadcasting to (B, 1, 1, 1)
        t_expand = t.view(B, 1, 1, 1)
        
        # 3. Construct the OT interpolation path (x_t) and target velocity (u_t)
        # x_t = (1 - t) * x0 + t * x1
        x_t = (1.0 - t_expand) * x0 + t_expand * x1
        target_v = x1 - (1.0 - self.sigma_min) * x0
        
        # NEW: Pass parent_latent to condition the velocity prediction
        pred_v = model(x_t, t, stem_labels, parent_latent)
        # 5. Compute Mean Squared Error
        # Unreduced loss shape: (B, S, C, T)
        loss = (pred_v - target_v) ** 2
        
        # Average across Channels and Time, leaving (Batch, Num_Stems)
        loss = loss.mean(dim=[2, 3])
        
        # 6. Apply validity mask to ignore absent sub-stems
        # Mask shape: (B, S) - cast to float for multiplication
        masked_loss = loss * valid_mask.float()
        
        # Average only over the valid (present) stems in the batch
        num_valid = valid_mask.float().sum()
        if num_valid > 0:
            final_loss = masked_loss.sum() / num_valid
        else:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return final_loss


class ODESolver:
    """
    Numerical solvers for integrating the learned vector field during inference.
    """
    @staticmethod
    @torch.no_grad()
    def sample_euler(model, stem_labels, shape, num_steps=50, device="cuda"):
        """
        shape: (Batch, Num_Stems, Channels, Time)
        stem_labels: (Batch, Num_Stems)
        """
        B = shape[0]
        x_t = torch.randn(shape, device=device)
        t_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), t_steps[i], device=device)
            v = model(x_t, t, stem_labels)
            x_t = x_t + v * dt
            
        return x_t

    @staticmethod
    @torch.no_grad()
    def sample_consistency(model, stem_labels, parent_latent, shape, num_steps=2, device="cuda"):
        """
        Accelerated inference for the distilled consistency model.
        num_steps is typically 1 to 4.
        """
        B = shape[0]
        # Start at noise (t=0)
        x_t = torch.randn(shape, device=device)
        t_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_current = torch.full((B,), t_steps[i], device=device)
            
            # Predict velocity
            v_pred = model(x_t, t_current, stem_labels, parent_latent)
            
            # Reparameterize to predict x_1
            t_exp = t_current.view(-1, 1, 1, 1)
            x1_pred = x_t + (1.0 - t_exp) * v_pred
            
            if i == num_steps - 1:
                # Final step returns the data prediction directly
                x_t = x1_pred
            else:
                # Intermediary steps: re-interpolate to the next timestep
                t_next = torch.full((B,), t_steps[i+1], device=device)
                t_next_exp = t_next.view(-1, 1, 1, 1)
                
                # We assume the path from pure noise to our x1_pred
                noise_estimate = (x_t - t_exp * x1_pred) / (1.0 - t_exp + 1e-5)
                x_t = (1.0 - t_next_exp) * noise_estimate + t_next_exp * x1_pred
                
        return x_t