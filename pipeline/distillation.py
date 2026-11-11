import torch
import torch.nn as nn
import copy

class ConsistencyFlowDistiller(nn.Module):
    """
    Distills a pretrained Flow-Matching Teacher into a Consistency Student.
    """
    def __init__(self, teacher_model, student_model, num_scales=50):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.student = student_model
        
        # EMA Student for stable target generation
        self.ema_student = copy.deepcopy(student_model)
        self.ema_student.eval()
        for p in self.ema_student.parameters():
            p.requires_grad = False
            
        self.num_scales = num_scales

    @torch.no_grad()
    def update_ema(self, decay=0.999):
        for ema_param, student_param in zip(self.ema_student.parameters(), self.student.parameters()):
            ema_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)

    def student_target_prediction(self, model, x_t, t, stem_labels, parent_latent):
        """
        Reparameterizes the velocity model to predict x_1 directly.
        Since x_t = (1-t)x_0 + t x_1  ->  v_t = x_1 - x_0
        x_1 = x_t + (1-t) * v_t
        """
        v_pred = model(x_t, t, stem_labels, parent_latent)
        
        # t_expanded for broadcasting to (B, S, C, T_seq)
        t_exp = t.view(-1, 1, 1, 1)
        x1_pred = x_t + (1.0 - t_exp) * v_pred
        return x1_pred

    def forward(self, x1, parent_latent, stem_labels, valid_mask):
        """
        x1: Target latents (Batch, Num_Stems, Channels, Time)
        """
        B = x1.shape[0]
        device = x1.device
        
        # 1. Sample noise
        x0 = torch.randn_like(x1)
        
        # 2. Sample discrete timesteps for distillation
        indices = torch.randint(0, self.num_scales - 1, (B,), device=device)
        
        # t_{n+1} (further from data) and t_n (closer to data)
        # Note: In OT-CFM, t=1 is data, t=0 is noise.
        t_n1 = indices / self.num_scales
        t_n = (indices + 1) / self.num_scales
        
        t_n1_exp = t_n1.view(-1, 1, 1, 1)
        
        # 3. Get x at t_{n+1}
        x_n1 = (1.0 - t_n1_exp) * x0 + t_n1_exp * x1
        
        with torch.no_grad():
            # 4. Teacher Euler step from t_{n+1} to t_n
            v_teacher = self.teacher(x_n1, t_n1, stem_labels, parent_latent)
            dt = (t_n - t_n1).view(-1, 1, 1, 1)
            x_n = x_n1 + v_teacher * dt
            
            # 5. EMA Student prediction at t_n (Target)
            target_x1 = self.student_target_prediction(
                self.ema_student, x_n, t_n, stem_labels, parent_latent
            )
            
        # 6. Student prediction at t_{n+1}
        pred_x1 = self.student_target_prediction(
            self.student, x_n1, t_n1, stem_labels, parent_latent
        )
        
        # 7. Consistency Loss (L2 distance between the two predictions)
        loss = (pred_x1 - target_x1) ** 2
        loss = loss.mean(dim=[2, 3])
        
        # Mask valid stems
        masked_loss = loss * valid_mask.float()
        num_valid = valid_mask.float().sum()
        
        return masked_loss.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, requires_grad=True)