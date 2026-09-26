"""Inference-only adapter for the original eight-step sampler."""
import torch
from torch import nn
from .denoiser import DW_Denoiser
from .length_predictor import LengthRegulator


class Generator(nn.Module):
    def __init__(self, weights):
        super().__init__()
        length = LengthRegulator(64, 0.25, 4, 3812).cuda().float()
        self.denoise_fn = DW_Denoiser(input_dim=2, unet_dim=64, dim_mults=[1,2,4],
            time_emb_dim=128, cond_emb_dim=128, class_num=3812, len_pred_cfg=length)
        state = torch.load(weights/'ghdit_ep029.pth', map_location='cpu', weights_only=True)
        if not any(k.startswith('denoise_fn.length_predictor.') for k in state):
            raise ValueError('Checkpoint lacks embedded length predictor')
        # Preserve checkpoint diffusion buffers exactly, without recomputing schedules.
        for name, value in state.items():
            if not name.startswith('denoise_fn.'):
                if '.' in name:
                    raise ValueError('Unexpected checkpoint key: '+name)
                self.register_buffer(name, torch.empty_like(value))
        self.load_state_dict(state, strict=True)
        self.cuda().eval().requires_grad_(False)
        if int(self.timesteps.item()) != 8:
            raise ValueError('Expected 8 diffusion steps')

    @torch.inference_mode()
    def predict_length(self, labels):
        return self.denoise_fn.length_predictor(labels).sum().item()

    @torch.inference_mode()
    def generate(self, labels, points, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x=torch.randn((1,points,2),device='cuda')
        mask=torch.ones((1,points),dtype=torch.bool,device='cuda')
        lb_mask=torch.ones_like(labels,dtype=torch.bool)
        counts=torch.tensor([points],dtype=torch.float32,device='cuda')
        for i in reversed(range(int(self.timesteps.item()))):
            t=torch.full((1,),i,device='cuda',dtype=torch.long)
            x0=self.denoise_fn(x,mask,t,labels,lb_mask,counts)
            def extract(v):
                return v.gather(-1,t).reshape(1,1,1)
            mean=extract(self.posterior_mean_coef1)*x0+extract(self.posterior_mean_coef2)*x
            noise=torch.randn(x.shape,device=x.device)
            nonzero=(1-(t==0).float()).reshape(1,1,1)
            x=mean+nonzero*(0.5*extract(self.posterior_log_variance_clipped)).exp()*noise
        if not torch.isfinite(x).all():
            raise ValueError('Generated trajectory contains NaN/Inf')
        return x[0].cpu().numpy()
