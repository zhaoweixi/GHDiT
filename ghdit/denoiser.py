import math
from functools import partial
import torch
from torch import nn, einsum
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange, reduce

def exists(value):
    return value is not None

class Residual(nn.Module):

    def __init__(self, fn):




        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):







        return self.fn(x, *args, **kwargs) + x

class SinusoidalPositionEmbeddings(nn.Module):


    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv1d(nn.Conv1d):






    def forward(self, x):
        eps = 1e-05 if x.dtype == torch.float32 else 0.001
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            scale, shift = (scale.permute(0, 2, 1), shift.permute(0, 2, 1))
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000, req_grad=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.req_grad = req_grad

    def forward(self, length):
        return Variable(self.pe[:, :length].permute(0, 2, 1), requires_grad=self.req_grad)

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_cond_dim=None, groups=8):
        super().__init__()
        self.tc_dim = time_cond_dim
        self.pe = PositionalEncoding(dim=dim) if exists(time_cond_dim) else None
        self.shift_scale = nn.Linear(in_features=time_cond_dim + dim, out_features=time_cond_dim + dim) if exists(time_cond_dim) else None
        self.dim_linear = nn.Linear(in_features=time_cond_dim + dim, out_features=dim_out * 2)
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_all=None):
        scale_shift = None
        if exists(self.pe) and exists(cond_all):
            pos_ebd = self.pe(x.shape[2])
            pos_ebd = pos_ebd.repeat(x.shape[0], 1, 1)
            pos_ebd = pos_ebd.permute(0, 2, 1)
            cond_all = torch.cat((cond_all, pos_ebd), dim=2)
            cond_all = F.silu(cond_all)
            cond_all = self.dim_linear(cond_all)
            scale_shift = cond_all.chunk(2, dim=2)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) l -> b h c l', h=self.heads), qkv)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c l -> b (h c) l', h=self.heads, l=l)
        return self.to_out(out)

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) l -> b h c l', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b (h d) l', l=l)
        return self.to_out(out)

class GatedFusion(nn.Module):




    def __init__(self, channels):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Conv1d(channels * 2, channels, 3, padding=1), nn.GroupNorm(num_groups=4, num_channels=channels), nn.ReLU(), nn.Conv1d(channels, channels, 3, padding=1), nn.Sigmoid())
        self.transform_skip = nn.Conv1d(channels, channels, 1)
        self.transform_main = nn.Conv1d(channels, channels, 1)

    def forward(self, x_main, x_skip):
        gate_input = torch.cat([x_main, x_skip], dim=1)
        gate = self.gate_net(gate_input)
        transformed_skip = self.transform_skip(x_skip)
        transformed_main = self.transform_main(x_main)
        fused = gate * transformed_skip + (1 - gate) * transformed_main
        return fused

class DW_Denoiser(nn.Module):

    def __init__(self, input_dim=3, dim_mults=(1, 2, 4, 8), time_emb_dim=128, cond_emb_dim=128, class_num=3812, unet_dim=32, len_pred_cfg=None, device=None):







        super(DW_Denoiser, self).__init__()
        self.device = device
        self.length_predictor = self.getLengthPredictor(len_pred_cfg)
        self.time_emb_dim = time_emb_dim
        self.time_embed = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.cls_ebd0 = torch.nn.Embedding(class_num, cond_emb_dim, padding_idx=0)
        self.cls_ebd1 = nn.LSTM(cond_emb_dim, cond_emb_dim, 1, batch_first=True, bidirectional=False)
        self.cls_linear = nn.Linear(cond_emb_dim, cond_emb_dim)
        self.input_proj = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        block_klass = partial(ResnetBlock, groups=8)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        dims = [32, *map(lambda m: unet_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([block_klass(dim_in, dim_in, time_cond_dim=time_emb_dim + cond_emb_dim), block_klass(dim_in, dim_in, time_cond_dim=time_emb_dim + cond_emb_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Residual(PreNorm(dim_in, LinearAttention(dim_in))), nn.Conv1d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, stride=2, padding=1)]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_cond_dim=time_emb_dim + cond_emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_cond_dim=time_emb_dim + cond_emb_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([block_klass(dim_in, dim_in, time_cond_dim=time_emb_dim + cond_emb_dim), block_klass(dim_in, dim_in, time_cond_dim=time_emb_dim + cond_emb_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Residual(PreNorm(dim_in, LinearAttention(dim_in))), GatedFusion(dim_in), nn.ConvTranspose1d(in_channels=dim_out, out_channels=dim_in, kernel_size=4, stride=2, padding=1)]))
        self.final_res_block = block_klass(dim_in, dim_in, time_cond_dim=time_emb_dim + cond_emb_dim)
        self.final_conv = nn.Conv1d(dim_in, input_dim, 1)

    def forward(self, x_t, mask, timestep, ydense, lb_mask, nodes_count):







        pred_length = self.length_predictor(ydense.to(self.device))
        pred_length = pred_length * lb_mask.cuda().to(dtype=torch.float32)
        pred_sum = torch.sum(pred_length, dim=1)
        pred_sum_ = pred_sum.unsqueeze(1)
        pred_length_scale = pred_length / pred_sum_
        scale_length = nodes_count.unsqueeze(1) * pred_length_scale
        scale_length = scale_length.floor()
        cond0_ebd = self.cls_ebd0(ydense)
        out, (hidden, cell) = self.cls_ebd1(cond0_ebd)
        new_label_embedding = self.repeat(A=out, B=scale_length, new_length=x_t.shape[1])
        c0 = self.cls_linear(new_label_embedding)
        t_emb = self.time_embed(timestep)
        new_step_t = t_emb.unsqueeze(1).repeat(1, new_label_embedding.shape[1], 1)
        cond_all = torch.cat((c0, new_step_t), dim=2)
        cond_all = cond_all.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x_t.permute(0, 2, 1)
        x = self.input_proj(x)
        c = []
        h = []
        l = []
        for block1, block2, attn1, attn2, ds in self.downs:
            x = block1(x, cond_all)
            x = attn1(x)
            x = block2(x, cond_all)
            x = attn2(x)
            h.append(x)
            l.append(x.shape[-1])
            x = ds(x)
            c.append(cond_all)
            cond_all = cond_all[:, ::2, :]
        x = self.mid_block1(x, cond_all)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond_all)
        for block1, block2, attn1, attn2, gate, us in self.ups:
            cond_all = c.pop()
            x = us(x)
            x = x[:, :, :l.pop()]
            gx = gate(x, h.pop())
            x = x + gx
            x = block1(x, cond_all)
            x = attn1(x)
            x = block2(x, cond_all)
            x = attn2(x)
        x = self.final_res_block(x, cond_all)
        out = self.final_conv(x)
        out = out.masked_fill(~mask.unsqueeze(1), 0.0)
        return out.permute(0, 2, 1)

    def getLengthPredictor(self, cfg=None):
        return cfg

    def repeat(self, A, B, new_length):











        batch_size, length, dims = A.shape
        device = A.device
        C = torch.zeros(batch_size, new_length, dims, device=device)
        for i in range(batch_size):
            idx = 0
            for j in range(length):
                repeat_times = int(B[i, j].item())
                if repeat_times == 0:
                    continue
                repeated_tensor = A[i, j:j + 1, :].repeat(repeat_times, 1)
                fill_length = min(repeated_tensor.size(0), new_length - idx)
                C[i, idx:idx + fill_length, :] = repeated_tensor[:fill_length, :]
                idx += fill_length
                if idx >= new_length:
                    break
        return C
