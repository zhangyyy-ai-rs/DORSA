import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from mmcv.cnn import build_norm_layer


class Stem(nn.Module):
    def __init__(self, in_chans=3, out_ch=48, norm_layer=dict(type='BN', requires_grad=True)):
        super().__init__()
        mid = max(out_ch // 2, 24)

        self.conv1 = nn.Conv2d(in_chans, mid, 3, stride=2, padding=1, groups=1, bias=False)
        self.norm1 = build_norm_layer(norm_layer, mid)[1]
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(mid, mid, 3, stride=1, padding=1, groups=mid, bias=False)
        self.norm2 = build_norm_layer(norm_layer, mid)[1]
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(mid, out_ch, 3, stride=2, padding=1, groups=1, bias=False)
        self.norm3 = build_norm_layer(norm_layer, out_ch)[1]
        self.act3 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=dict(type='BN', requires_grad=True)):
        super().__init__()

        self.path1_conv1 = nn.Conv2d(in_dim, in_dim, 3, stride=2, padding=1, groups=in_dim, bias=False)
        self.path1_norm1 = build_norm_layer(norm_layer, in_dim)[1]
        self.path1_act1 = nn.GELU()
        self.path1_conv2 = nn.Conv2d(in_dim, out_dim // 2, 1, stride=1, padding=0, groups=1, bias=False)
        self.path1_norm2 = build_norm_layer(norm_layer, out_dim // 2)[1]
        self.path1_act2 = nn.GELU()

        self.path2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.path2_conv = nn.Conv2d(in_dim, out_dim // 2, 1, stride=1, padding=0, groups=1, bias=False)
        self.path2_norm = build_norm_layer(norm_layer, out_dim // 2)[1]
        self.path2_act = nn.GELU()

        self.fuse_conv = nn.Conv2d(out_dim, out_dim, 1, stride=1, padding=0, groups=1, bias=False)
        self.fuse_norm = build_norm_layer(norm_layer, out_dim)[1]
        self.fuse_act = nn.GELU()

    def forward(self, x):
        p1 = self.path1_act1(self.path1_norm1(self.path1_conv1(x)))
        p1 = self.path1_act2(self.path1_norm2(self.path1_conv2(p1)))

        p2 = self.path2_pool(x)
        p2 = self.path2_act(self.path2_norm(self.path2_conv(p2)))

        x = torch.cat([p1, p2], dim=1)
        x = self.fuse_act(self.fuse_norm(self.fuse_conv(x)))
        return x


class StateDescriptor(nn.Module):
    def __init__(self, dim, hidden=32, norm_layer=dict(type='BN', requires_grad=True)):
        super().__init__()

        self.proj_conv1 = nn.Conv2d(dim, hidden, 1, stride=1, padding=0, groups=1, bias=False)
        self.proj_norm1 = build_norm_layer(norm_layer, hidden)[1]
        self.proj_act1 = nn.GELU()
        self.proj_conv2 = nn.Conv2d(hidden, hidden, 3, stride=1, padding=1, groups=hidden, bias=False)
        self.proj_norm2 = build_norm_layer(norm_layer, hidden)[1]
        self.proj_act2 = nn.GELU()

        self.mix_conv = nn.Conv2d(hidden + 4, hidden, 1, stride=1, padding=0, groups=1, bias=False)
        self.mix_norm = build_norm_layer(norm_layer, hidden)[1]
        self.mix_act = nn.GELU()
        self.mix_out = nn.Conv2d(hidden, 4, kernel_size=1, bias=True)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        lap = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)
        self.register_buffer('lap', lap, persistent=False)

    def forward(self, x):
        proxy = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(proxy, self.sobel_x, padding=1)
        gy = F.conv2d(proxy, self.sobel_y, padding=1)
        rough = torch.abs(F.conv2d(proxy, self.lap, padding=1))
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        anis = torch.abs(gx) - torch.abs(gy)
        reliab = torch.exp(-torch.abs(proxy - F.avg_pool2d(proxy, 3, 1, 1)))
        stats = torch.cat([rough, edge, anis, reliab], dim=1)

        feat = self.proj_act1(self.proj_norm1(self.proj_conv1(x)))
        feat = self.proj_act2(self.proj_norm2(self.proj_conv2(feat)))

        y = torch.cat([feat, stats], dim=1)
        y = self.mix_act(self.mix_norm(self.mix_conv(y)))
        return self.mix_out(y)


class PrimitiveOperatorBank(nn.Module):
    def __init__(self, hidden_dim, norm_layer=dict(type='BN', requires_grad=True), strip_size=7):
        super().__init__()
        p = strip_size // 2

        self.local_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False)
        self.local_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.local_act = nn.GELU()

        self.hstrip_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(1, strip_size), padding=(0, p), groups=hidden_dim, bias=False
        )
        self.hstrip_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.hstrip_act = nn.GELU()

        self.vstrip_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=(strip_size, 1), padding=(p, 0), groups=hidden_dim, bias=False
        )
        self.vstrip_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.vstrip_act = nn.GELU()

        self.dilate_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=1, padding=2, dilation=2, groups=hidden_dim, bias=False
        )
        self.dilate_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.dilate_act = nn.GELU()

        self.smooth_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.smooth_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, stride=1, padding=0, groups=1, bias=False)
        self.smooth_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.smooth_act = nn.GELU()

        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.context_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.context_act = nn.GELU()
        self.num_bases = 6

    def forward(self, z):
        b0 = self.local_act(self.local_norm(self.local_conv(z)))
        b1 = self.hstrip_act(self.hstrip_norm(self.hstrip_conv(z)))
        b2 = self.vstrip_act(self.vstrip_norm(self.vstrip_conv(z)))
        b3 = self.dilate_act(self.dilate_norm(self.dilate_conv(z)))

        b4 = self.smooth_pool(z)
        b4 = self.smooth_act(self.smooth_norm(self.smooth_conv(b4)))

        ctx = self.context_pool(z)
        ctx = self.context_act(self.context_conv(ctx))
        b5 = ctx.expand_as(z)
        return [b0, b1, b2, b3, b4, b5]


class OperatorAtomMixer(nn.Module):
    def __init__(self, hidden_dim, num_bases=6, num_atoms=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bases = num_bases
        self.num_atoms = num_atoms
        self.atom_basis_logits = nn.Parameter(torch.zeros(num_atoms, num_bases))
        self.atom_channel_scale = nn.Parameter(torch.zeros(num_atoms, hidden_dim))
        self.atom_channel_bias = nn.Parameter(torch.zeros(num_atoms, hidden_dim))
        trunc_normal_(self.atom_basis_logits, std=0.02)
        trunc_normal_(self.atom_channel_scale, std=0.02)
        nn.init.zeros_(self.atom_channel_bias)

    def forward(self, bases: List[torch.Tensor]):
        base_stack = torch.stack(bases, dim=1)  # [B, M, C, H, W]
        coeff = torch.softmax(self.atom_basis_logits, dim=1)  # [K, M]
        atoms = torch.einsum('km,bmchw->bkchw', coeff, base_stack)  # [B, K, C, H, W]
        scale = self.atom_channel_scale.view(1, self.num_atoms, self.hidden_dim, 1, 1)
        bias = self.atom_channel_bias.view(1, self.num_atoms, self.hidden_dim, 1, 1)
        atoms = atoms * (1.0 + scale) + bias
        return atoms


class SparseRouter(nn.Module):
    def __init__(self, dim, num_atoms=8, topk=2, route_ratio=4,
                 norm_layer=dict(type='BN', requires_grad=True), temperature=1.0):
        super().__init__()
        self.topk = topk
        self.temperature = temperature
        hidden = max(dim // route_ratio, 16)
        semantic_hidden = hidden
        desc_hidden = min(hidden, 32)

        self.desc_head = StateDescriptor(dim, hidden=desc_hidden, norm_layer=norm_layer)

        self.semantic_conv1 = nn.Conv2d(dim, semantic_hidden, 1, stride=1, padding=0, groups=1, bias=False)
        self.semantic_norm1 = build_norm_layer(norm_layer, semantic_hidden)[1]
        self.semantic_act1 = nn.GELU()
        self.semantic_conv2 = nn.Conv2d(
            semantic_hidden, semantic_hidden, 3, stride=1, padding=1, groups=semantic_hidden, bias=False
        )
        self.semantic_norm2 = build_norm_layer(norm_layer, semantic_hidden)[1]
        self.semantic_act2 = nn.GELU()

        self.to_logits = nn.Conv2d(semantic_hidden + 4, num_atoms, kernel_size=1, bias=True)

    def forward(self, x):
        desc = self.desc_head(x)
        sem = self.semantic_act1(self.semantic_norm1(self.semantic_conv1(x)))
        sem = self.semantic_act2(self.semantic_norm2(self.semantic_conv2(sem)))
        logits = self.to_logits(torch.cat([sem, desc], dim=1)).clamp(-8, 8)
        if self.topk < logits.shape[1]:
            topv, topi = torch.topk(logits, k=self.topk, dim=1)
            sparse = torch.full_like(logits, float('-inf'))
            sparse.scatter_(1, topi, topv)
            logits = sparse
        weights = torch.softmax(logits / self.temperature, dim=1)
        return weights, desc


class DynamicRouteBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, num_atoms=8, topk=2, hidden_ratio=0.5,
                 drop=0.0, drop_path=0.0, norm_layer=dict(type='BN', requires_grad=True),
                 temperature=1.0):
        super().__init__()
        hidden_dim = max(int(dim * hidden_ratio), 24)

        self.pre_conv = nn.Conv2d(dim, hidden_dim, 1, stride=1, padding=0, groups=1, bias=False)
        self.pre_norm = build_norm_layer(norm_layer, hidden_dim)[1]
        self.pre_act = nn.GELU()

        self.bank = PrimitiveOperatorBank(hidden_dim, norm_layer=norm_layer)
        self.mixer = OperatorAtomMixer(hidden_dim, num_bases=self.bank.num_bases, num_atoms=num_atoms)
        self.router = SparseRouter(dim, num_atoms=num_atoms, topk=topk, norm_layer=norm_layer, temperature=temperature)

        self.project_conv = nn.Conv2d(hidden_dim, dim, 1, stride=1, padding=0, groups=1, bias=False)
        self.project_norm = build_norm_layer(norm_layer, dim)[1]
        self.project_act = nn.Identity()

        self.local_guard_conv = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False)
        self.local_guard_norm = build_norm_layer(norm_layer, dim)[1]
        self.local_guard_act = nn.Identity()

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden = int(dim * mlp_ratio)
        self.ffn_conv1 = nn.Conv2d(dim, mlp_hidden, 1, stride=1, padding=0, groups=1, bias=False)
        self.ffn_norm1 = build_norm_layer(norm_layer, mlp_hidden)[1]
        self.ffn_act1 = nn.GELU()
        self.ffn_conv2 = nn.Conv2d(
            mlp_hidden, mlp_hidden, 3, stride=1, padding=1, groups=mlp_hidden, bias=False
        )
        self.ffn_norm2 = build_norm_layer(norm_layer, mlp_hidden)[1]
        self.ffn_act2 = nn.GELU()
        self.ffn_conv3 = nn.Conv2d(mlp_hidden, dim, kernel_size=1, bias=False)
        self.ffn_norm3 = build_norm_layer(norm_layer, dim)[1]
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _project(self, x):
        return self.project_act(self.project_norm(self.project_conv(x)))

    def _local_guard(self, x):
        return self.local_guard_act(self.local_guard_norm(self.local_guard_conv(x)))

    def _ffn(self, x):
        x = self.ffn_act1(self.ffn_norm1(self.ffn_conv1(x)))
        x = self.ffn_act2(self.ffn_norm2(self.ffn_conv2(x)))
        x = self.ffn_norm3(self.ffn_conv3(x))
        return x

    def forward(self, x):
        weights, desc = self.router(x)
        z = self.pre_act(self.pre_norm(self.pre_conv(x)))
        bases = self.bank(z)
        atoms = self.mixer(bases)  # [B, K, C, H, W]
        mixed = torch.einsum('bkhw,bkchw->bchw', weights, atoms)

        rough, edge, anis, reliab = torch.chunk(desc, 4, dim=1)
        geo_gain = torch.sigmoid(edge + rough + 0.5 * torch.abs(anis) + reliab)
        mixed = mixed * geo_gain

        out = self._project(mixed)
        out = self.beta * self._local_guard(x) + torch.tanh(self.alpha) * out
        x = x + self.drop_path(out)
        x = x + self.drop_path(torch.tanh(self.gamma) * self._ffn(x))
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class CleanStateEncoder(nn.Module):
    def __init__(self, dim, hidden_ratio=1.5):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.norm = LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, hidden, 1, bias=False)
        self.local3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.local5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.mix_reduce = nn.Conv2d(hidden * 2, hidden, 1, bias=False)
        self.mix_act = nn.GELU()
        self.mix_out = nn.Conv2d(hidden, dim, 1, bias=False)
        self.gate = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        z = self.expand(self.norm(x))
        a = self.local3(z)
        b = self.local5(z)
        y = torch.cat([a, b], dim=1)
        y = self.mix_reduce(y)
        y = self.mix_act(y)
        y = self.mix_out(y)
        return y * torch.sigmoid(self.gate(y) + 1.0)


class GeometryStateEncoder(nn.Module):
    def __init__(self, dim, hidden_ratio=1.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.norm = LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, hidden, 1, bias=False)

        self.hstrip = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden, bias=False)
        self.vstrip = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden, bias=False)
        self.dil = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False)

        self.mix_reduce = nn.Conv2d(hidden * 4, hidden, 1, bias=False)
        self.mix_act = nn.GELU()
        self.mix_out = nn.Conv2d(hidden, dim, 1, bias=False)

    @staticmethod
    def _diag_mix(x):
        return (x
                + torch.roll(x, shifts=(1, 1), dims=(2, 3))
                + torch.roll(x, shifts=(-1, -1), dims=(2, 3))
                + torch.roll(x, shifts=(1, -1), dims=(2, 3))
                + torch.roll(x, shifts=(-1, 1), dims=(2, 3))) / 5.0

    def forward(self, x):
        z = self.expand(self.norm(x))
        h = self.hstrip(z)
        v = self.vstrip(z)
        d = self._diag_mix(self.dil(z))

        avg3 = F.avg_pool2d(z, 3, stride=1, padding=1)
        peak = z - avg3
        y = torch.cat([h, v, d, peak], dim=1)
        y = self.mix_reduce(y)
        y = self.mix_act(y)
        y = self.mix_out(y)
        return y


class OrthogonalStateMixer(nn.Module):
    def __init__(self, dim, clean_bias=1.4):
        super().__init__()
        self.clean_bias = nn.Parameter(torch.tensor(float(clean_bias)))
        self.geo_bias = nn.Parameter(torch.tensor(0.0))

        self.ctrl_reduce = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.ctrl_act = nn.GELU()
        self.ctrl_score = nn.Conv2d(dim, 2, 1, bias=True)

        self.out_reduce = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        nn.init.zeros_(self.ctrl_score.weight)
        nn.init.zeros_(self.ctrl_score.bias)

    def _orthogonalize(self, g, c):
        num = (g * c).sum(1, keepdim=True)
        den = (c * c).sum(1, keepdim=True) + 1e-6
        return g - num / den * c

    def forward(self, clean, geom):
        geom_inc = self._orthogonalize(geom, clean)
        pooled = F.adaptive_avg_pool2d(torch.cat([clean, geom_inc], dim=1), 1)
        delta = self.ctrl_reduce(pooled)
        delta = self.ctrl_act(delta)
        delta = self.ctrl_score(delta).flatten(1)
        w = torch.softmax(torch.stack([
            self.clean_bias.expand_as(delta[:, 0]),
            self.geo_bias.expand_as(delta[:, 1])
        ], dim=1) + delta, dim=1)

        c = clean * w[:, 0:1, None, None]
        g = geom_inc * w[:, 1:2, None, None]
        out = torch.cat([c, g], dim=1)
        out = self.out_reduce(out)
        out = self.out_act(out)
        return self.out_proj(out)


class DualStateOrthogonalCell(nn.Module):
    def __init__(self, dim, alpha_init=-3.0, clean_ratio=1.5, geo_ratio=1.0):
        super().__init__()
        self.clean = CleanStateEncoder(dim, hidden_ratio=clean_ratio)
        self.geo = GeometryStateEncoder(dim, hidden_ratio=geo_ratio)
        self.mix = OrthogonalStateMixer(dim)

        self.alpha = nn.Parameter(torch.full((1, dim, 1, 1), alpha_init))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        c = self.clean(x)
        g = self.geo(x)
        delta = self.mix(c, g)

        alpha = torch.sigmoid(self.alpha)
        beta = torch.tanh(self.beta) * 0.05
        return x + alpha * delta + beta * c


class DualPathDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        conv_dim = out_dim * 3 // 4
        pool_dim = out_dim - conv_dim

        self.conv_dw = nn.Conv2d(in_dim, in_dim, 3, stride=2, padding=1, groups=in_dim, bias=False)
        self.conv_pw = nn.Conv2d(in_dim, conv_dim, 1, bias=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_pw = nn.Conv2d(in_dim, pool_dim, 1, bias=False)

        self.fuse_norm = LayerNorm2d(out_dim)
        self.fuse_conv = nn.Conv2d(out_dim, out_dim, 1, bias=False)

    def forward(self, x):
        conv_feat = self.conv_pw(self.conv_dw(x))
        pool_feat = self.pool_pw(self.pool(x))
        y = torch.cat([conv_feat, pool_feat], dim=1)
        y = self.fuse_norm(y)
        return self.fuse_conv(y)


class DORSANet(nn.Module):
    def __init__(
        self,
        in_chans=3,
        stem_dim=56,
        depths=(2, 2, 8, 2),
        mlp_ratio=2.0,
        hidden_ratios=(0.5, 0.5, 0.5, 0.5),
        num_atoms=(4, 4, 8, 8),
        topks=(1, 1, 2, 2),
        drop_path_rate=0.1,
        fork_feat=True,
        temperature=1.25,
        norm_layer=dict(type='SyncBN', requires_grad=True),
        init_cfg=None,
        pretrained=None,
        use_stage1_cell=True,
        use_stage2_cell=True,
        stage1_position='mid',
        stage2_position='pre',
    ):
        super().__init__()

        self.fork_feat = fork_feat
        self.init_cfg = init_cfg
        self.pretrained = pretrained
        self.num_stages = len(depths)

        dims = [stem_dim * 2 ** i for i in range(self.num_stages)]
        self.dims = dims

        self.stem = Stem(in_chans, dims[0], norm_layer=norm_layer)

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        stages = []
        for i in range(self.num_stages):
            blocks = nn.ModuleList()
            for _ in range(depths[i]):
                blocks.append(
                    DynamicRouteBlock(
                        dim=dims[i],
                        mlp_ratio=mlp_ratio,
                        num_atoms=num_atoms[i],
                        topk=topks[i],
                        hidden_ratio=hidden_ratios[i],
                        drop_path=dp_rates[cur],
                        norm_layer=norm_layer,
                        temperature=temperature,
                    )
                )
                cur += 1
            stages.append(blocks)
        self.stages = nn.ModuleList(stages)

        # Keep the same effective stage transitions used by vx.py.
        self.stage2_down = DualPathDownsample(dims[0], dims[1])
        self.down2 = Downsample(dims[1], dims[2], norm_layer=norm_layer)
        self.down3 = Downsample(dims[2], dims[3], norm_layer=norm_layer)

        self.use_stage1_cell = use_stage1_cell
        self.use_stage2_cell = use_stage2_cell
        self.stage1_position = stage1_position
        self.stage2_position = stage2_position

        self.stage1_cell = DualStateOrthogonalCell(
            dims[0], alpha_init=-3.0, clean_ratio=1.5, geo_ratio=1.0
        ) if use_stage1_cell else nn.Identity()
        self.stage2_cell = DualStateOrthogonalCell(
            dims[1], alpha_init=-3.4, clean_ratio=1.35, geo_ratio=0.85
        ) if use_stage2_cell else nn.Identity()

        if not self.fork_feat:
            self.out_norm = build_norm_layer(norm_layer, dims[-1])[1]
            self.out_head = nn.Linear(dims[-1], 1000)

        # Match vx.py behavior: trunk is trunc_normal-initialized,
        # while stage cells and custom stage2_down keep PyTorch defaults.
        self._init_trunk_weights()
        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def _init_trunk_weights(self):
        for module in [self.stem, self.stages, self.down2, self.down3]:
            for m in module.modules():
                self._init_weights(m)

        if not self.fork_feat:
            for module in [self.out_norm, self.out_head]:
                for m in module.modules():
                    self._init_weights(m)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            try:
                from mmdet.utils import get_root_logger
                from mmcv.runner import _load_checkpoint
            except Exception:
                return

            logger = get_root_logger()
            ckpt = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            logger.info(f'DORSANet load pretrained. missing={len(missing)}, unexpected={len(unexpected)}')

    @staticmethod
    def _forward_stage(x, stage):
        for block in stage:
            x = block(x)
        return x

    @classmethod
    def _run_with_cell(cls, x, stage, cell, mode='mid'):
        if mode == 'pre':
            x = cell(x)
            x = cls._forward_stage(x, stage)
            return x
        if mode == 'post':
            x = cls._forward_stage(x, stage)
            x = cell(x)
            return x
        if len(stage) > 0:
            x = stage[0](x)
            x = cell(x)
            for blk in list(stage)[1:]:
                x = blk(x)
            return x
        return cell(x)

    def forward_features(self, x):
        outs = []

        x = self.stem(x)

        if self.use_stage1_cell:
            x = self._run_with_cell(x, self.stages[0], self.stage1_cell, self.stage1_position)
        else:
            x = self._forward_stage(x, self.stages[0])
        outs.append(x)

        x = self.stage2_down(x)
        if self.use_stage2_cell:
            x = self._run_with_cell(x, self.stages[1], self.stage2_cell, self.stage2_position)
        else:
            x = self._forward_stage(x, self.stages[1])
        outs.append(x)

        x = self.down2(x)
        x = self._forward_stage(x, self.stages[2])
        outs.append(x)

        x = self.down3(x)
        x = self._forward_stage(x, self.stages[3])
        outs.append(x)

        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        if self.fork_feat:
            return outs
        x = outs[-1]
        x = self.out_norm(x)
        x = x.mean(dim=(2, 3))
        return self.out_head(x)


def DORSA_T_2262_s48(**kwargs):
    cfg = dict(
        in_chans=3,
        stem_dim=48,
        depths=(2, 2, 6, 2),
        mlp_ratio=2.0,
        hidden_ratios=(0.5, 0.5, 0.5, 0.5),
        num_atoms=(4, 4, 8, 8),
        topks=(1, 1, 2, 2),
        drop_path_rate=0.1,
        fork_feat=False,
        temperature=1.25,
        norm_layer=dict(type='SyncBN', requires_grad=True),
        init_cfg=None,
        pretrained=None,
        use_stage1_cell=True,
        use_stage2_cell=True,
        stage1_position='mid',
        stage2_position='pre',
    )
    cfg.update(kwargs)
    return DORSANet(**cfg)

