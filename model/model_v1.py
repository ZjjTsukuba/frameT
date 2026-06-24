# -*- coding: utf-8 -*-
"""
PAF V1: frozen CLIP frame branch + PointNet point branch + simple concat fusion,
aligned to frozen CLIP TEXT embeddings of the class names (closed-set, top-1).
FROZEN: CLIP image + text encoders.  TRAINABLE: point encoder + fusion + logit scale.
This is the pipeline-validation baseline; the fusion is deliberately simple (v2 upgrades it).
New file — touches no other paper's code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """Dependency-free PointNet: shared per-point MLP -> (max,mean) pool -> MLP."""
    def __init__(self, in_dim=4, feat_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim),
        )

    def forward(self, pts):                 # pts [B,N,4]
        x = self.mlp(pts.transpose(1, 2))   # [B,256,N]
        g = torch.cat([x.max(-1).values, x.mean(-1)], dim=1)  # [B,512]
        return self.head(g)                 # [B,feat_dim]


# ---------- PointNet++ (SSG) over event (x,y,t) clouds — dependency-free ----------
def _fps(xyz, npoint, deterministic=False):              # xyz [B,N,3] -> idx [B,npoint]
    B, N, _ = xyz.shape
    dev = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev)
    far = (torch.zeros(B, dtype=torch.long, device=dev) if deterministic   # fixed start -> deterministic eval
           else torch.randint(0, N, (B,), device=dev))
    ar = torch.arange(B, device=dev)
    for i in range(npoint):
        idx[:, i] = far
        d = ((xyz - xyz[ar, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.max(-1).indices
    return idx


def _index(points, idx):                                 # points [B,N,C], idx [B,...] -> [B,...,C]
    B = points.shape[0]
    vshape = [B] + [1] * (idx.dim() - 1)
    bidx = torch.arange(B, device=points.device).view(vshape).expand_as(idx)
    return points[bidx, idx]


class SAModule(nn.Module):
    """Set Abstraction: FPS centroids -> KNN group -> relative coords -> mini-PointNet -> max."""
    def __init__(self, npoint, K, in_ch, mlp):
        super().__init__()
        self.npoint, self.K = npoint, K
        layers, last = [], in_ch + 3                     # +3 relative (Δx,Δy,Δt)
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat):                        # xyz [B,N,3], feat [B,N,C]
        new_xyz = _index(xyz, _fps(xyz, self.npoint, deterministic=not self.training))    # [B,M,3]
        knn = torch.cdist(new_xyz, xyz).topk(self.K, dim=-1, largest=False).indices  # [B,M,K]
        rel = _index(xyz, knn) - new_xyz.unsqueeze(2)    # relative coords [B,M,K,3] (local motion)
        g = torch.cat([rel, _index(feat, knn)], dim=-1)  # [B,M,K,3+C]
        g = self.mlp(g.permute(0, 3, 1, 2)).max(-1).values  # [B,out,M]
        return new_xyz, g.transpose(1, 2)                # [B,M,3], [B,M,out]


class PointNet2Encoder(nn.Module):
    """3-stage PointNet++ SSG on (x,y,t); polarity as input feature."""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.sa1 = SAModule(512, 24, 1, [64, 64, 128])
        self.sa2 = SAModule(128, 24, 128, [128, 128, 256])
        self.sa3 = SAModule(32, 24, 256, [256, 256, 512])
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))

    def forward(self, pts, return_tokens=False):         # pts [B,N,4] = (x,y,t,p)
        xyz, feat = pts[:, :, :3].contiguous(), pts[:, :, 3:4].contiguous()
        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        xyz, feat = self.sa3(xyz, feat)
        if return_tokens:
            return feat                                  # [B,M,512] motion tokens
        return self.head(feat.max(1).values)             # global max -> [B,feat_dim]


# ---------- + EventMamba-style explicit temporal (Mamba over time-ordered groups) ----------
class TemporalMamba(nn.Module):
    """Sort groups by centroid time, model the sequence -> explicit temporal/motion."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            from mamba_ssm import Mamba
            self.seq, self.kind = Mamba(d_model=dim), "mamba"
        except Exception as e:                           # robust fallback
            self.seq, self.kind = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True), "gru"
            print(f"[TemporalMamba] mamba_ssm unavailable ({e}); using BiGRU")

    def forward(self, xyz, feat):                        # xyz [B,M,3], feat [B,M,C]
        order = xyz[:, :, 2].argsort(dim=1)              # chronological by t
        xyz, feat = _index(xyz, order), _index(feat, order)
        y = self.norm(feat)
        y = self.seq(y)[0] if self.kind == "gru" else self.seq(y)
        return xyz, feat + y                             # residual


class PointMambaEncoder(nn.Module):
    """PointNet++ local groups + Mamba explicit temporal (EventMamba-style GlobalFE)."""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.sa1 = SAModule(512, 24, 1, [64, 64, 128]);  self.tm1 = TemporalMamba(128)
        self.sa2 = SAModule(128, 24, 128, [128, 128, 256]); self.tm2 = TemporalMamba(256)
        self.sa3 = SAModule(32, 24, 256, [256, 256, 512])
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))

    def forward(self, pts, return_tokens=False):         # pts [B,N,4]=(x,y,t,p)
        xyz, feat = pts[:, :, :3].contiguous(), pts[:, :, 3:4].contiguous()
        xyz, feat = self.tm1(*self.sa1(xyz, feat))
        xyz, feat = self.tm2(*self.sa2(xyz, feat))
        xyz, feat = self.sa3(xyz, feat)
        if return_tokens:
            return feat                                  # [B,M,512] motion tokens
        return self.head(feat.max(1).values)


class SeqMamba(nn.Module):
    """One temporal SSM block over a pre-ordered sequence [B,L,dim] (no sorting). Mamba; BiGRU fallback; residual."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            from mamba_ssm import Mamba
            self.seq, self.kind = Mamba(d_model=dim), "mamba"
        except Exception as e:
            self.seq, self.kind = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True), "gru"
            print(f"[SeqMamba] mamba_ssm unavailable ({e}); using BiGRU")

    def forward(self, x):                                          # [B,L,dim]
        y = self.norm(x)
        y = self.seq(y)[0] if self.kind == "gru" else self.seq(y)
        return x + y


class TemporalPointEncoder(nn.Module):
    """Direction-A strong point encoder: sort events by t → K fine time-slices → per-slice spatial token
    (shared MLP + segment-max) → deep Mamba over the K-length time sequence. Targets the INTRA-WINDOW
    temporal structure the accumulated frame histogram erases (K=32 ≫ frame T=8). New design."""
    def __init__(self, feat_dim=512, n_slices=32, dim=512):
        super().__init__()
        self.K = n_slices
        self.embed = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(256, dim)
        self.norm_in = nn.LayerNorm(dim)
        self.tm1, self.tm2 = SeqMamba(dim), SeqMamba(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, feat_dim))

    def forward(self, pts, return_tokens=False):                   # pts [B,N,4]=(x,y,t,p)
        B, N, _ = pts.shape
        t = pts[:, :, 2]
        f = self.embed(pts.transpose(1, 2)).transpose(1, 2)        # [B,N,256] per-point feature
        bidx = (t * self.K).clamp(0, self.K - 1).long()            # [B,N] time-slice index
        tok = pts.new_zeros(B, self.K, 256)
        for k in range(self.K):                                    # K small; segment-max per time slice
            m = (bidx == k).unsqueeze(-1)                          # [B,N,1]
            has = m.squeeze(-1).any(dim=1, keepdim=True)           # [B,1] non-empty slice?
            mk = f.masked_fill(~m, float("-inf")).max(dim=1).values  # [B,256]
            tok[:, k] = torch.where(has, mk, torch.zeros_like(mk))
        x = self.norm_in(self.proj(tok))                           # [B,K,dim]
        x = self.tm2(self.tm1(x))                                  # temporal SSM over time slices
        if return_tokens:
            return x                                               # [B,K,dim] motion tokens
        return self.head(x.max(1).values)                         # [B,feat_dim]


class CrossAttnFusion(nn.Module):
    """Frame (appearance) tokens query Point (motion) tokens -> inject motion into frame."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.nq, self.nkv, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, q_tok, kv_tok):                    # [B,T,dim],[B,M,dim] -> [B,dim]
        kv = self.nkv(kv_tok)
        a, _ = self.attn(self.nq(q_tok), kv, kv)
        x = q_tok + a                                    # motion injected into each frame token
        x = x + self.ffn(self.n2(x))
        return x.mean(1)                                 # pool over frame tokens


class PAFClipPoint(nn.Module):
    def __init__(self, class_prompts, clip_name="openai/clip-vit-base-patch32", point_dim=512,
                 point_encoder="pointnet2", fusion="concat"):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.tok = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        dim = self.clip.config.projection_dim          # 512 for ViT-B/32

        if point_encoder == "pointmamba":
            self.point = PointMambaEncoder(point_dim)
        elif point_encoder == "pointnet2":
            self.point = PointNet2Encoder(point_dim)
        else:
            self.point = PointNetEncoder(4, point_dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim + point_dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim),
        )
        self.fusion = fusion
        self.fusion_mod = CrossAttnFusion(dim) if fusion == "cross" else None
        self.logit_scale = nn.Parameter(torch.tensor(2.659438))  # ln(1/0.07), like CLIP

        with torch.no_grad():                            # frozen text prototypes
            tk = self.tok(class_prompts, padding=True, return_tensors="pt")
            tf = F.normalize(self.clip.get_text_features(**tk), dim=-1)
        self.register_buffer("text_feats", tf)           # [C,dim]

    @torch.no_grad()
    def _frame_feat(self, frames, pool=True):            # [B,T,3,H,W] -> [B,dim] or [B,T,dim] (frozen)
        B, T = frames.shape[:2]
        f = self.clip.get_image_features(pixel_values=frames.flatten(0, 1)).view(B, T, -1)
        return f.mean(1) if pool else f

    def forward(self, frames, points, branch="both"):
        if self.fusion == "cross" and branch != "point":
            f_tok = self._frame_feat(frames, pool=False)          # [B,T,dim] frozen appearance tokens
            if branch == "frame":
                fused = f_tok.mean(1)                             # frame-only
            else:
                p_tok = self.point(points, return_tokens=True)    # [B,M,dim] motion tokens
                fused = self.fusion_mod(f_tok, p_tok)             # motion injected into frame via x-attn
        else:                                                     # concat fusion (also point-only)
            f_point = self.point(points)
            if branch == "point":                                 # drop frame, skip CLIP
                f_frame = points.new_zeros(points.shape[0], self.text_feats.shape[1])
            else:
                f_frame = self._frame_feat(frames)
                if branch == "frame":
                    f_point = torch.zeros_like(f_point)
            fused = self.fuse(torch.cat([f_frame, f_point], dim=1))
        fused = F.normalize(fused, dim=-1)
        return self.logit_scale.exp() * fused @ self.text_feats.t()   # [B,C]
