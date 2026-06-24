# -*- coding: utf-8 -*-
"""
PAF V2: ExACT-style CoOp learnable text prompt on HF transformers CLIP (SAME backbone as V1)
+ point branch + cross-attn fusion (reused from V1).
  text='coop' -> [BOS][n_ctx learnable ctx][class name .][EOS], CLIP text transformer frozen.
  text='hand' -> fixed prompts (== V1, for ablation).
FROZEN: CLIP image + text.  TRAINABLE: prompt ctx + point encoder + fusion.
New file.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

from paf.model_v1 import PointMambaEncoder, PointNet2Encoder, PointNetEncoder, CrossAttnFusion, TemporalPointEncoder


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update: W·x + scale·B(A·x). B=0 at init."""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero -> initial delta = 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def add_lora_to_clip_vision(clip_model, r=8, alpha=16):
    """Inject LoRA into every ViT self-attention q/k/v/out projection (adapts ViT to event frames)."""
    for layer in clip_model.vision_model.encoder.layers:
        a = layer.self_attn
        a.q_proj = LoRALinear(a.q_proj, r, alpha)
        a.k_proj = LoRALinear(a.k_proj, r, alpha)
        a.v_proj = LoRALinear(a.v_proj, r, alpha)
        a.out_proj = LoRALinear(a.out_proj, r, alpha)


class DeepFusion(nn.Module):
    """Joint multimodal transformer: [CLS]+frame tokens+point tokens, bidirectional self-attn, take CLS."""
    def __init__(self, dim, depth=2, heads=8):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_emb = nn.Parameter(torch.zeros(3, dim))     # 0=cls 1=frame 2=point
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True,
                                           activation="gelu", dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, f_tok, p_tok):                          # [B,T,d],[B,M,d] -> [B,d]
        B = f_tok.shape[0]
        cls = self.cls.expand(B, -1, -1) + self.type_emb[0]
        x = torch.cat([cls, f_tok + self.type_emb[1], p_tok + self.type_emb[2]], dim=1)
        return self.enc(x)[:, 0]                              # CLS token


class GatedCrossAttnFusion(nn.Module):
    """Motion injected into appearance via cross-attn, but GATED by appearance: gate=sigmoid(MLP(frame)).
    Gate bias init negative -> starts ~appearance-only; learns to open motion only where appearance is unsure."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.nq, self.nkv, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.gate = nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(inplace=True), nn.Linear(dim // 4, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)       # sigmoid(-2)=0.12 -> start mostly appearance

    def forward(self, q_tok, kv_tok):                     # frame [B,T,d], point [B,M,d] -> [B,d]
        kv = self.nkv(kv_tok)
        a, _ = self.attn(self.nq(q_tok), kv, kv)          # motion attended per frame token
        g = torch.sigmoid(self.gate(q_tok))               # [B,T,1] appearance-conditioned gate
        x = q_tok + g * a                                 # gated motion injection
        x = x + self.ffn(self.n2(x))
        return x.mean(1)


class InjectBlock(nn.Module):
    """Flamingo-style gated cross-attn: inject motion tokens into a ViT layer's hidden states.
    tanh-gate init 0 -> starts as identity (pure appearance), learns to open injection."""
    def __init__(self, vis_dim, mot_dim, heads=8):
        super().__init__()
        self.mot_proj = nn.Linear(mot_dim, vis_dim)
        self.nq, self.nkv = nn.LayerNorm(vis_dim), nn.LayerNorm(vis_dim)
        self.attn = nn.MultiheadAttention(vis_dim, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))         # tanh(0)=0 -> identity at init

    def forward(self, h, motion):                        # h [N,L,vis], motion [N,M,mot]
        kv = self.nkv(self.mot_proj(motion))
        a, _ = self.attn(self.nq(h), kv, kv)
        return h + torch.tanh(self.gate) * a


class FrameTemporalMamba(nn.Module):
    """Order-aware temporal modeling over the T per-frame CLIP tokens (frames pre-ordered in time).
    Same recipe as model_v1.TemporalMamba (Mamba SSM; BiGRU fallback) minus the t-sort; residual.
    Fixes the order-blindness of the mean-pool over T (targets temporally-confusable actions)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        try:
            from mamba_ssm import Mamba
            self.seq, self.kind = Mamba(d_model=dim), "mamba"
        except Exception as e:                                # robust fallback
            self.seq, self.kind = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True), "gru"
            print(f"[FrameTemporalMamba] mamba_ssm unavailable ({e}); using BiGRU")

    def forward(self, x):                                     # x [B,T,dim], frames time-ordered
        y = self.norm(x)
        y = self.seq(y)[0] if self.kind == "gru" else self.seq(y)
        return x + y                                          # residual


class PromptLearner(nn.Module):
    """CoOp: learnable context tokens spliced between [BOS] and [class name .][EOS] (HF CLIP)."""
    def __init__(self, classnames, clip_model, tokenizer, n_ctx=16):
        super().__init__()
        ctx_dim = clip_model.text_model.config.hidden_size               # 512
        ctx = torch.empty(n_ctx, ctx_dim); nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)                                     # learnable context
        prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prefix} {name}." for name in classnames]
        ids = tokenizer(prompts, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids  # [C,77]
        with torch.no_grad():
            emb = clip_model.text_model.embeddings.token_embedding(ids)  # [C,77,512]
        self.register_buffer("input_ids", ids)
        self.register_buffer("head", emb[:, :1, :])                      # BOS
        self.register_buffer("tail", emb[:, 1 + n_ctx:, :])              # name + EOS + pad
        self.n_cls = len(classnames)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.head, ctx, self.tail], dim=1)             # [C,77,512]


class PAFClipPointV2(nn.Module):
    def __init__(self, classnames, hand_prompts=None, point_encoder="pointmamba",
                 fusion="cross", text="coop", n_ctx=16, lora_r=0, lora_alpha=16,
                 img_adapter=False, inject_every=0, temporal_mamba=False,
                 clip_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.tok = CLIPTokenizer.from_pretrained(clip_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.use_lora = lora_r > 0
        if self.use_lora:
            add_lora_to_clip_vision(self.clip, lora_r, lora_alpha)        # adapt ViT to event frames
        dim = self.clip.config.projection_dim                            # 512

        if point_encoder == "pointmamba":
            self.point = PointMambaEncoder(dim)
        elif point_encoder == "tfirst":
            self.point = TemporalPointEncoder(dim)
        elif point_encoder == "pointnet2":
            self.point = PointNet2Encoder(dim)
        else:
            self.point = PointNetEncoder(4, dim)
        self.fusion = fusion
        self.fusion_mod = (DeepFusion(dim) if fusion == "deep"
                           else GatedCrossAttnFusion(dim) if fusion == "gated"
                           else CrossAttnFusion(dim) if fusion == "cross" else None)
        self.fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        if img_adapter:                                                  # CLIP-Adapter (MLP on frozen feats)
            self.img_adapter = nn.Sequential(nn.Linear(dim, dim // 4), nn.ReLU(inplace=True),
                                             nn.Linear(dim // 4, dim))
            nn.init.zeros_(self.img_adapter[-1].weight)                  # zero-init -> start == frozen
            nn.init.zeros_(self.img_adapter[-1].bias)
        else:
            self.img_adapter = None
        self.inject_every = inject_every                                 # per-layer motion injection
        if inject_every > 0:
            assert self.use_lora, "per-layer inject needs the LoRA spine (set lora_r>0)"
            vdim = self.clip.config.vision_config.hidden_size            # 768
            nL = len(self.clip.vision_model.encoder.layers)             # 12
            self.inject = nn.ModuleDict({str(i): InjectBlock(vdim, dim)
                                         for i in range(nL) if (i + 1) % inject_every == 0})
            print(f"[inject] motion injected after ViT layers: {sorted(int(k) for k in self.inject)}")
        else:
            self.inject = nn.ModuleDict()
        self.temporal = FrameTemporalMamba(dim) if temporal_mamba else None   # order-aware over T frames
        if temporal_mamba:
            print("[temporal] FrameTemporalMamba on frame tokens (order-aware, replaces mean-pool blindness)")
        self.logit_scale = nn.Parameter(torch.tensor(2.659438))

        self.text = text
        if text == "coop":
            self.prompt_learner = PromptLearner(classnames, self.clip, self.tok, n_ctx)
        else:
            prompts = hand_prompts or [f"a photo of a person {c}" for c in classnames]
            with torch.no_grad():
                tk = self.tok(prompts, padding=True, return_tensors="pt")
                tf = self.clip.get_text_features(**tk)
            self.register_buffer("text_fixed", F.normalize(tf, dim=-1))

    def _text_feats(self):
        if self.text != "coop":
            return self.text_fixed
        pl, tm = self.prompt_learner, self.clip.text_model
        embeds = pl()                                                    # [C,77,512] (ctx trainable)
        hidden = tm.embeddings(inputs_embeds=embeds)                     # + positional
        causal = _create_4d_causal_attention_mask(embeds.shape[:2], embeds.dtype, embeds.device)
        x = tm.encoder(inputs_embeds=hidden, causal_attention_mask=causal)[0]
        x = tm.final_layer_norm(x)
        pooled = x[torch.arange(x.shape[0]), pl.input_ids.argmax(-1)]    # EOT token
        return F.normalize(self.clip.text_projection(pooled), dim=-1)    # [C,dim] learnable

    def _img_tokens(self, frames):                                       # [B,T,3,H,W]->[B,T,dim]
        B, T = frames.shape[:2]
        ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
        with ctx:                                                        # LoRA needs grad into ViT
            f = self.clip.get_image_features(pixel_values=frames.flatten(0, 1))
        return f.view(B, T, -1)

    def _img_tokens_inject(self, frames, motion):                        # motion injected per ViT layer
        B, T = frames.shape[:2]
        vm = self.clip.vision_model
        M = motion.repeat_interleave(T, dim=0)                           # [B*T, Mtok, mot_dim]
        h = vm.pre_layrnorm(vm.embeddings(frames.flatten(0, 1)))
        for i, layer in enumerate(vm.encoder.layers):
            h = layer(h, None, None)[0]                                  # frozen(+LoRA) ViT block
            if str(i) in self.inject:
                h = self.inject[str(i)](h, M)                            # gated motion injection
        pooled = vm.post_layernorm(h[:, 0, :])
        return self.clip.visual_projection(pooled).view(B, T, -1)

    def forward(self, frames, points, branch="both"):
        if branch == "dual":                                            # late-fuse frame→text + point→text (fair-to-motion open-vocab)
            t = self._text_feats().t(); s = self.logit_scale.exp()
            f = F.normalize(self._img_tokens(frames).mean(1), dim=-1)
            p = F.normalize(self.point(points), dim=-1)
            return s * (f @ t) + s * (p @ t)                            # both modalities aligned to text
        if branch == "point":
            fused = self.point(points)
        elif self.inject_every and branch == "both":                    # per-layer motion injection
            fused = self._img_tokens_inject(frames, self.point(points, return_tokens=True)).mean(1)
        else:
            f_tok = self._img_tokens(frames)                            # frozen [B,T,dim]
            if self.temporal is not None:
                f_tok = self.temporal(f_tok)                            # order-aware temporal over T frames
            if self.img_adapter is not None:
                f_tok = f_tok + self.img_adapter(f_tok)                 # trainable MLP feature adapter
            if self.fusion in ("cross", "deep", "gated"):
                fused = f_tok.mean(1) if branch == "frame" else self.fusion_mod(
                    f_tok, self.point(points, return_tokens=True))
            else:
                f_pt = self.point(points)
                if branch == "frame":
                    f_pt = torch.zeros_like(f_pt)
                fused = self.fuse(torch.cat([f_tok.mean(1), f_pt], dim=1))
        fused = F.normalize(fused, dim=-1)
        return self.logit_scale.exp() * fused @ self._text_feats().t()
