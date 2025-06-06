import torch, torch.nn as nn, torch.nn.functional as F, math

# ─────────────────── 共用卷积编解码 ───────────────────
class _Enc(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 , 64 , 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, (2,1), 1), nn.GELU(),    # F ↓2
            nn.Conv2d(128,128, 3, (2,1), 1), nn.GELU())    # F ↓4
        # --- 动态推断展平维度 ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 257, 128)   # (B,C,F,T)
            self.flat_dim = self.conv(dummy).view(1, -1).size(1)

        self.mu     = nn.Linear(self.flat_dim, zdim)
        self.logv   = nn.Linear(self.flat_dim, zdim)
        self.log_nu = nn.Parameter(torch.log(torch.tensor(10.)))

    def forward(self, x):
        h = self.conv(x).flatten(1)
        mu = self.mu(h)
        logv = self.logv(h)
        nu = self.log_nu.exp()
        nu = torch.clamp(nu, min=1.0, max=100.0)  # 添加这一行
        return mu, logv, nu


class _Dec(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.fc = nn.Linear(zdim, 128 * (257//4) * 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,(2,1),1,output_padding=(0,0)), nn.GELU(),
            nn.ConvTranspose2d(128, 64,3,(2,1),1,output_padding=(0,0)), nn.GELU(),
            nn.Conv2d(64,1,3,1,1), nn.Sigmoid())
    def forward(self, z, noisy):                 # noisy:(B,1,257,128)
        B = z.size(0)
        h = self.fc(z).view(B,128,257//4,128)    # (B,128,65,128)
        mask = self.deconv(h)                    # 得到 (B,1,253,128) 或别的
        # ---- 对齐频率维 ----
        Freq = noisy.size(2)                     # =257
        if mask.size(2) > Freq:                  # 裁剪
            mask = mask[:, :, :Freq, :]
        elif mask.size(2) < Freq:                # 右侧 0 填充
            pad = Freq - mask.size(2)
            mask = F.pad(mask, (0,0,0,pad))      # pad freq-dim (left=0, right=pad)
        return mask * noisy


# ─────────────────── Re-parameterization ───────────────────
def rep_gauss(mu, logv):
    return mu + (0.5*logv).exp() * torch.randn_like(mu)

# def rep_student(mu, logv, nu):
#     std = torch.exp(0.5 * logv)
#     gamma = torch.distributions.Gamma(nu / 2, 0.5)
#     g = gamma.sample(std.shape).to(std.device)
#     eps = torch.randn_like(std)
#     return mu + std * eps / torch.sqrt(g / nu)

def rep_student(mu, logv, nu):
    if torch.isnan(nu).any():
        raise ValueError(f"‼️ nu 出现 NaN:{nu}")
    nu = torch.clamp(nu, min=1.0, max=100.0)
    gamma = torch.distributions.Gamma(nu / 2, 0.5)
    g = torch.clamp(gamma.sample(mu.shape), min=1e-4).to(mu.device)
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logv) * eps / torch.sqrt(g / nu)


# ─────────────────── KL 散度函数 ───────────────────
def kl_gaussian(mu, logv):
    """KL( N(μ,σ²) || N(0,1) )"""
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def kl_student_approx(mu, logv, nu):
    """
    论文 Zhang 2018 / Takahashi 2020 的简化近似：
    KL( 𝒯_ν(μ,σ²) || 𝒩(0,1) ) ≈ KL_Gauss + ½( log ν − ψ(ν/2) + log π )
    """
    kl_g = kl_gaussian(mu, logv)
    const = 0.5 * ( torch.log(nu) - torch.digamma(nu/2) + math.log(math.pi) )
    return kl_g + const.to(mu.device)

# ─────────────────── 两种 VAE ───────────────────
class GaussVAE(nn.Module):
    def __init__(self, z=32, beta=1.0):
        super().__init__(); self.enc=_Enc(z); self.dec=_Dec(z); self.beta=beta
    def forward(self, clean, noisy):
        mu, logv, _ = self.enc(clean)
        z      = rep_gauss(mu, logv)
        recon  = self.dec(z, noisy)
        rec    = F.mse_loss(recon, clean)
        kl     = kl_gaussian(mu, logv)
        return rec + self.beta*kl, rec, kl, recon

class StudentTVAE(nn.Module):
    def __init__(self, z=32, beta=0.3):
        super().__init__(); self.enc=_Enc(z); self.dec=_Dec(z); self.beta=beta
    def forward(self, clean, noisy):
        mu, logv, nu = self.enc(clean)
        z      = rep_student(mu, logv, nu)
        recon  = self.dec(z, noisy)
        rec    = F.mse_loss(recon, clean)
        kl     = kl_student_approx(mu, logv, nu)
        return rec + self.beta*kl, rec, kl, recon
