import torch, torch.nn as nn, torch.nn.functional as F, math

# ─────────────────── 共用卷积编解码 ───────────────────
class _Enc(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 , 64 , 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, (2,1), 1), nn.GELU(),    # F ↓2
            nn.Conv2d(128,128, 3, (2,1), 1), nn.GELU())    # F ↓4
        self.flat_dim = 128 * (257//4) * 128
        self.mu, self.logv = nn.Linear(self.flat_dim, zdim), nn.Linear(self.flat_dim, zdim)
        self.log_nu = nn.Parameter(torch.log(torch.tensor(10.)))   # Student-t 自由度 ν

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.mu(h), self.logv(h), self.log_nu.exp()

class _Dec(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.fc = nn.Linear(zdim, 128 * (257//4) * 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,(2,1),1,output_padding=(1,0)), nn.GELU(),
            nn.ConvTranspose2d(128, 64,3,(2,1),1,output_padding=(1,0)), nn.GELU(),
            nn.Conv2d(64,1,3,1,1), nn.Sigmoid())

    def forward(self, z, noisy):                       # noisy:(B,1,257,128)
        B = z.size(0)
        h = self.fc(z).view(B,128,257//4,128)
        mask = self.deconv(h)
        return mask * noisy                           # 输出幅度谱

# ─────────────────── Re-parameterization ───────────────────
def rep_gauss(mu, logv):
    return mu + (0.5*logv).exp() * torch.randn_like(mu)

def rep_student(mu, logv, nu):
    std = (0.5*logv).exp()
    eps = torch.randn_like(std)
    g   = torch.empty_like(std).chi2_(nu)             # s ~ χ²(ν)
    return mu + std * eps / torch.sqrt(g / nu)

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
    def __init__(self, z=32, beta=0.5):
        super().__init__(); self.enc=_Enc(z); self.dec=_Dec(z); self.beta=beta
    def forward(self, clean, noisy):
        mu, logv, nu = self.enc(clean)
        z      = rep_student(mu, logv, nu)
        recon  = self.dec(z, noisy)
        rec    = F.mse_loss(recon, clean)
        kl     = kl_student_approx(mu, logv, nu)
        return rec + self.beta*kl, rec, kl, recon
