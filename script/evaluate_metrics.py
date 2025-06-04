import os, csv, numpy as np, torch, librosa, soundfile as sf
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from vae_models import GaussVAE, StudentTVAE

root='..'; testN=f'{root}/data_proc/noisy_testset_wav'; testC=f'{root}/data_proc/clean_testset_wav'
device='cuda' if torch.cuda.is_available() else 'cpu'
g=GaussVAE(); g.load_state_dict(torch.load('gauss_vae.pt',map_location=device)); g.eval().to(device)
t=StudentTVAE(); t.load_state_dict(torch.load('student_vae.pt',map_location=device)); t.eval().to(device)

def mag2wav(m,ph): return librosa.istft(m*np.exp(1j*ph))
def snr(c,e): return 10*np.log10(np.sum(c**2)/(np.sum((c-e)**2)+1e-8))

rows=[]
for f in tqdm(os.listdir(testC)):
    if not f.endswith('_spec.npy'): continue
    n_mag,c_mag = np.load(os.path.join(testN,f)), np.load(os.path.join(testC,f))
    phase=np.load(os.path.join(testN,f.replace('_spec','_phase')))
    n_wav, c_wav = mag2wav(n_mag,phase), mag2wav(c_mag,phase)
    x=torch.tensor(np.log1p(n_mag)[None,None,:,:],dtype=torch.float32).to(device)
    with torch.no_grad():
        _,_,_,rg=g(x,x); _,_,_,rt=t(x,x)
    g_wav=mag2wav(np.expm1(rg.squeeze().cpu().numpy()),phase)
    t_wav=mag2wav(np.expm1(rt.squeeze().cpu().numpy()),phase)
    L=min(len(c_wav),len(g_wav),len(t_wav)); c_wav,n_wav,g_wav,t_wav=[w[:L] for w in (c_wav,n_wav,g_wav,t_wav)]
    rows.append({'file':f,
                 'SNR_noisy':snr(c_wav,n_wav), 'SNR_gauss':snr(c_wav,g_wav), 'SNR_st':snr(c_wav,t_wav),
                 'PESQ_noisy':pesq(16000,c_wav,n_wav,'wb'),'PESQ_gauss':pesq(16000,c_wav,g_wav,'wb'),'PESQ_st':pesq(16000,c_wav,t_wav,'wb'),
                 'STOI_noisy':stoi(c_wav,n_wav,16000,False),'STOI_gauss':stoi(c_wav,g_wav,16000,False),'STOI_st':stoi(c_wav,t_wav,16000,False)})
# 保存 & 平均
with open('metric_compare.csv','w',newline='') as f:
    csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader(); csv.writer(f).writerows([r.values() for r in rows])
avg=lambda k:np.mean([r[k] for r in rows])
print("\n=== 平均 ===")
for m in ('SNR','PESQ','STOI'):
    print(f"{m} noisy:{avg(m+'_noisy'):.2f}  gauss:{avg(m+'_gauss'):.2f}  st:{avg(m+'_st'):.2f}")
