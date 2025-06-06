import os, csv, torch, numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from torch.utils.data import DataLoader
from vae_models import GaussVAE, StudentTVAE
from dataset_logmag import LogMagDataset

def mag2wav(mag, phase):
    return librosa.istft(mag * np.exp(1j * phase))

def snr(clean, est):
    return 10 * np.log10(np.sum(clean**2) / (np.sum((clean - est)**2) + 1e-8))

# ===== 主要评估函数 =====
def evaluate(model_g, model_t, dataset, phase_dir, output_csv):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_g.eval().to(device)
    model_t.eval().to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []

    # for i, (n, c) in tqdm(enumerate(loader), total=len(loader)):
    #     n, c = n.to(device), c.to(device)

    #     # 获取文件名对应 phase 文件路径
    #     fname = os.path.basename(dataset.pairs[i][0])
    #     phase_path = os.path.join(phase_dir, fname.replace('_spec.npy', '_phase.npy'))
    #     try:
    #         # phase = np.load(phase_path)
    #         if phase.shape[1] >= 128:
    #           phase = phase[:, :128]
    #         else:
    #           pad = 128 - phase.shape[1]
    #           phase = np.pad(phase, ((0, 0), (0, pad)))
    #     except:
    #         print(f"⚠️ phase 文件缺失：{phase_path}，已跳过")
    #         continue
    
    for i, (n, c, basename) in tqdm(enumerate(loader), total=len(loader)):
        n, c = n.to(device), c.to(device)

        phase_path = os.path.join(phase_dir, basename[0].replace('_spec.npy', '_phase.npy'))
        fname = os.path.basename(dataset.pairs[i][0])
        try:
            phase = np.load(phase_path)
        # 对 phase 补齐到 128 帧
            if phase.shape[1] >= 128:
               phase = phase[:, :128]
            else:
               phase = np.pad(phase, ((0, 0), (0, 128 - phase.shape[1])))
   
        except:
         print(f"⚠️ phase 文件缺失：{phase_path}，已跳过")
         continue


        with torch.no_grad():
            _, _, _, rg = model_g(n, n)
            _, _, _, rt = model_t(n, n)

        n_mag = np.expm1(n.squeeze().cpu().numpy())
        c_mag = np.expm1(c.squeeze().cpu().numpy())
        g_mag = np.expm1(rg.squeeze().cpu().numpy())
        t_mag = np.expm1(rt.squeeze().cpu().numpy())

        n_wav = mag2wav(n_mag, phase)
        c_wav = mag2wav(c_mag, phase)
        g_wav = mag2wav(g_mag, phase)
        t_wav = mag2wav(t_mag, phase)

        L = min(len(c_wav), len(n_wav), len(g_wav), len(t_wav))
        c_wav, n_wav, g_wav, t_wav = [w[:L] for w in (c_wav, n_wav, g_wav, t_wav)]

        rows.append({
            'file': fname,
            'SNR_noisy': snr(c_wav, n_wav),
            'SNR_gauss': snr(c_wav, g_wav),
            'SNR_st': snr(c_wav, t_wav),
            'PESQ_noisy': pesq(16000, c_wav, n_wav, 'wb'),
            'PESQ_gauss': pesq(16000, c_wav, g_wav, 'wb'),
            'PESQ_st': pesq(16000, c_wav, t_wav, 'wb'),
            'STOI_noisy': stoi(c_wav, n_wav, 16000, False),
            'STOI_gauss': stoi(c_wav, g_wav, 16000, False),
            'STOI_st': stoi(c_wav, t_wav, 16000, False)
        })

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

    # 打印平均指标
    print("\n=== 平均指标 ===")
    print(f"{'Metric':<8} | {'Noisy':>7} | {'Gauss':>7} | {'Student':>7}")
    print("-" * 36)
    avg = lambda k: np.mean([r[k] for r in rows])
    for m in ('SNR', 'PESQ', 'STOI'):
        print(f"{m:<8} | {avg(m+'_noisy'):>7.2f} | {avg(m+'_gauss'):>7.2f} | {avg(m+'_st'):>7.2f}")

# ===== 脚本入口 =====
if __name__ == '__main__':
    root = '..'
    testN = f'{root}/data_proc/noisy_testset_wav'
    testC = f'{root}/data_proc/clean_testset_wav'
    phase_dir = testN
    output_csv = 'metric_compare.csv'

    # 加载模型
    g = GaussVAE()
    g.load_state_dict(torch.load('gauss_vae.pt', map_location='cpu'))

    t = StudentTVAE()
    t.load_state_dict(torch.load('student_vae.pt', map_location='cpu'))

    # 构造数据集（统一输入为 128 帧）
    test_set = LogMagDataset(testN, testC, seg=128, mode='eval')

    # 开始评估
    evaluate(g, t, test_set, phase_dir, output_csv)
