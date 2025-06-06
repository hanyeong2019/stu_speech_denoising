import torch, torch.optim as optim, os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_logmag import LogMagDataset
from vae_models     import GaussVAE, StudentTVAE

root = '..'   # 若目录不同请改
trainN=f'{root}/data_proc/noisy_trainset_28spk_wav'; trainC=f'{root}/data_proc/clean_trainset_28spk_wav'
testN =f'{root}/data_proc/noisy_testset_wav';        testC =f'{root}/data_proc/clean_testset_wav'

loader = lambda d: DataLoader(d,batch_size=32,shuffle=True,num_workers=0)
train_set = LogMagDataset(trainN, trainC, seg=128, mode='train')
val_set   = LogMagDataset(testN, testC, seg=128, mode='eval')
train_loader, val_loader = loader(train_set), loader(val_set)

models = {'gauss': GaussVAE(beta=1.0), 'student': StudentTVAE(beta=0.5)}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for tag, model in models.items():
    model.to(device); opt=optim.Adam(model.parameters(),1e-4)
    w=SummaryWriter(f'runs/{tag}')
    dummy_input = torch.randn(1, 1, 257, 128).to(device)
    try:
        w.add_graph(model, (dummy_input, dummy_input))
    except Exception as e:
        print(f'⚠️ Graph 写入失败（通常不影响功能）：{e}')
    # 1e-4是否更加合理？
    best=1e9
    for ep in range(1,21):
        # train
        model.train(); tloss=0
        for n,c in tqdm(train_loader,f'{tag} E{ep}',ncols=80):
            n,c=n.to(device),c.to(device)
            loss,rec,kl,_=model(c,n)
            opt.zero_grad(); loss.backward(); opt.step()
            tloss+=loss.item()
        w.add_scalar('loss/train', tloss/len(train_loader), ep)
        w.add_scalar('loss/rec', rec.item(), ep)
        w.add_scalar('loss/kl',  kl.item(), ep)
        # val
        model.eval(); vloss=0
        with torch.no_grad():
            for n,c in val_loader:
                loss,_,_,_=model(c.to(device),n.to(device)); vloss+=loss.item()
        vavg=vloss/len(val_loader); w.add_scalar('loss/val',vavg,ep)
        print(f'[{tag}] E{ep} train {tloss/len(train_loader):.4f} val {vavg:.4f}')
        if vavg<best:
            best=vavg; torch.save(model.state_dict(),f'{tag}_vae.pt'); print('  **saved**')
    w.close()
