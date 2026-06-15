import os, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sadbnet_model import SADBNet
from det_dataset import DetDataset

BATCH_SIZE = 8
EPOCHS = 100
LR = 5e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = '/kaggle/working/checkpoints'
TRAIN_IMG = '/kaggle/input/datasets/kagglemodeltraining/icdar-2015-detection/icdar2015_detection/icdar2015/train_images'
TRAIN_GT = '/kaggle/input/datasets/kagglemodeltraining/icdar-2015-detection/icdar2015_detection/icdar2015/train_gt'

class DBNetLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, l1_scale=10.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')
        self.l1_scale = l1_scale

    def forward(self, pred, target):
        prob, thresh, binary = pred
        t_prob = target['prob_map']
        mask = target['mask']
        t_thresh = target['thresh_map']
        t_mask = target['thresh_mask']
        bce = torch.sum(self.bce(prob, t_prob) * mask) / (torch.sum(mask) + 1e-6)
        l1 = torch.sum(self.l1(thresh, t_thresh) * t_mask) / (torch.sum(t_mask) + 1e-6) * self.l1_scale
        inter = torch.sum(binary * t_prob * mask)
        union = torch.sum(binary * mask) + torch.sum(t_prob * mask)
        dice = 1.0 - (2.0 * inter) / (union + 1e-6)
        total = bce + self.alpha * l1 + self.beta * dice
        return total, bce, l1, dice

def train():
    os.makedirs(CKPT, exist_ok=True)
    print('='*60)
    print('SA-DBNet — Fine-Tuning from 44% F1')
    print('='*60)

    ds = DetDataset(TRAIN_IMG, TRAIN_GT, augment=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = SADBNet().to(DEVICE)

    # Load the 44% F1 checkpoint
    pretrained = '/kaggle/input/datasets/kagglemodeltraining/icdar-2015-detection/sadbnet_flawless_best.pth'
    if os.path.exists(pretrained):
        model.load_state_dict(torch.load(pretrained, map_location=DEVICE))
        print('Loaded pre-trained 44% F1 weights!')
    else:
        print('WARNING: pretrained not found!')
    # Use both T4 GPUs if available
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params/1e6:.2f}M')

    criterion = DBNetLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_bce, total_dice = 0, 0, 0

        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            target = {k: v.to(DEVICE) for k, v in batch.items() if k != 'image'}

            optimizer.zero_grad()
            pred = model(images)
            loss, bce, l1, dice = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()
            pbar.set_postfix(loss=f'{loss.item():.3f}', bce=f'{bce.item():.4f}')

        scheduler.step()
        avg = total_loss / len(loader)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1} | Loss: {avg:.4f} | BCE: {total_bce/len(loader):.4f} | Dice: {total_dice/len(loader):.4f} | LR: {lr:.6f}')

        # Save
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, os.path.join(CKPT, 'sadbnet_kaggle_latest.pth'))
        if avg < best_loss:
            best_loss = avg
            torch.save(state, os.path.join(CKPT, 'sadbnet_kaggle_best.pth'))
            print(f'  >>> New Best: {best_loss:.4f}')

    print(f'\nDone! Best Loss: {best_loss:.4f}')

if __name__ == '__main__':
    train()
