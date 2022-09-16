# Solving Sudoku with Feed-Forward Neural Networks
### By: Mohamed Abdelhamid Ghanem -- 70026144

## 1. Global Imports and Dependencies:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

```

## 2. Dataset Loading & Preprocessing

**Note**: Before you proceed, you need to download the Sudoku 9M dataset from Kaggle and place he `sudoku.csv` file it in the same directory as this notebook.


```python
import pandas as pd
sudoku_df = pd.read_csv("sudoku.csv")
sudoku_records = sudoku_df.to_dict('records')
```

In our dataloader, we employ a trick to augment training by taking puzzle solutions and emptying random cells then pass them to the model.


```python
from torch.utils.data import Dataset
import numpy as np
import random

random.seed(2022)

class SudokuDataset(Dataset):
    def __init__(self, training=True, augment_prob=0.5, split_ratio=0.8, device="cuda") -> None:
        super().__init__()
        self.training = training
        self.augment_prob = augment_prob
        self.device = device
        train_size = int(len(sudoku_records)*split_ratio)
        if training:
            self.puzzles = sudoku_records[:train_size]
        else:
            self.puzzles = sudoku_records[train_size:]
    
    def empty_randomly(self, arr):
        empty_count = random.randint(1, 43)
        indices = random.sample(range(9*9), empty_count)
        arr[indices] = 0
        return arr
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, index):
        if self.training and random.random() < self.augment_prob:
            puzzle = np.array([int(x) for x in self.puzzles[index]["solution"]])
            puzzle = self.empty_randomly(puzzle)
        else:
            puzzle = np.array([int(x) for x in self.puzzles[index]["puzzle"]])
        puzzle = puzzle.reshape((9,9))

        solution = np.array([int(x) for x in self.puzzles[index]["solution"]])-1
        solution = solution.reshape((9,9))

        # Convert to tensor and normalize puzzle
        puzzle_norm = torch.tensor(puzzle).to(self.device).unsqueeze(0)/9.0 - 0.5
        puzzle_ref = torch.tensor(puzzle)#F.one_hot(puzzle, 10).permute(2, 0, 1).float()
        solution = torch.tensor(solution).to(self.device)

        return puzzle_norm, puzzle_ref, solution

```

## 3. Model Definition
For this task, we use two types of CNNs: a regular purely convolutional network (no up/down-sampling) and the famous UNet convolutional architecture. UNet is essentially an encoder-decoder architecture that generates the output class map by downsampling then upsampling the input map. Note that this turns the problem into an image-to-image task.

#### 3.1 Building Blocks
The model is built out of convolutional ReLU-activated upsampling and downsampling blocks.


```python
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ReLUConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ReLUConv2d, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            nn.ReflectionPad2d((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)
```

#### 3.2 Model Architecture
In addition to encoder/decoder blocks, the architecture uses skip connections to retain spatial information lost in downsampling.


```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class PureCNN(nn.Module):
    def __init__(self):
        super(PureCNN, self).__init__()
        self.conv_layers = nn.Sequential(ReLUConv2d(1,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3),
                                         ReLUConv2d(512,512,3))
        self.out_conv = nn.Conv2d(512, 9, 1)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.out_conv(x)
        return x
```

## 4. Model Training


```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer():
    def __init__(self, model_type, batch_size, model_path=None, augment_prob=0.5, device="cuda"):
        if model_type == "PureCNN":
            self.model = PureCNN().to(device)
        else:
            self.model = UNet(n_channels=1, n_classes=9).to(device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.batch_size = batch_size
        self.train_dataset = SudokuDataset(training=True, augment_prob=augment_prob, device=device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.val_dataset = SudokuDataset(training=False, device=device)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=512)

        self.optimizer = Adam(self.model.parameters(), lr=7e-4)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train(self, n_epochs, start_epoch=0, early_stop=True):
        best_accr = 0.05
        for epoch in range(start_epoch, n_epochs):
            self.model.train()
            for puzzle, puzzle_ref, sol_true in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                self.optimizer.zero_grad()
                sol_logits = self.model(puzzle)
                loss = self.loss_fn(sol_logits, sol_true)
                loss.backward()
                self.optimizer.step()
            torch.save(self.model.state_dict(), f"pretrained/unet_sudoku_epoch_{epoch}.pth")
            val_accr = self.evaluate()
            print(f"Epoch #{epoch} validation accuracy= {val_accr*100:.2f}%")
            if val_accr > best_accr:
                best_accr = val_accr
                torch.save(self.model.state_dict(), "pretrained/unet_sudoku_best_model.pth")
            elif early_stop:
                return


    def evaluate(self):
        correct = 0
        self.model.eval()
        print(f"Evaluating model..")
        for puzzle, puzzle_ref, sol_true in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
            sol_logits = self.model(puzzle)
            sol_pred = sol_logits.max(1)[1]
            for i in range(sol_pred.shape[0]):
                correct += torch.all(sol_pred[i, puzzle_ref[i]==0] == sol_true[i, puzzle_ref[i]==0]).item()
        accr = correct/(len(self.val_dataloader)*self.val_dataloader.batch_size)
        return accr

```

**Note**: in case you want to start training from a pretrained checkpoint, you can provide the `model_path` argument to `Trainer`.


```python
trainer = Trainer("PureCNN", 128, model_path=None, augment_prob=0.2, device="cuda")
trainer.train(5, start_epoch=0)
```

## 5. Performance & Improvement
After 5 epochs of training, the UNet model scores a best validation accuracy of 8% while its pure CNN counterpart scored 86% proving that a typical UNet architecture is highly unfit for this task mainly due to limited network size which can be alleviated by increasing the number of convolutional filters in each layer. Note that this percentage metric represents the ratio of puzzles that were fuly solved correctly, not the percentage of digits correctly placed. The latter, of course, would be higher than the former.

One possible improvement would be to solve the puzzle cell-by-cell instead of one-shot. In this manner, we only take the model's most confident prediction each time. Note that this incurs an extra cost of time.

## Acknowledgments
The UNet PyTorch implementation used here was adapted from `milesial`'s [Pytorch-UNet repository](https://github.com/milesial/Pytorch-UNet).
