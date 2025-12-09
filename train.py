import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


# データセットの前処理を定義
ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,  # 訓練用を指定
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,  # テスト用を指定
    download=True,
    transform=ds_transform
)

# ミニバッチに分割する DataLoader を作る
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

# バッチを取り出す実験
# この後の処理では不要なので、確認したら削除してよい
# for image_batch, label_batch in dataloader_test:
#     print(image_batch.shape)
#     print(label_batch.shape)
#    break  # 1つ目で終了

# モデルのインスタンスを作成
model = models.MyModel()

# 損失関数 (誤差関数・ロス関数) の選択
loss_fn = torch.nn.CrossEntropyLoss()

# 最適化の方法の選択
lerning_rate = 1e-3  # 学習率
optimizer = torch.optim.SGD(model.parameters(), lr=lerning_rate)

n_epochs = 20

train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f'epoch {epoch+1}/{n_epochs}')
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    print(f'    training loss: {train_loss}') 
    train_loss_log.append(train_loss)

    val_loss = models.test(model, dataloader_train, loss_fn)
    print(f'    validation loss: {val_loss}') 
    val_loss_log.append(val_loss)

    train_acc = models.test_accuracy(model, dataloader_train)
    print(F'    training accuracy: {train_acc*100:.3f}%')
    train_acc_log.append(train_acc)

    val_acc = models.test_accuracy(model, dataloader_test)
    print(F'    validation accuracy: {val_acc*100:.3f}%')
    val_acc_log.append(val_acc)

# グラフを表示
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), train_loss_log)
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))
plt.ylabel('loss')
plt.legend('train')
plt.grid()

plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), val_loss_log)
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))
plt.ylabel('loss')
plt.legend('validation')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), train_acc_log)
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))
plt.ylabel('accuracy')
plt.legend('train')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), val_acc_log)
plt.xlabel('epochs')
plt.xticks(range(1, n_epochs+1))
plt.ylabel('accuracy')
plt.legend('validation')
plt.grid()

plt.show()