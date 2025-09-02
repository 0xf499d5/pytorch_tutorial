import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F


""" 准备数据集 """
class ToyDataset(Dataset):
    def __init__(self, x, t):
        self.features = x
        self.targets = t

    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target
    
    def __len__(self):
        return self.targets.shape[0]


def prepare_dataset():
    batch_size = 2
    num_workers = 0

    x_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    t_train = torch.tensor([0, 0, 0, 1, 1])
    train_ds = ToyDataset(x_train, t_train)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,  # 锁页内存，类似DMA的访存加速
        sampler=DistributedSampler(train_ds)  # 数据分片：在多张GPU上分发不重复的数据批次
    )

    x_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    t_test = torch.tensor([0, 1])
    test_ds = ToyDataset(x_test, t_test)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return train_loader, test_loader


""" 定义模型 """
class MLP(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, features):
        logits = self.layers(features)
        return logits


""" 计算预测准确率 """
def compute_accuracy(model, dataloader, device):
    correct = 0.0
    num_samples = 0

    model.eval()
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        preds = torch.argmax(logits, dim=-1)
        compare = preds == targets
        correct += torch.sum(compare)
        num_samples += len(compare)
    acc = (correct / num_samples).item()
    return acc


""" 训练模型 """
def train(rank: str, world_size: int, num_epochs: int):
    """ 设置DDP环境 """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    init_process_group(
        backend="nccl",
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)

    """ 实例化模型，优化器以及数据加载器 """
    model = MLP(num_inputs=2, num_outputs=2)
    model.to(rank)
    models = DDP(  # 将一组可以互相通信以同步梯度的模型副本封装在一起，使得它使用起来就像单个模型接口一样
        module=model,
        device_ids=[rank]
    )
    
    optimizer = torch.optim.SGD(
        params=models.parameters(),
        lr=0.5
    )

    train_loader, test_loader = prepare_dataset()

    """ 训练循环 """
    for epoch in range(num_epochs):
        models.train()
        train_loader.sampler.set_epoch(epoch)  # 确保每个GPU在每个epoch看到不同的数据顺序

        for features, targets in train_loader:
            features = features.to(rank)
            targets = targets.to(rank)

            logits = models(features)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging:
            print(
                f"[GPU{rank} Epoch: {epoch:03d}/{num_epochs:03d}] | "
                f"Batch size {targets.shape[0]:03d} | "
                f"Train/Valid Loss: {loss:.2f}"
            )

    """ 评测模型 """
    models.eval()
    try:
        acc_train = compute_accuracy(models, train_loader, device=rank)
        acc_test = compute_accuracy(models, test_loader, device=rank)
        print(
            f"[GPU{rank} Training Accuracy: {acc_train:.2f}]\n"
            f"[GPU{rank} Testing Accuracy: {acc_test:.2f}]"
        )
    except Exception as e:
        print("#" * 50)
        print(str(e))
        print("#" * 50)

    """ 释放分布式资源 """
    destroy_process_group()


def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPUs used (it means 'rank'): {os.environ['CUDA_VISIBLE_DEVICES']}")

    torch.manual_seed(42)
    world_size = torch.cuda.device_count()
    num_epochs = 3
    mp.spawn(
        train,
        args=(world_size, num_epochs),
        nprocs=world_size
    )


if __name__ == "__main__":
    main()