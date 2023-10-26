import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split


def load_data(data_dir='./data/CIFAR10/train', num_clients=10, batch_size=32, val_split=0.1):
    # 2. 資料的前處理和轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 1. 從CIFAR10/train加載數據
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # 3. & 4. 分割資料以模擬多個客戶端的環境且確保均勻的數據分佈
    data_size = len(dataset)
    client_data_size = data_size // num_clients
    indices = list(range(data_size))
    random.shuffle(indices)

    client_loaders = []
    client_val_loaders = []

    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = start_idx + client_data_size if i != (num_clients - 1) else data_size
        subset_indices = indices[start_idx:end_idx]
        dataset_per_client = torch.utils.data.Subset(dataset, subset_indices)

        val_size = int(len(dataset_per_client) * val_split)
        train_size = len(dataset_per_client) - val_size

        train_dataset, val_dataset = random_split(dataset_per_client, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        client_loaders.append(train_loader)
        client_val_loaders.append(val_loader)

    return client_loaders, client_val_loaders



if __name__ == "__main__":
    client_loaders, client_val_loaders = load_data()
    for idx, loader in enumerate(client_loaders):
        print(f"Client {idx + 1} data batches:", len(loader))
