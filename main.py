import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))  # 顯示第一個GPU的名稱
# Define device for running operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from client import Client
from server import Server
from utils.data_utils import load_data
from models.architecture import DNN,CNN,ResNet18
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch.optim as optim


# Parameters
NUM_CLIENTS = 10
ROUNDS = 8
EPOCHS = 1
BATCH_SIZE = 32

# Load data for all clients
# Step 1: Load validation data
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.ImageFolder('./data/CIFAR10/validation', transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
client_loaders, client_val_loaders = load_data(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

# Initialize server and clients
initial_model = ResNet18().to(device)  # or CNN(), ResNet18()
optimizer = optim.Adam(initial_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


server = Server(initial_model,optimizer = optimizer)
clients = [Client(client_id=i, 
                  data_loader=client_loaders[i], 
                  model=initial_model,
                  optimizer=optimizer,
                  criterion = criterion) for i in range(NUM_CLIENTS)]

# 初始化loss和accuracy的記錄
all_losses = []
all_accuracies = []

# 修改每一輪的訓練過程
for round in range(ROUNDS):
    client_gradients = []
    round_train_losses = []
    round_val_losses = []
    round_accuracies = []

    # 1. Train and validate on each client
    for client_id, (client, val_loader) in enumerate(zip(clients, client_val_loaders)):
        print(f"Training client {client_id + 1} in round {round + 1}...")
        gradients, train_loss, val_loss, accuracy = client.train_and_validate(epochs=EPOCHS, val_loader=val_loader)
        client_gradients.append(gradients)
        round_train_losses.append(train_loss)
        round_val_losses.append(val_loss)
        round_accuracies.append(accuracy)

    # 2. Aggregate the models on the server
    print(f"Aggregating models for round {round + 1}...")
    server.aggregate(client_gradients)

    # 3. Broadcast the updated global model to all clients
    global_model = server.get_global_model()
    for client in clients:
        client.set_model(global_model)

    # 計算並記錄此輪的平均損失和準確度
    avg_train_loss = sum(round_train_losses) / len(clients)
    avg_val_loss = sum(round_val_losses) / len(clients)
    avg_accuracy = sum(round_accuracies) / len(clients)

    all_losses.append(avg_val_loss)
    all_accuracies.append(avg_accuracy)

    print(f"Round {round + 1} - Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, Avg Accuracy: {avg_accuracy}%")


print("FedAVG Training Completed!")




# Step 2: Set model to evaluation mode and move it to device
server.global_model = server.global_model.to(device)
server.global_model.eval()

# Step 3: Predict on validation set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the device
        outputs = server.global_model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

# Step 4: Calculate accuracy
accuracy = 100 * correct / total
print(f'Validation Accuracy of the global model: {accuracy}%')

# 使用seaborn繪製loss和accuracy隨著rounds的變化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.lineplot(x=range(1, ROUNDS+1), y="Loss", data=pd.DataFrame({"Rounds": range(1, ROUNDS+1), "Loss": all_losses}))
plt.title("Average Validation Loss over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
sns.lineplot(x=range(1, ROUNDS+1), y="Accuracy (%)", data=pd.DataFrame({"Rounds": range(1, ROUNDS+1), "Accuracy (%)": all_accuracies}))
plt.title("Average Validation Accuracy over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()

