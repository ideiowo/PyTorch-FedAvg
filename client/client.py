import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torch
import torch.nn as nn
from tqdm import tqdm

class Client:
    def __init__(self, client_id, data_loader, model, optimizer, criterion):
        self.client_id = client_id
        self.data_loader = data_loader      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_and_validate(self, epochs=1, val_loader=None, validate=True):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                # 梯度裁剪
                #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                #print(f"loss: {loss.item()}")

            gradients = {name: param.grad for name, param in self.model.named_parameters()}
            train_loss = total_loss / len(self.data_loader)

            if validate and val_loader:
                val_loss, accuracy = self.validate(val_loader)
            else:
                val_loss, accuracy = None, None

            print(f"Client {self.client_id} - Epoch {epoch+1} - Train Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}%")

        return gradients, train_loss, val_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        return val_loss, accuracy

    def set_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        
