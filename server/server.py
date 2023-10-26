import torch

class Server:
    def __init__(self, initial_model, optimizer):
        """
        Initialize the server with the global model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = initial_model.to(device)  # Ensure the model is on the correct device
        self.optimizer = optimizer
        
    def aggregate(self, client_gradients):
        self.optimizer.zero_grad()
        # 使用第一個客戶端的梯度初始化 aggregated_gradients 字典
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}
        
        # 對每個客戶端的梯度，將其加到 aggregated_gradients
        for gradients in client_gradients:
            for name, gradient in gradients.items():
                aggregated_gradients[name] += gradient

        # 計算平均梯度
        num_clients = len(client_gradients)
        for name, gradient in aggregated_gradients.items():
            aggregated_gradients[name] = gradient / num_clients

        # 使用平均梯度更新全局模型
        
        for name, param in self.global_model.named_parameters():
            param.grad = aggregated_gradients[name]
        self.optimizer.step()

    def get_global_model(self):
        """
        获取经过聚合的全局模型。
        """
        if self.global_model is None:
            raise ValueError("Global model is not set or aggregated yet.")
        return self.global_model
