import torch
import torch.nn as nn

# Định nghĩa thiết bị chung cho toàn bộ FL Core
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    """Mạng CNN/MLP dùng cho FL."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelUtils:
    """Các hàm thao tác với trọng số."""
    @staticmethod
    def get_params(model: nn.Module) -> dict:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def set_params(model: nn.Module, params: dict):
        state_dict = {k: (v.clone().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)) for k, v in params.items()}
        model.load_state_dict(state_dict, strict=True)

