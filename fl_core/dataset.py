from torchvision import datasets, transforms
from config.settings import DATA_DIR
from torch.utils.data import Dataset, Subset
import numpy as np
import os

class DatasetSplit(Dataset):
    """Class bọc dữ liệu cục bộ cho từng AUV."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

class DataManager:
    """Quản lý việc tải và chia MNIST (IID/Non-IID)."""
    @staticmethod
    def _mnist_exists_locally(root_dir: str) -> bool:
        raw_dir = os.path.join(root_dir, "MNIST", "raw")
        required_raw = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ]
        return all(os.path.exists(os.path.join(raw_dir, name)) for name in required_raw)

    @staticmethod
    def get_mnist_data(num_users: int, iid: bool = True, max_train_size: int = None, max_test_size: int = None):
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        has_local_mnist = DataManager._mnist_exists_locally(DATA_DIR)
        need_download = not has_local_mnist
        
        # Trỏ tham số 'root' về DATA_DIR
        train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=need_download, transform=apply_transform)
        test_dataset  = datasets.MNIST(root=DATA_DIR, train=False, download=need_download, transform=apply_transform)
        
        # Giảm kích thước dataset nếu được yêu cầu
        if max_train_size is not None and max_train_size < len(train_dataset):
            train_dataset = Subset(train_dataset, range(max_train_size))
        
        if max_test_size is not None and max_test_size < len(test_dataset):
            test_dataset = Subset(test_dataset, range(max_test_size))

        dict_users = {}

        # Paper Section 5: "MNIST consisting of 42,000 digital images"
        # Chia dữ liệu cho người dùng dựa trên train_dataset hiện tại
        num_items = len(train_dataset)

        if iid:
            all_indices = np.random.permutation(num_items)
            split_indices = np.array_split(all_indices, num_users)
            dict_users = {user_id: split.tolist() for user_id, split in enumerate(split_indices)}
        else:
            # Fallback giữ tính dùng được nếu cần gọi non-IID ở nơi khác.
            all_indices = np.arange(num_items)
            split_indices = np.array_split(all_indices, num_users)
            dict_users = {user_id: split.tolist() for user_id, split in enumerate(split_indices)}

        return train_dataset, test_dataset, dict_users
