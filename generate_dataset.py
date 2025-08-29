from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from utils import set_seed
from config import config
from sklearn.model_selection import train_test_split

class ImageFolderDataset:
    def __init__(self, image_path, img_size=(224, 224), batch_size=32, valid_ratio=0.2, seed=42):
        # self.root_path = os.path.join(root_path, "data")
        self.image_path = image_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.seed = seed

        set_seed(self.seed)
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.7),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = None
        self.valid_dataset = None
        self.classes = None
        self.num_classes = None

    """load full dataset from the dataset path, then split into training / validation subsets 
        while preserving the class label distribution (stratified split).
        the resulting subsets are stored in self.train_dataset and self.valid_dataset
    """
    def split_dataset(self):
        full_dataset = datasets.ImageFolder(self.image_path, transform=self.transforms)
        targets = full_dataset.targets  # class of each image, e.g [0, 0, 1, 1, 1, 2, ...]
        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)
        
        # 依原資料集比例抽樣 (可用可不用)
        sss = StratifiedShuffleSplit(
            n_splits=1, # this para used to control cutting times, like cross validation
            test_size=self.valid_ratio,
            random_state=self.seed
        )

        train_idx, valid_idx = next(sss.split(range(len(full_dataset)), targets))   # ensure radio are same with original dataset
        self.train_dataset = Subset(full_dataset, train_idx)
        print(f"len of training dataset: {len(self.train_dataset)}")
        self.valid_dataset = Subset(full_dataset, valid_idx)
        print(f"len of validation dataset: {len(self.valid_dataset)}")

        

        targets = full_dataset.targets  # class of each image, e.g [0, 0, 1, 1, 1, 2, ...]
        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)

        sss = StratifiedShuffleSplit(
            n_splits=1, # this para used to control cutting times, like cross validation
            test_size=self.valid_ratio,
            random_state=self.seed
        )

        train_idx, valid_idx = next(sss.split(range(len(full_dataset)), targets))   # ensure radio are same with original dataset
        self.train_dataset = Subset(full_dataset, train_idx)
        print(f"len of training dataset: {len(self.train_dataset)}")
        self.valid_dataset = Subset(full_dataset, valid_idx)
        print(f"len of validation dataset: {len(self.valid_dataset)}")


        # full_dataset = datasets.ImageFolder(self.image_path, transform=self.transforms)
        # self.train_dataset, self.valid_dataset = train_test_split(full_dataset, test_size=0.3, random_state=self.seed)
        # rest, self.train_dataset = train_test_split(self.train_dataset, test_size=0.5, random_state=self.seed)
        # print(f"len of training dataset: {len(self.train_dataset)}")
        # print(f"len of validation dataset: {len(self.valid_dataset)}")


    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        print(f"Using {len(self.train_dataset)} images")
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def valid_dataloader(self):
        print(f"Using {len(self.valid_dataset)} images")
        return self.get_dataloader(self.valid_dataset, shuffle=False)



# ImageFolderDataset.set_global_seed(42)

data_module = ImageFolderDataset(
            image_path=config.image_path,
            img_size=(config.image_size, config.image_size), 
            batch_size=config.training_batch_size,
            valid_ratio=0.2, seed=42
)

data_module.split_dataset()
print(f'Classes Number: {data_module.classes}')