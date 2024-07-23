import os
import torchvision
from ofa.imagenet_classification.data_providers import ImagenetDataProvider

__all__ = [
  'FGVCDataProvider',
  'AircraftDataProvider', 'CarDataProvider', 'Flowers102DataProvider', 'CUB200DataProvider', 'PetsDataProvider',
  'Food101DataProvider', 'CIFAR10DataProvider', 'CIFAR100DataProvider', 'ImageNet_C_DataProvider'
]


class FGVCDataProvider(ImagenetDataProvider):

  @staticmethod
  def name():
    raise not NotImplementedError

  @property
  def n_classes(self):
    raise not NotImplementedError

  @property
  def save_path(self):
    raise not NotImplementedError


class AircraftDataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'aircraft'

  @property
  def n_classes(self):
    return 100

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')
  
  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.FGVCAircraft(self.save_path, split='train', annotation_level='family', transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.FGVCAircraft(self.save_path, split='test', annotation_level='family', transform=_transforms, download=True)
    return dataset


class CarDataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'car'

  @property
  def n_classes(self):
    return 196

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')
  
  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.StanfordCars(self.save_path, split='train', transform=_transforms, download=False)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.StanfordCars(self.save_path, split='test', transform=_transforms, download=False)
    return dataset


class Flowers102DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'flowers102'

  @property
  def n_classes(self):
    return 102

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')

  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.Flowers102(self.save_path, split='train', transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.Flowers102(self.save_path, split='test', transform=_transforms, download=True)
    return dataset


class Food101DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'food101'

  @property
  def n_classes(self):
    return 101

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')
  
  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.Food101(self.save_path, split='train', transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.Food101(self.save_path, split='test', transform=_transforms, download=True)
    return dataset


class CUB200DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'cub200'

  @property
  def n_classes(self):
    return 200

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')


class PetsDataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'pets'

  @property
  def n_classes(self):
    return 37

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')
  
  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.OxfordIIITPet(self.save_path, split='trainval', transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.OxfordIIITPet(self.save_path, split='test', transform=_transforms, download=True)
    return dataset


class CIFAR10DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'cifar10'

  @property
  def n_classes(self):
    return 10

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset/cifar10')

  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.CIFAR10(self.save_path, train=True, transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.CIFAR10(self.save_path, train=False, transform=_transforms, download=True)
    return dataset


class CIFAR100DataProvider(CIFAR10DataProvider):

  @staticmethod
  def name():
    return 'cifar100'

  @property
  def n_classes(self):
    return 100

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset/cifar100')

  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.CIFAR100(self.save_path, train=True, transform=_transforms, download=True)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.CIFAR100(self.save_path, train=False, transform=_transforms, download=True)
    return dataset

class ImageNetDataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'ImageNet'

  @property
  def n_classes(self):
    return 1000

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset/ImageNet2012')

  def train_dataset(self, _transforms):
    dataset = torchvision.datasets.ImageNet(self.save_path, split='train', transform=_transforms)
    return dataset

  def test_dataset(self, _transforms):
    dataset = torchvision.datasets.ImageNet(self.save_path, split='val', transform=_transforms)
    return dataset

### Corruption Datasets ###
import numpy as np
from robustbench.data import load_cifar10c
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

class CustomVisionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, tensor_dataset, transform=None, target_transform=None):
        super(CustomVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target
class CIFAR10_C_DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'cifar10-c'

  @property
  def n_classes(self):
    return 10

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')

  def __init__(self, corruption_type=[], severity=5, train_n=1000, **kwargs):
    
    self.corruption_type = corruption_type
    self.severity = severity
    self.train_n = train_n

    x_corr, y_corr = load_cifar10c(
        10000, self.severity, self.save_path, False, [self.corruption_type]
    )

    labels = {}
    num_classes = int(max(y_corr)) + 1
    for i in range(num_classes):
        labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
    num_ex = self.train_n // num_classes
    tr_idxs = []
    val_idxs = []
    test_idxs = []
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        # tr_idxs.append(labels[i][:num_ex])
        # val_idxs.append(labels[i][num_ex:num_ex+10])
        tr_idxs.append(labels[i][:num_ex+10])
        test_idxs.append(labels[i][num_ex+10:num_ex+100])
    tr_idxs = np.concatenate(tr_idxs)
    # val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)
    
    self.train_data = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
    # self.val_data = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
    self.test_data = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])

    super().__init__(**kwargs)

  def train_dataset(self, _transforms):
    # Extract the list of transforms
    transform_list = _transforms.transforms
    # Filter out the ToTensor transform
    filtered_transforms = [t for t in transform_list if not isinstance(t, transforms.ToTensor)]
    # Create a new transforms.Compose object without ToTensor
    new_transforms = transforms.Compose(filtered_transforms)

    return CustomVisionDataset(self.train_data, transform=new_transforms)

  ### split part of train_dataset as valid_dataset; do not use valid_dataset during training; pick best_val_model as final model for test
  
  def test_dataset(self, _transforms):
    # Extract the list of transforms
    transform_list = _transforms.transforms
    # Filter out the ToTensor transform
    filtered_transforms = [t for t in transform_list if not isinstance(t, transforms.ToTensor)]
    # Create a new transforms.Compose object without ToTensor
    new_transforms = transforms.Compose(filtered_transforms)

    return CustomVisionDataset(self.test_data, transform=new_transforms)

class ImageNet_C_DataProvider(FGVCDataProvider):

  @staticmethod
  def name():
    return 'imagenet-c'

  @property
  def n_classes(self):
    return 1000

  @property
  def save_path(self):
    return os.path.expanduser('~/dataset')

  def __init__(self, corruption_type=[], severity=5, train_n=1000, **kwargs):
    
    self.corruption_type = corruption_type
    self.severity = severity
    self.train_n = train_n

    data_root = os.path.expanduser('~/dataset')
    image_dir = os.path.join(data_root, 'imagenet-c', corruption_type, str(severity))
    # dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
    dataset = ImageFolder(image_dir)
    indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
    assert self.train_n <= 20000
    labels = {}
    y_corr = dataset.targets
    for i in range(max(y_corr)+1):
        labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
    num_ex = self.train_n // (max(y_corr)+1)
    tr_idxs = []
    val_idxs = []
    test_idxs = []
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        # tr_idxs.append(labels[i][:num_ex])
        # val_idxs.append(labels[i][num_ex:num_ex+10])
        tr_idxs.append(labels[i][:num_ex+10])
        test_idxs.append(labels[i][num_ex+10:num_ex+20])
    tr_idxs = np.concatenate(tr_idxs)
    # val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)

    self.train_data = Subset(dataset, tr_idxs)
    # self.val_data = Subset(dataset, val_idxs)
    self.test_data = Subset(dataset, test_idxs)

    super().__init__(**kwargs)

  def train_dataset(self, _transforms):
    return CustomVisionDataset(self.train_data, transform=_transforms)

  ### split part of train_dataset as valid_dataset; do not use valid_dataset during training; pick best_val_model as final model for test
  
  def test_dataset(self, _transforms):
    return CustomVisionDataset(self.test_data, transform=_transforms)