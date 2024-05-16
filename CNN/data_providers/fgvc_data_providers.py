import os
import torchvision
from ofa.imagenet_classification.data_providers import ImagenetDataProvider

__all__ = [
  'FGVCDataProvider',
  'AircraftDataProvider', 'CarDataProvider', 'Flowers102DataProvider', 'CUB200DataProvider', 'PetsDataProvider',
  'Food101DataProvider', 'CIFAR10DataProvider', 'CIFAR100DataProvider',
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
