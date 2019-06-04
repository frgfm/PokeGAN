from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataloader(batch_size, image_size, data_dir='data'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1),
                                    transforms.RandomRotation(15),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(data_dir, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    f_min, f_max = feature_range
    x = x * (f_max - f_min) + f_min
    return x
