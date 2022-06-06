from canny import Canny

import torch as t
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tf
import torchvision as tv

import os
import matplotlib.pyplot as plt


def tensor_imshow(_img, dpi=100, axis='off'):
    _img = _img.cpu()
    _img = _img.sub( _img.min() )
    _img = _img.div( _img.max() ).clamp(0,1)
    topil = tf.ToPILImage()

    plt.figure(dpi=dpi)
    plt.axis(axis)
    plt.imshow(topil(_img))
    plt.show(block=False)


if __name__ == "__main__":
    device = t.device('cuda:0')
    # device = t.device('cpu')

    # create filter
    # canny_filter = Canny(thresh_lo=0.1, thresh_hi=0.2).to(device)
    canny_filter = t.jit.script(Canny(0.1, 0.2)).to(device)

    # load data
    proj_dir = os.path.split(__file__)[0]
    image_folder = os.path.join(proj_dir, "test_images")

    resize = tf.Resize(128)
    normalize = tf.Normalize(mean=0.5, std=0.1)

    dataset = tv.datasets.ImageFolder(image_folder, tf.Compose([resize, tf.ToTensor()]) )
    loader = DataLoader(dataset, batch_size=4)

    for data, labels in loader:
        for img in data: tensor_imshow(img, dpi=100)
        image_edges = canny_filter(data.to(device))
        for img in image_edges: tensor_imshow(img, dpi=100)
        break
