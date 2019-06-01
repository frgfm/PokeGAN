import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from PIL import Image
import glob
import os


def convert_to_png(origin_folder, dest_folder):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in glob.glob(os.path.join(origin_folder, '*.png')):
        img = Image.open(file)
        jpg = img.convert('RGB')
        # print(file)
        jpg.save(os.path.join(dest_folder, f"{file.split('/')[-1].split('.')[0]}.jpg")) 


# helper function for viewing a list of passed in sample images
def print_samples(samples, title=None, img_size=32):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) * 255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((img_size, img_size, 3)))
    if isinstance(title, str):
        plt.suptitle(title)


def print_gradflow(named_parameters, title=None, zoom=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            # Shorten the layer name
            if n.split('.')[0] != 'fc' and len(n.split('.')) >= 3:
                n_split = n.split('.')
                layer_idx = n_split[-3].split('_')[-1]
                layers.append(f"{n_split[-2]}_{layer_idx}.{n_split[-1]}")
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    if zoom:
        plt.ylim(bottom=-0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient value")
    if isinstance(title, str):
        plt.title(title)
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
