import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional


def print_samples(
    samples: torch.Tensor,
    title: Optional[str] = None,
    img_size: int = 32,
    output_file: Optional[str] = None
) -> None:
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) * 255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((img_size, img_size, 3)))
    if isinstance(title, str):
        plt.suptitle(title)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, transparent=True)
    plt.show()
