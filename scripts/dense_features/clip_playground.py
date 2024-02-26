import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from helpers.features import clip
from helpers.features.clip import tokenize
from helpers.features.clip_extract import CLIPArgs, extract_clip_features, CLIPEmbedder

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# _IMAGE_DIR = os.path.join(_MODULE_DIR, "images")
_IMAGE_DIR = os.path.join(_MODULE_DIR, "new_images_mine")

# image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.png", "frame_2.png", "frame_3.png"]]
# image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.png"]]
image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.jpg"]]

def plt_sims(imgs, sims, query, paths=False):
    plt.figure()
    cmap = plt.get_cmap("jet")
    for idx, (img, sim) in enumerate(zip(imgs, sims)):
        plt.subplot(2, len(imgs), idx + 1)
        if paths:
            plt.imshow(Image.open(img))
        else:
            plt.imshow(imgs[idx])
        # plt.title(os.path.basename(image_path))
        plt.axis("off")

        plt.subplot(2, len(imgs), len(imgs) + idx + 1)
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
        heatmap = cmap(sim_norm.cpu().numpy())
        plt.imshow(heatmap)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query \'{query}\'')
    plt.show()

def plt_single(img, sim, query, paths=False):
    plt.figure()
    cmap = plt.get_cmap("jet")
    plt.subplot(1, 2, 1)
    if paths:
        plt.imshow(Image.open(img))
    else:
        plt.imshow(img)
    # plt.title(os.path.basename(image_path))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
    heatmap = cmap(sim_norm.cpu().numpy())
    plt.imshow(heatmap)
    # plt.colorbar()  # Add colorbar for the heatmap
    plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query \'{query}\'')
    plt.show()

def plt_vertical(img, sim, query, paths=False):
    plt.figure()
    cmap = plt.get_cmap("jet")
    plt.subplot(2, 1, 1)
    if paths:
        plt.imshow(Image.open(img))
    else:
        plt.imshow(img)
    # plt.title(os.path.basename(image_path))
    plt.axis("off")

    plt.subplot(2, 1, 2)
    sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
    heatmap = cmap(sim_norm.cpu().numpy())
    plt.imshow(heatmap)
    # plt.colorbar()  # Add colorbar for the heatmap
    plt.axis("off")

    plt.suptitle(f'Similarity to language query \'{query}\'')
    plt.tight_layout()
    plt.show()

def plt_single(img, sim, query, paths=False):
    
    cmap = plt.get_cmap("jet")
    
    sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
    heatmap = cmap(sim_norm.cpu().numpy())
    plt.imshow(heatmap)
    # plt.colorbar()  # Add colorbar for the heatmap
    plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query \'{query}\'')
    plt.show()
    
@torch.no_grad()
def feats(text_query:str):
    embedder = CLIPEmbedder()
    img_embs = embedder.image_embeddings(image_paths)
    text_embs = embedder.text_embeddings([text_query])
    sims = img_embs @ text_embs.T
    # plt_sims(image_paths, sims.squeeze(), text_query, paths=True)
    plt_vertical(image_paths[0], sims[0].squeeze(), text_query, paths=True)
    # plt_single(image_paths[0], sims[0].squeeze(), text_query, paths=True)

if __name__=="__main__":
    feats(text_query="Traffic cone")
    # plt.imshow(Image.open(image_paths[0]))
    # plt.show()
