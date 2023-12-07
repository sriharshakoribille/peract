import gc
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm
from helpers.features.clip import clip
import numpy as np
from typing import Union

class CLIPArgs:
    model_name: str = "ViT-L/14@336px"
    skip_center_crop: bool = True
    batch_size: int = 64


@torch.no_grad()
def extract_clip_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    """Extract dense patch-level CLIP features for given images"""

    model, preprocess = clip.load(CLIPArgs.model_name, device=device)
    print(f"Loaded CLIP model {CLIPArgs.model_name}")

    # Patch the preprocess if we want to skip center crop
    if CLIPArgs.skip_center_crop:
        # Check there is exactly one center crop transform
        is_center_crop = [isinstance(t, CenterCrop) for t in preprocess.transforms]
        assert (
            sum(is_center_crop) == 1
        ), "There should be exactly one CenterCrop transform"
        # Create new preprocess without center crop
        preprocess = Compose(
            [t for t in preprocess.transforms if not isinstance(t, CenterCrop)]
        )
        print("Skipping center crop")

    # Preprocess the images
    images = [Image.open(path) for path in image_paths]
    preprocessed_images = torch.stack([preprocess(image) for image in images])
    preprocessed_images = preprocessed_images.to(device)  # (b, 3, h, w)
    print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")

    # Get CLIP embeddings for the images
    embeddings = []
    for i in tqdm(
        range(0, len(preprocessed_images), CLIPArgs.batch_size),
        desc="Extracting CLIP features",
    ):
        batch = preprocessed_images[i : i + CLIPArgs.batch_size]
        embeddings.append(model.get_patch_encodings(batch))
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings from flattened patches to patch height and width
    h_in, w_in = preprocessed_images.shape[-2:]
    if CLIPArgs.model_name.startswith("ViT"):
        h_out = h_in // model.visual.patch_size
        w_out = w_in // model.visual.patch_size
    elif CLIPArgs.model_name.startswith("RN"):
        h_out = max(h_in / w_in, 1.0) * model.visual.attnpool.spacial_dim
        w_out = max(w_in / h_in, 1.0) * model.visual.attnpool.spacial_dim
        h_out, w_out = int(h_out), int(w_out)
    else:
        raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
    embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    print(f"Extracted CLIP embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del model
    del preprocess
    del preprocessed_images
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings

class CLIPEmbedder:
    def __init__(self, args = CLIPArgs(), device = 'cuda') -> None:
        self.model, self.preprocess = clip.load(args.model_name, device=device)
        self.args = args
        self.device = device
        if self.args.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in self.preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            self.preprocess = Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, CenterCrop)]
            )
        print(f"Loaded CLIP model {CLIPArgs.model_name}")
    
    def __del__(self):
        del self.model
        del self.preprocess
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def image_embeddings(self, imgs: Union[List[str], List[torch.Tensor], List[np.ndarray]]) -> torch.Tensor:
        # Preprocess the images
        if isinstance(imgs[0], str):
            images = [Image.open(path) for path in imgs]
        elif isinstance(imgs[0], np.ndarray):
            images = [Image.fromarray(img) for img in imgs]
        elif isinstance(imgs[0], torch.Tensor):
            images = [Image.fromarray(img.cpu().numpy()) for img in imgs]
            # images = imgs
        preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        preprocessed_images = preprocessed_images.to(self.device)  # (b, 3, h, w)
        print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")

        # Get CLIP embeddings for the images
        img_embs = []
        for i in tqdm(
            range(0, len(preprocessed_images), CLIPArgs.batch_size),
            desc="Extracting CLIP features",
        ):
            batch = preprocessed_images[i : i + CLIPArgs.batch_size]
            img_embs.append(self.model.get_patch_encodings(batch))
        img_embs = torch.cat(img_embs, dim=0)

        # Reshape embeddings from flattened patches to patch height and width
        h_in, w_in = preprocessed_images.shape[-2:]
        if CLIPArgs.model_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
        elif CLIPArgs.model_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            h_out, w_out = int(h_out), int(w_out)
        else:
            raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
        img_embs = rearrange(img_embs, "b (h w) c -> b h w c", h=h_out, w=w_out)
        print(f"Extracted CLIP embeddings of shape {img_embs.shape}")

        img_embs /= img_embs.norm(dim=-1, keepdim=True)
        return img_embs
    
    @torch.no_grad()
    def text_embeddings(self, texts : List[str]) -> torch.Tensor:
        # Encode text query
        tokens = clip.tokenize(texts).to(self.device)
        text_embs = self.model.encode_text(tokens)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs

def update_clip_embs(dense_embedder, cameras, obs_dict, lang:str, cfg):
    if dense_embedder is not None:
        cam_imgs = []
        for cam in cameras:
            rgb= obs_dict['%s_rgb' % cam]
            cam_imgs.append(np.transpose(rgb, [1, 2, 0]))
        
        if cam_imgs:
            import cv2
            img_embs = dense_embedder.image_embeddings(cam_imgs)

            desc = lang.split()[-2] + ' ' + lang.split()[-1]
            # desc = lang
            print(desc)
            text_embs = dense_embedder.text_embeddings([desc])
            sims = img_embs @ text_embs.T
            sims = sims.squeeze().cpu().numpy()

            for i,cam in enumerate(cameras):
                sim_norm = (sims[i] - sims.min()) / (sims.max() - sims.min())
                sim_norm_scaled = cv2.resize((sim_norm * 255).astype(np.uint8), tuple(cfg.rlbench.camera_resolution))
                if cfg.method.no_rgb:
                    img_combined = np.expand_dims(sim_norm_scaled,axis=-1)
                else:
                    img_combined = np.concatenate([cam_imgs[i],np.expand_dims(sim_norm_scaled,axis=-1)],axis=-1)
                img_combined = np.transpose(img_combined, [2, 0, 1])
                obs_dict['%s_rgb' % cam] = img_combined
        else:
            assert False, "No camera images found in observation."