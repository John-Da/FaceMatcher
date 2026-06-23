import os
import cv2
import numpy as np
import torch
import torchreid

MODELS_PATH = "data/models/reids"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_reid(model_name: str):
    base = model_name.lower()
    if "dinov2" in base:
        from core.dinov2_reid import DINOv2ReID

        return DINOv2ReID(model_name)
    else:
        return ReID(model_name)

class ReID:

    INPUT_SIZE = (256, 128)  # standard OSNet input: (height, width)

    def __init__(self, model_name: str):
        self.device = get_device()

        self.model = torchreid.models.build_model(
            name=self._resolve_model_name(model_name),
            num_classes=1000,
            pretrained=False,
        )

        weight_path = os.path.join(MODELS_PATH, model_name)
        torchreid.utils.load_pretrained_weights(self.model, weight_path)

        self.model.eval()
        self.model.to(self.device)

    def _resolve_model_name(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]       # osnet_x1_0_imagenet
        name = name.replace("_imagenet", "")        # osnet_x1_0
        name = name.replace("_market", "")          # handle _market suffix too
        name = name.replace("_msmt17", "")          # and _msmt17
        return name

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Resize crop → normalize → convert to tensor.
        Input:  BGR numpy array (H, W, 3)
        Output: tensor (1, 3, H, W) on device
        """
        img = cv2.resize(crop, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a normalized embedding from a single person crop.
        Returns: 1D numpy array (512,)
        """
        tensor = self._preprocess(crop)
        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = embedding.squeeze().cpu().numpy()
        # L2 normalize so cosine similarity = dot product
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        return embedding

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embeddings.
        Returns: float between 0.0 and 1.0
        """
        return float(np.dot(embedding_a, embedding_b))
