import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

MODELS_PATH = "data/models/reids"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DINOv2ReID:

    INPUT_SIZE = (224, 224)

    # maps filename → torch hub model name
    MODEL_MAP = {
        "dinov2_vits14": "dinov2_vits14",  # small  — 384-dim
        "dinov2_vitb14": "dinov2_vitb14",  # base   — 768-dim
        "dinov2_vitl14": "dinov2_vitl14",  # large  — 1024-dim
    }

    def __init__(self, model_name: str):
        self.device = get_device()

        # strip extension to get base name
        base = model_name.replace(".pth", "").replace(".pt", "")
        hub_name = self.MODEL_MAP.get(base, "dinov2_vitb14")

        self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract normalized embedding from a BGR crop.
        Returns: 1D numpy array
        """
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = embedding.squeeze().cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        return embedding

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return float(np.dot(embedding_a, embedding_b))
