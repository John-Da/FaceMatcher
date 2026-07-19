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
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        print(f"[ReID] Loaded on device: {next(self.model.parameters()).device}")

    def _resolve_model_name(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]  # osnet_x1_0_imagenet
        name = name.replace("_imagenet", "")  # osnet_x1_0
        name = name.replace("_market", "")  # handle _market suffix too
        name = name.replace("_msmt17", "")  # and _msmt17
        return name

    def _preprocess_np(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize + normalize a single crop, return as (C, H, W) float32 array.
        Kept as pure numpy (no tensor/device work) so batch stacking is cheap.
        """
        img = cv2.resize(crop, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Single-crop version, kept for extract()."""
        arr = self._preprocess_np(crop)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return tensor.to(self.device)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a normalized embedding from a single person crop.
        Use for one-off extractions (e.g. the reference face image).
        For per-frame detections, use extract_batch instead.
        Returns: 1D numpy array (512,)
        """
        tensor = self._preprocess(crop)
        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = embedding.squeeze(0).cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        return embedding

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extract normalized embeddings for a list of crops in ONE forward pass.
        This is the one to use in the per-frame detection loop — avoids paying
        per-call dispatch/sync overhead once per person.
        Returns: list of 1D numpy arrays (512,), same order as input, [] if crops is empty.
        """
        if not crops:
            return []

        # Stack all crops into a single (N, C, H, W) batch on the CPU first —
        # only touch the device once with the full batch.
        batch_np = np.stack([self._preprocess_np(c) for c in crops], axis=0)
        batch_tensor = torch.from_numpy(batch_np).float().to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch_tensor)  # (N, 512)

        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6
        embeddings = embeddings / norms

        return [embeddings[i] for i in range(embeddings.shape[0])]

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized embeddings.
        Returns: float between 0.0 and 1.0
        """
        return float(np.dot(embedding_a, embedding_b))
