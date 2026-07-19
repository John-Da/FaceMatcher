# import torch
# import torchreid

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())

# osnet_model = torchreid.models.build_model(
#     name="osnet_x1_0",  # architecture name
#     num_classes=1000,
#     pretrained=False,
# )

# torchreid.utils.load_pretrained_weights(
#     osnet_model, "data/models/reids/osnet_x1_0_imagenet.pth"
# )

# osnet_model.eval()
# print("OSNet loaded successfully")

# dinov2_vits14_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# torch.save(dinov2_vits14_model.state_dict(), "data/models/reids/dinov2_vits14.pt")
# print("Dinov2 Vits14 loaded successfully")


import cv2
import time

index = 0  # match your camera index
cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

print("Reported FPS (cap.get):", cap.get(cv2.CAP_PROP_FPS))
print(
    "Reported resolution:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
)
print(
    "FourCC:",
    int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore"),
)

# Measure ACTUAL delivered fps, no throttling, no processing
frame_count = 0
t_start = time.time()
while time.time() - t_start < 5.0:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

elapsed = time.time() - t_start
print(f"Measured raw capture fps over {elapsed:.1f}s: {frame_count / elapsed:.1f}")
cap.release()
