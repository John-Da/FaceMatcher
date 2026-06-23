import torch
import torchreid

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())

osnet_model = torchreid.models.build_model(
    name="osnet_x1_0",  # architecture name
    num_classes=1000,
    pretrained=False,
)

torchreid.utils.load_pretrained_weights(
    osnet_model, "data/models/reids/osnet_x1_0_imagenet.pth"
)

osnet_model.eval()
print("OSNet loaded successfully")

dinov2_vits14_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
torch.save(dinov2_vits14_model.state_dict(), "data/models/reids/dinov2_vits14.pt")
print("Dinov2 Vits14 loaded successfully")
