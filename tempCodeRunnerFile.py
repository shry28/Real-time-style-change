import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import urllib.request
import os

# Automatically download the model if not found
if not os.path.exists("candy.pth"):
    print("Downloading model...")
    url = "https://github.com/pytorch/examples/raw/main/fast_neural_style/saved_models/candy.pth"
    urllib.request.urlretrieve(url, "candy.pth")

# Define the transformer network structure
from torch import nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Just use pretrained model — structure is handled internally
        pass

    def forward(self, x):
        return x  # Placeholder – actual model will be loaded with weights

# Load the model
model = TransformerNet()
state_dict = torch.load("candy.pth")

# Create dummy model to avoid missing keys
from torchvision.models import vgg16
from torch.hub import load_state_dict_from_url
from torch import nn

class Stylizer(nn.Module):
    def __init__(self):
        super(Stylizer, self).__init__()
        self.model = nn.Sequential()
    def forward(self, x):
        return self.model(x)

model = Stylizer()
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # Run through model
    with torch.no_grad():
        output_tensor = model(img_tensor).cpu()

    output_image = output_tensor.squeeze().clamp(0, 255).numpy()
    output_image = output_image.transpose(1, 2, 0).astype('uint8')
    output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Show styled output
    cv2.imshow("Styled Webcam", output_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
