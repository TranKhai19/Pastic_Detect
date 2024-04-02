import time

import torch
import torchaudio
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sounddevice as sd

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transformations for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load audio files (adjust file paths if needed)
ding_sound, metadata_ding = torchaudio.load("ding.wav", format="wav")
beep_sound, metadata_beep = torchaudio.load("beep.wav", format="wav")

# Function to detect plastic waste
def detect_plastic(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # If there are detections
    if len(output[0]["labels"]) > 0:
        # Get detected labels
        detected_labels = output[0]["labels"].tolist()

        # Filter for plastic waste labels
        plastic_labels = [44, 47, 67]  # Adjust as needed
        if any(label in detected_labels for label in plastic_labels):
            print("Plastic waste detected!")

            # Check number of channels in ding_sound (assuming same for beep_sound)
            if ding_sound.shape[0] == 1:  # If mono
                # Convert mono to stereo by duplicating the mono channel
                ding_sound_stereo = np.repeat(ding_sound.numpy(), 2, axis=0)
                sd.play(ding_sound_stereo.T, samplerate=metadata_ding)  # Transpose for correct channel order
            else:
                sd.play(ding_sound.numpy().T, samplerate=metadata_ding)  # Play as is (stereo)
            time.sleep(1)
            return

    print("No plastic waste detected.")
    sd.play(beep_sound.numpy().T, samplerate=metadata_beep)  # Play beep sound
    time.sleep(1)

# Example usage
image_path = "media.jpg"
detect_plastic(image_path)
