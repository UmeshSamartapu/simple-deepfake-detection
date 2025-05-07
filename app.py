
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch import nn
import glob
import os
from PIL import Image as pImage

# Define global variables
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std), std=np.divide([1,1,1],std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define Model
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Define Dataset
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    
    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

# Prediction function
def predict(model, img):
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

# Load model function
def load_model(model_path, sequence_length):
    model = Model(2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Running Prediction on a sample video
def run_prediction(video_path, model_path, sequence_length=60):
    model = load_model(model_path, sequence_length)
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)
    prediction, confidence = predict(model, dataset[0])
    return "REAL" if prediction == 1 else "FAKE", confidence

# Example usage
# video_path = "path/to/sample_video.mp4"
# model_path = "path/to/trained_model.pt"
# result, conf = run_prediction(video_path, model_path)
# print(f"Prediction: {result}, Confidence: {conf}%")

# Example usage
video_path = "rdjedt1.mp4"
model_path = "model.pt"
result, conf = run_prediction(video_path, model_path)
print(f"Prediction: {result}, Confidence: {conf}%")