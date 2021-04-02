import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Images
imgs = ['./bus.jpg']  # batched list of images

# Inference
results = model(imgs)

# Results
results.print()
results.save('./bus_new.jpg')  # or .show()

# Data
print(results.xyxy[0])
