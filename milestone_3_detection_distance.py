import os
import cv2
import torch
import numpy as np
from pathlib import Path

# 1. Object Detection Setup (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Small model for speed

# 2. Depth Estimation Setup (MiDaS)
depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')  # Small for efficiency
depth_model.eval()
midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# 3. Dataset Path
dataset_path = './Dataset_Occluded_Pedestrian'
output_dir = './output_images/'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn’t exist
image_files = [f for f in Path(dataset_path).glob('*.png') if f.name.startswith(('A', 'C'))]
print(f"Found {len(image_files)} images: {[f.name for f in image_files]}")

def process_image(image_path):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path.name}")
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Object Detection
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]  # Bounding boxes in DataFrame

    # Filter for pedestrians (YOLOv5 'person' class is ID 0)
    pedestrians = detections[detections['class'] == 0]  # 'person' class

    if pedestrians.empty:
        print(f"No pedestrians detected in {image_path.name}")
        return []

    # Depth Estimation
    img_tensor = midas_transform(img_rgb).to('cpu')  # Preprocess for MiDaS
    with torch.no_grad():
        depth = depth_model(img_tensor)  # Raw depth map
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
        ).squeeze().cpu().numpy()  # Resize to image dims

    # Process each pedestrian and annotate image
    outputs = []
    for _, ped in pedestrians.iterrows():
        # Bounding box
        xmin, ymin, xmax, ymax = int(ped['xmin']), int(ped['ymin']), int(ped['xmax']), int(ped['ymax'])
        bbox = (xmin, ymin, xmax, ymax)

        # Crop depth map to pedestrian region (handle edge cases)
        depth_crop = depth[max(0, ymin):min(depth.shape[0], ymax), max(0, xmin):min(depth.shape[1], xmax)]
        if depth_crop.size == 0:
            print(f"Invalid depth crop for {image_path.name} at {bbox}")
            continue
        
        # Average depth
        avg_depth = np.mean(depth_crop)

        # Draw bounding box and depth on image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box
        label = f"Depth: {avg_depth:.2f}"
        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        outputs.append({
            'image': image_path.name,
            'bbox': bbox,
            'avg_depth': avg_depth
        })

    # Save annotated image
    output_path = os.path.join(output_dir, image_path.name)
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image: {output_path}")

    return outputs

# Main execution
results = []
for img_path in image_files:
    print(f"Processing {img_path.name}...")
    result = process_image(img_path)
    results.extend(result)

# Output results
if results:
    print("\nDetection Results:")
    for res in results:
        print(f"Image: {res['image']}, Bounding Box: {res['bbox']}, Avg Depth: {res['avg_depth']:.2f}")
else:
    print("No pedestrians detected in any images.")

# Save to file
with open('pedestrian_results.txt', 'w') as f:
    for res in results:
        f.write(f"Image: {res['image']}, BBox: {res['bbox']}, Depth: {res['avg_depth']:.2f}\n")
