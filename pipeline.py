import os
import cv2
import json
import torch
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

class PedestrianDepthFn(beam.DoFn):
    def setup(self):
        # Load models once per worker
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.depth_model.eval()
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

    def process(self, element):
        # Element is a Pub/Sub message (e.g., {"image_path": "gs://.../A_002.png"})
        try:
            msg = json.loads(element.decode('utf-8'))
            img_path = msg['image_path']
        except Exception as e:
            print(f"Invalid message: {element}, error: {e}")
            return []

        # Download image from GCS
        gcs = beam.io.gcp.gcsio.GcsIO()
        try:
            img_bytes = gcs.open(img_path).read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
            if img is None:
                print(f"Failed to decode {img_path}")
                return []
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return []

        # Object Detection
        results = self.model(img_rgb)
        detections = results.pandas().xyxy[0]
        pedestrians = detections[(detections['class'] == 0) & (detections['confidence'] > 0.5)]

        if pedestrians.empty:
            print(f"No pedestrians in {img_path}")
            return []

        # Depth Estimation
        img_tensor = self.transform(img_rgb).to('cpu')
        with torch.no_grad():
            depth = self.depth_model(img_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
            ).squeeze().cpu().numpy()

        # Process pedestrians
        outputs = []
        for _, ped in pedestrians.iterrows():
            xmin, ymin, xmax, ymax = map(int, [ped['xmin'], ped['ymin'], ped['xmax'], ped['ymax']])
            bbox = [xmin, ymin, xmax, ymax]  # List for JSON serialization
            depth_crop = depth[max(0, ymin):min(depth.shape[0], ymax), max(0, xmin):min(depth.shape[1], xmax)]
            if depth_crop.size == 0:
                continue
            avg_depth = float(np.mean(depth_crop))  # Float for JSON

            outputs.append({
                'image': os.path.basename(img_path),
                'bbox': bbox,
                'avg_depth': avg_depth,
                'confidence': float(ped['confidence'])
            })

        return [json.dumps(output).encode('utf-8') for output in outputs]

def run():
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        data = (p 
                | 'ReadFromPubSub' >> beam.io.ReadFromPubSub(topic='projects/phrasal-chiller-451822-a1/topics/pedestrian_input')
                | 'ProcessPedestrians' >> beam.ParDo(PedestrianDepthFn())
                | 'WriteToPubSub' >> beam.io.WriteToPubSub(topic='projects/phrasal-chiller-451822-a1/topics/pedestrian_output'))

if __name__ == '__main__':
    run()