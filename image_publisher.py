from google.cloud import pubsub_v1                
import json
import os 
import random
import numpy as np        
import time
from dotenv import load_dotenv
import csv
import base64

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_account_key"
project_id = os.environ.get("PROJECT_ID")
topic_name = os.environ.get("IMAGE_TOPIC_NAME")

publisher_options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
publisher = pubsub_v1.PublisherClient( publisher_options=publisher_options)
topic_path = publisher.topic_path(project_id, topic_name)

key="image";

def publish_image(image_path, key="image"):
    try:    
        with open(image_path, "rb") as f:
            value =  base64.b64encode(f.read());  
        
        future = publisher.publish(topic_path, value, ordering_key=key);
        future.result()    
        print("The messages has been published successfully")
    except: 
        print("Failed to publish the message")

def publish_images_from_directory(directory_path):
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        print(image_path)
        print(f"Publishing image: {image_path}")
        publish_image(image_path)
    
publish_images_from_directory("Dataset_Occluded_Pedestrian")