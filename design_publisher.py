from google.cloud import pubsub_v1                
import json
import os 
import random
import numpy as np        
import time
from dotenv import load_dotenv
import csv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_account_key"
project_id = os.environ.get("PROJECT_ID")
topic_name = os.environ.get("DESIGN_TOPIC_NAME")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

csv_cols = ["Timestamp", "Car1_Location_X", "Car1_Location_Y", "Car1_Location_Z","Car2_Location_X","Car2_Location_Y","Car2_Location_Z","Occluded_Image_view","Occluding_Car_view","Ground_Truth_View","pedestrianLocationX_TopLeft","pedestrianLocationY_TopLeft","pedestrianLocationX_BottomRight","pedestrianLocationY_BottomRight"]

with open("./Labels.csv", mode='r') as file:
    csv_reader = csv.reader(file)

    for i,row in enumerate(csv_reader):
        if not i:
            continue
        if i == 3:
            break

        entry_data = {}
        for field_name, value in zip(csv_cols,row):
            entry_data[field_name] = value

        record_value=json.dumps(entry_data).encode('utf-8');

        try:    
            
            future = publisher.publish(topic_path, record_value);
            
            future.result()    
            print("The messages {} has been published successfully".format(record_value))
        except: 
            print("Failed to publish the message")
        
        time.sleep(.5)  
    


