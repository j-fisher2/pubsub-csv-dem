from google.cloud import pubsub_v1     
import glob                     
import json
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_account_key"

project_id = os.environ.get("PROJECT_ID")
topic_name = os.environ.get("DESIGN_TOPIC_NAME")
subscription_id = os.environ.get("DESIGN_SUBSCRIPTION_ID")

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)
topic_path = 'projects/{}/topics/{}'.format(project_id,topic_name);

print(f"Listening for messages on {subscription_path}..\n")

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    message_data = json.loads(message.data.decode('utf-8'));
    print("Consumed record with value : {}" .format(message_data))
    message.ack()
    
with subscriber:
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()