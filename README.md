For GCP data pipeline: 

1. Command to test dataflow job `python pipeline.py   --runner DirectRunner   --project $PROJECT   --streaming`
2. Command to test data job responsiveness `gcloud pubsub topics publish pedestrian_input \
--message='{"image_path": "gs://path-to-gs-image"}' \
--project=$PROJECT`
