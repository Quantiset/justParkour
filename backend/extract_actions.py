import cv2
import numpy as np
import pickle
import json
import torch as th
from inverse_dynamics_model import IDMAgent
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

s3 = boto3.client("s3")
model_path = "4x_idm.model"
weights_path = "4x_idm.weights"
bucket_name = "minecraft-videos-dance"

def parse_id(name):
    video_path = f"{name}.mp4"
    output_path = f"{name}.jsonl"
    batch_size = 128

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping processing.")
        return

    s3.download_file(bucket_name, video_path, "local.mp4")
    video_path = "local.mp4"

    print("Loading model...")
    agent_parameters = pickle.load(open(model_path, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)

    print(f"Reading {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # The model expects a resolution of 640x360. 
        # If your video differs, you MUST resize it:
        if frame.shape[0] != 360 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 360))
            
        # Convert from BGR (OpenCV default) to RGB
        frames.append(frame[..., ::-1])

    cap.release()
    frames = np.stack(frames)
    total_frames = frames.shape[0]
    print(f"Loaded {total_frames} frames.")

    # 4. Predict Actions in Batches
    print("Predicting actions...")
    all_predicted_actions = []

    # Process in chunks to prevent running out of GPU memory
    for i in range(0, total_frames, batch_size):
        batch_frames = frames[i : i + batch_size]
        
        # IDMAgent.predict_actions returns a dict of arrays shape (1, batch_size)
        predicted_actions_dict = agent.predict_actions(batch_frames)
        
        # Reformat from dictionary-of-arrays to a list-of-dictionaries (one per frame)
        current_batch_size = batch_frames.shape[0]
        for j in range(current_batch_size):
            frame_actions = {}
            for action_name, action_array in predicted_actions_dict.items():
                value = action_array[0, j]
                # Handle multi-dimensional actions (e.g. camera [pitch, yaw])
                if hasattr(value, '__len__'):
                    frame_actions[action_name] = [int(v) for v in value]
                else:
                    frame_actions[action_name] = int(value)
            all_predicted_actions.append(frame_actions)

    with open(output_path, "w") as f:
        for entry in all_predicted_actions:
            f.write(json.dumps(entry) + "\n")

    print(f"Success! Actions saved to {output_path}")
