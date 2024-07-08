# from IPython.display import display, Image, Audio
from PIL import Image
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import io
from openai import OpenAI
import os
import requests
import math
import json
from moviepy.editor import *

client = OpenAI()

step_classification_prompt = """The input images are frames extracted from camera on the person. The person is working on the following task:
{task}
Can you figure out which step the person is now working on?"""

step_prediction_prompt = """The input sequence of images are frames extracted from a short video. The video is about a comprehensive guidance to perform a specific task. 

The full task guideline is as follows:
{task}

The input images follows time order and stops on a specific step of the whole task. Can you reason step by step to predict what is the next step to compete in this task?
"""

def read_video(video_path):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames

def get_transcription(video_path):
    # Load the video file
    audio_file = video_path.replace("mp4", "mp3")
    if not os.path.exists(audio_file):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_file)
        video.close()
    audio_file= open(audio_file, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
        )
    print(transcript)
    with open(video_path.replace(".mp4", "_transcription.txt"), "w") as f:
        f.write(transcript.text)
    return transcript.text


def get_action_seq(frames, transcription):
    sample_interval = math.ceil(len(frames)/20)
    sampled_frames = frames[0::sample_interval]
    # action_segmentation_prompt = f"These are frames extracted from a video that I want to upload. Here is the trascripton to the audio of the video {transcription}. The trascription presents a sequence of actions to complete a task. Could you firstly split the long paragraph into a sequence of actions, in which each action is marked with their order number? Next, please align each frame to the corresponding action in the format: frame 1: Action 1."
    action_segmentation_prompt = f"These are frames extracted from a video that I want to upload. Here is the trascripton to the audio of the video {transcription}. The trascription presents a sequence of actions to complete a task. Could you split the long paragraph into a sequence of actions, in which each action is marked with their order number?"
    return get_gpt4_output(prompt=action_segmentation_prompt.format(transcription=transcription), seq_frames=sampled_frames)

def save_images(seq_frames, video_id, instruction_id):
    image_paths = []
    for i, frame in enumerate(seq_frames):
        with open(f"lavia/images/video_{video_id}_{instruction_id}_{i}.png", "wb") as fh:
            fh.write(base64.b64decode(frame))
        image_paths.append(f"lavia/images/video_{video_id}_{instruction_id}_{i}.png") 
    return image_paths


def get_gpt4_output(prompt, seq_frames):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768}, seq_frames),
            ],
        },
    ]


    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1000,
    }

    result = client.chat.completions.create(**params)
    # print(result.choices[0].message.content)
    return result.choices[0].message.content


if __name__=="__main__":
    video_id = 0
    video_path = "./samples/honda_engine_oil_change.mp4"
    print(video_path)
    video_frames = read_video(video_path=video_path)
    if os.path.exists(video_path.replace(".mp4", "_transcription.txt")):
        transcription = open(video_path.replace(".mp4", "_transcription.txt")).read()
    else:
        transcription = get_transcription(video_path=video_path)
    # action_seq = get_action_seq(video_frames, transcription)
    sampling_rate = 30
    num_images = 15
    interval = sampling_rate * num_images
    for index in range(0, len(video_frames), interval):
        seq_frames = video_frames[index: index+interval: sampling_rate]
        instruction_answer = get_gpt4_output(prompt=step_prediction_prompt.format(task=transcription), seq_frames=seq_frames)
        image_paths = save_images(seq_frames, video_id, index)
        sample = {"id": f"{video_id}_{index}", "image": "", "conversations": []}
        image_tokens = "".join(["<image>\n" for i in range(len(seq_frames))])
        prompt = image_tokens + step_prediction_prompt.format(task=transcription)
        sample['conversations'].append({'from': 'human', 'value': prompt})
        sample['conversations'].append({'from': 'gpt', 'value': instruction_answer})
        sample['image'] = image_paths
        print(sample)
        with open("./samples/{}.json".format(sample['id']), 'w') as file:
            file.write(json.dumps(sample, ensure_ascii=False, indent=2))