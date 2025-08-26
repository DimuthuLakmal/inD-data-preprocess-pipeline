import cv2
import numpy as np
from pathlib import Path
import re
import os
import math

from loguru import logger
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_frame(map_img, vehicle_data, timestep, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img)

    for seq in vehicle_data:
        if timestep < len(seq):
            x, y, heading, ax_, ay_ = seq[timestep]
            rect = patches.FancyArrow(x, y, 10*np.cos(heading), 10*np.sin(heading),
                                      width=1, color='red')
            ax.add_patch(rect)

    ax.set_axis_off()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


scene_ids = set()
for fname in os.listdir("../data"):
    match = re.match(r"(\d+)_.*\.csv", fname)
    if match:
        scene_ids.add(match.group(1))

scene_ids = sorted(scene_ids)
print(f"Found scenes: {scene_ids}")

input_path = "../data"

tracks = []
tracks_meta = []
visibility_data = []
background_images = {}
fixed_blocks_info = {}
frame_to_track_idxs = {}

start_scene = 0
end_scene = 21
filename = "data.pkl"

data_dict = {}

# Check data file exists
index_file_path = Path(os.path.join(input_path, filename))
if index_file_path.exists():
    logger.info("Loading index map from {}", index_file_path)
    data_dict = pickle.load(open(index_file_path, "rb"))

    # Loading background images
    for scene_id in scene_ids:
        if int(scene_id) < start_scene or int(scene_id) > end_scene:
            continue

        # Store background images for scenes
        bg_path = os.path.join(input_path, 'semantic_maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[int(scene_id)] = img

keys = list(data_dict.keys())
print("Done Loading")

for idx in range(len(keys)):
    key = keys[idx]
    keys = key.split("_")
    scene_id = int(keys[0])
    current_frame = int(keys[1])
    ego_vehicle_track_idx = int(keys[2])

    data_dict = data_dict[key]

    # map
    backgrond_img = background_images[scene_id]

    # Historical observations
    historical_adjacent_obs, historical_ego_obs = data_dict["historical_adjacent_obs"], data_dict["historical_ego_obs"]

    vehicle_data = []
    for historical_adjacent_ob in historical_adjacent_obs.values():
        x = historical_adjacent_ob[0] * backgrond_img.shape[1]
        y = historical_adjacent_ob[1] * backgrond_img.shape[0]
        heading = historical_adjacent_ob[2]
        ax = historical_adjacent_ob[3]
        ay = historical_adjacent_ob[4]
        vehicle_data.append((x, y, heading, ax, ay))

    num_timesteps = 21

    for t in range(num_timesteps):
        draw_frame(map_img, vehicle_data, t, save_path=f"frames/frame_{t:03d}.png")

    image_folder = 'frames'
    video_name = 'vehicle_animation.mp4'
    frame_rate = 5  # fps

    frame_array = []
    for t in range(num_timesteps):
        filename = f"{image_folder}/frame_{t:03d}.png"
        img = cv2.imread(filename)
        height, width, _ = img.shape
        frame_array.append(img)

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for frame in frame_array:
        out.write(frame)
    out.release()


def interactive_playback(image_folder, total_frames):
    fig, ax = plt.subplots()
    img = mpimg.imread(f"{image_folder}/frame_000.png")
    im = ax.imshow(img)
    ax.set_title("Use arrow keys to scroll frames")

    def on_key(event):
        nonlocal frame_idx
        if event.key == 'right':
            frame_idx = (frame_idx + 1) % total_frames
        elif event.key == 'left':
            frame_idx = (frame_idx - 1) % total_frames
        img = mpimg.imread(f"{image_folder}/frame_{frame_idx:03d}.png")
        im.set_data(img)
        fig.canvas.draw_idle()

    frame_idx = 0
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# Run interactive viewer
interactive_playback("frames", num_timesteps)





