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


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def get_vert(x, y, heading, length=10.0, width=10.0):
    """
    Returns Nx2 array of polygon vertices for a rectangle centered at (x, y)
    rotated by `heading` (radians). length/width are in the same units as x,y.
    """
    # rectangle corners in the vehicle's local frame (centered at origin)
    L, W = length, width
    local = np.array([
        [+L / 2, +W / 2],
        [+L / 2, -W / 2],
        [-L / 2, -W / 2],
        [-L / 2, +W / 2],
    ])

    R = _rot2d(heading)
    return (local @ R.T) + np.array([x, y])

def draw_cells(ax, x, y, heading, label, cell_size=20, as_center=True):
    """
    cells: list of tuples (x, y, is_black) where is_black ∈ {0,1}
           x,y in the same pixel coord system as your map.
    cell_size: side length in pixels.
    as_center: True if (x,y) is the cell center; False if it's top-left.
    """

    verts = get_vert(x, y, heading, length=cell_size, width=cell_size)
    face = 'black' if label else 'white'
    edge = 'white' if label else 'black'
    ax.add_patch(patches.Polygon(verts, closed=True,
                                 facecolor=face))


def draw_poly(ax, vehicle, color, timestep, map_img=None):
    x, y, heading, ax_, ay_ = vehicle[timestep][0], vehicle[timestep][1], vehicle[timestep][2], vehicle[timestep][
            3], vehicle[timestep][4]
    x = x * map_img.shape[1]
    y = y * map_img.shape[0]
    verts = get_vert(x, y, heading)
    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=color,
                                     edgecolor=color, linewidth=0.8))


def draw_frame(map_img, adj_vehicle_data, ego_vehicle_data, cells, timestep, num_timesteps, save_path=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img)

    # Draw cells ONLY on the last timestep
    if (cells is not None) and (timestep == num_timesteps - 1):
        for cell in cells:
            x = cell[0] * map_img.shape[1]
            y = cell[1] * map_img.shape[0]
            label = cell[2]
            draw_cells(ax, x, y, ego_vehicle_data[timestep][2], label, cell_size=20, as_center=True)

    for vehicle in adj_vehicle_data:
        draw_poly(ax, vehicle, 'brown', timestep, map_img)

    draw_poly(ax, ego_vehicle_data, 'black', timestep, map_img)

    ax.set_axis_off()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

    print("Frame drawn")


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
    hidden_ogm_cells = data_dict["hidden_ogm_cells"]

    num_timesteps = 21

    for t in range(num_timesteps):
        draw_frame(backgrond_img, historical_adjacent_obs.values(), historical_ego_obs, hidden_ogm_cells, t,
                   num_timesteps)

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
