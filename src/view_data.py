import cv2
import numpy as np
from pathlib import Path
import re
import os
import json

from loguru import logger
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.widgets import Button
from shapely.geometry import Polygon as ShapelyPolygon


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
    a = np.deg2rad(heading)
    R = _rot2d(a)
    return (local @ R.T) + np.array([x, y])

def draw_cells(ax, x, y, heading, label, cell_size=20, as_center=True):
    """
    cells: list of tuples (x, y, is_black) where is_black ∈ {0,1}
           x,y in the same pixel coord system as your map.
    cell_size: side length in pixels.
    as_center: True if (x,y) is the cell center; False if it's top-left.
    """

    verts = get_vert(x, y, heading, length=cell_size, width=cell_size)
    if label == 1:
        face = 'black'
    elif label == 0:
        face = 'white'
    elif label == 0.7:
        face = 'gray'
    elif label == 3:
        face = 'green'
    ax.add_patch(patches.Polygon(verts, closed=True,
                                 facecolor=face, picker=True,))


def draw_circle(ax, vehicle, color, timestep, map_img=None):
    x, y, heading, vx_, vy_, vehicle_type = vehicle[timestep][0], vehicle[timestep][1], vehicle[timestep][2], vehicle[timestep][
            3], vehicle[timestep][4], vehicle[timestep][7]
    x = x * map_img.shape[1]
    y = y * map_img.shape[0]

    if vehicle_type == 2.0:  # Bycle
        color='pink'
    elif vehicle_type == 3.0:  # Pedestrian
        color='purple'

    ax.add_patch(patches.Circle((x,y), 10, facecolor=color))

def draw_poly(ax, poly_pts, color):
    ax.add_patch(patches.Polygon(poly_pts, True, facecolor=color))


def overlay_clickable_polygons(ax, cell_coords, visibilities, labels, edge_unselected="yellow", edge_selected="red",
                               hidden_track_ids=None,
                               face_unselected=(1,1,0,0.10),  # light fill for easier picking
                               face_selected=(1,0,0,0.25)):
    """
    polygons: list of Nx2 numpy arrays (vertex coordinates in image/data space), or anything iterable of (x,y).
    info_for_each: optional list of arbitrary metadata objects (len must match polygons).
                   If provided, each polygon gets its corresponding info attached.

    Returns: (patches_list, disconnect_fn)
    """
    id_to_data = {}
    patch_to_id = {}

    # draw patches with picking enabled
    drawn = []
    for i, (pts, visibility, label) in enumerate(zip(cell_coords, visibilities, labels)):
        alpha=0.5
        if visibility == 1:
            face = 'yellow'
            alpha=1
        elif visibility == 0:
            face = 'white'
        elif visibility == 0.7 or visibility == 0.5:
            face = 'gray'
        elif visibility == 3:
            face = 'green'
        else:
            face = 'orange'
        p = patches.Polygon(pts, closed=True, facecolor=face, picker=True, alpha=alpha, edgecolor=edge_unselected)
        ax.add_patch(p)
        drawn.append(p)

        shapely_polygon = ShapelyPolygon(pts)
        centroid_point = shapely_polygon.centroid

        row = int(i/20) # assuming 20 cells per row
        col = int(i%20)

        track_ids = -1
        if hidden_track_ids is not None:
            track_ids = hidden_track_ids.get((row, col), -1)

        id_to_data[i] = {
            "index": i,
            "label": label,  # raw vertices
            "centroid": [centroid_point.x, centroid_point.y],
            "track_ids": [track_ids]
        }

        patch_to_id[p] = i

    # return drawn, _disconnect
    return {
            "patches": drawn,
            "id_to_data": id_to_data,
            "patch_to_id": patch_to_id,
            "styles": {
                "edge_unselected": edge_unselected,
                "edge_selected": edge_selected,
            }
        }


# --- helper: figure -> RGB array ---
def _fig_to_rgb_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    arr = buf.reshape((h, w, 3))
    plt.close(fig)  # important to avoid piling up figures
    return arr


def draw_frame(map_img, adj_vehicle_data, ego_vehicle_data, hidden_tracks_pts, timestep, num_timesteps):

    fig, ax = plt.subplots(figsize=(map_img.shape[1] / 100, map_img.shape[0] / 100), dpi=100)
    ax.imshow(map_img)

    for vehicle in adj_vehicle_data:
        draw_circle(ax, vehicle, 'yellow', timestep, map_img)

    draw_circle(ax, ego_vehicle_data, 'red', timestep, map_img)

    if hidden_tracks_pts is not None:
        for pts in hidden_tracks_pts:
            draw_poly(ax, pts, 'black')

    ax.set_axis_off()
    fig.tight_layout(pad=0)

    # return the rendered image instead of showing/saving
    return _fig_to_rgb_array(fig)


def interactive_playback(frames, omg_cells, visibilities, labels, map_shape, on_save=None, key=None, hidden_tracks_id_ogm_data=None):
    """frames: list of HxWx3 uint8 arrays"""
    fig, ax = plt.subplots(figsize=(map_shape[1] / 100, map_shape[0] / 100), dpi=100)
    im = ax.imshow(frames[0])
    ax.set_axis_off()
    ax.set_title(f"Scene: {key}")

    ax_save = fig.add_axes([0.13, 0.02, 0.12, 0.05])
    ax_clear = fig.add_axes([0.27, 0.02, 0.12, 0.05])
    ax_next = fig.add_axes([0.41, 0.02, 0.12, 0.05])  # NEW: Next button

    btn_save = Button(ax_save, "Save")
    btn_clear = Button(ax_clear, "Clear")
    btn_next = Button(ax_next, "Next")

    status_text = fig.text(0.70, 0.03, "0 selected | 0 saves", ha="left", va="center")

    frame_idx = {"i": 0}

    frame_i = 0  # (if not already defined above)
    frame_text = ax.text(
        0.02, 0.02,  # x,y in axes coords (0-1)
        f"Frame {frame_i + 1}/{len(frames)}",  # initial text
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
        zorder=10
    )

    overlays = {"patches": [], "id_to_data": {}, "patch_to_id": {}, "styles": {}}
    selected_ids = set()
    saved_clicks = []  # list of lists (snapshot of selected payloads at each Save)
    pick_cid = {"id": None}

    def _update_status():
        status_text.set_text(f"{len(selected_ids)} selected | {len(saved_clicks)} saves")
        fig.canvas.draw_idle()

    def _apply_style(poly, selected: bool):
        st = overlays["styles"]
        if not st:  # no overlays present
            return
        if selected:
            poly.set_edgecolor(st["edge_selected"])
        else:
            poly.set_edgecolor(st["edge_unselected"])

    def _collect_selected_payload():
        # Build a stable list of payload dicts from current selection
        payload = []
        for pid in sorted(selected_ids):
            d = overlays["id_to_data"][pid]
            payload.append({
                "index": int(d["index"]),
                "centroid": (float(d["centroid"][0]), float(d["centroid"][1])),
                "label": d["label"],
                "track_ids": d["track_ids"],
            })
        return payload

    def _disconnect_picker():
        if pick_cid["id"] is not None:
            fig.canvas.mpl_disconnect(pick_cid["id"])
            pick_cid["id"] = None

    def _clear_overlays(remove_patches=True):
        nonlocal overlays, selected_ids
        _disconnect_picker()
        if remove_patches:
            for p in overlays["patches"]:
                p.remove()
        overlays = {"patches": [], "id_to_data": {}, "patch_to_id": {}, "styles": {}}
        selected_ids = set()

    def _maybe_overlay_last_frame():
        _clear_overlays(remove_patches=True)
        if (frame_idx["i"] == len(frames) - 1) and len(omg_cells) > 0:
            # Draw overlays
            ovr = overlay_clickable_polygons(ax, omg_cells, visibilities, labels, hidden_track_ids=hidden_tracks_id_ogm_data)
            overlays.update(ovr)

            # Picker handler (toggle selection)
            def _on_pick(event):
                artist = event.artist
                if artist in overlays["patch_to_id"]:
                    pid = overlays["patch_to_id"][artist]
                    if pid in selected_ids:
                        selected_ids.remove(pid)
                        _apply_style(artist, False)
                    else:
                        selected_ids.add(pid)
                        _apply_style(artist, True)
                    fig.canvas.draw_idle()
                    _update_status()

            pick_cid["id"] = fig.canvas.mpl_connect('pick_event', _on_pick)

    def on_key(event):
        if (event.key == 'right' or event.key == 'd') and frame_idx["i"] != len(frames) - 1:
            frame_idx["i"] = (frame_idx["i"] + 1) % len(frames)
        elif (event.key == 'left' or event.key == 'a') and frame_idx["i"] != 0:
            frame_idx["i"] = (frame_idx["i"] - 1) % len(frames)
        im.set_data(frames[frame_idx["i"]])
        frame_text.set_text(f"Frame {(frame_idx['i'] + 1)}/{len(frames)}")

        _maybe_overlay_last_frame()
        fig.canvas.draw_idle()

    def on_save_click(event):
        # Only allow save on final frame (where overlays exist)
        if frame_idx["i"] != len(frames) - 1 or not overlays["patches"]:
            print("[Save] Not on final frame or no polygons.")
            return
        payload = _collect_selected_payload()
        saved_clicks.append(payload)
        if on_save is not None:
            try:
                on_save(payload)  # user callback
            except Exception as e:
                print(f"[Save] on_save callback error: {e}")
        print(f"[Save] Saved {len(payload)} polygon(s).")
        _update_status()

    def on_clear_click(event):
        # Clear current selection and reset styles (keep overlays)
        for p in overlays["patches"]:
            _apply_style(p, False)
        selected_ids.clear()
        print("[Clear] Selection cleared.")
        _update_status()
        fig.canvas.draw_idle()

    def on_next_click(_):         # NEW: close without saving
        saved_payload = None      # sentinel meaning "skipped/next"
        plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    btn_save.on_clicked(on_save_click)
    btn_clear.on_clicked(on_clear_click)
    btn_next.on_clicked(on_next_click)

    # initial overlay if we start at last frame (rare)
    _maybe_overlay_last_frame()
    _update_status()
    plt.show()

    # Return what was saved during the session + the last live selection when closed
    return {"saved_clicks": saved_clicks, "last_selection": _collect_selected_payload()}


scene_ids = set()
for fname in os.listdir("../data"):
    match = re.match(r"(\d+)_.*\.csv", fname)
    if match:
        scene_ids.add(match.group(1))

scene_ids = sorted(scene_ids)
print(f"Found scenes: {scene_ids}")

input_path = "../data"
output_path = "../data/annotations"

tracks = []
tracks_meta = []
visibility_data = []
background_images = {}
fixed_blocks_info = {}
frame_to_track_idxs = {}

start_scene = 5
end_scene = 6
filename = "index_map17.pkl"

data_dict_all = {}

# Check data file exists
index_file_path = Path(os.path.join(input_path, filename))
if index_file_path.exists():
    logger.info("Loading index map from {}", index_file_path)
    data_dict_all = pickle.load(open(index_file_path, "rb"))

    # Loading background images
    for scene_id in scene_ids:
        if int(scene_id) < start_scene or int(scene_id) >= end_scene:
            continue

        # Store background images for scenes
        bg_path = os.path.join(input_path, 'semantic_maps', f"{scene_id}_background.png")
        img = cv2.imread(bg_path)
        background_images[int(scene_id)] = img

keys = list(data_dict_all.keys())
print("Done Loading")

for idx in range(len(keys)):
    key = keys[idx]
    key_elements = key.split("_")
    scene_id = int(key_elements[0])
    current_frame = int(key_elements[1])
    ego_vehicle_track_idx = int(key_elements[2])

    skip_until_scene_id = 5
    if skip_until_scene_id != -1 and scene_id < skip_until_scene_id:
        continue

    skip_until_frame = -1
    if skip_until_frame != -1 and current_frame < skip_until_frame:
        continue

    data_dict = data_dict_all[key]

    # map
    backgrond_img = background_images[scene_id]

    # Historical observations
    historical_adjacent_obs, historical_ego_obs = data_dict["historical_adjacent_obs"], data_dict["historical_ego_obs"]
    hidden_ogm_cells = data_dict["hidden_ogm_cells"]
    hidden_tracks_pts = data_dict["hidden_tracks_pts"]
    hidden_tracks_id_ogm_data = data_dict["hidden_tracks_id_ogm"]
    omg_cells_coords = np.array(data_dict["cell_coords"])
    ogm_cell_label_data = data_dict["ogm_gt"].reshape(-1)
    ogm_cell_visibility_data = data_dict["ogm"].reshape(-1)

    num_timesteps = 21

    frames = []
    for t in range(num_timesteps):
        if t != num_timesteps - 1:
            frames.append(draw_frame(backgrond_img, historical_adjacent_obs.values(), historical_ego_obs, None,
                                     t, num_timesteps))
        if t == num_timesteps - 1:
            frames.append(draw_frame(backgrond_img, historical_adjacent_obs.values(), historical_ego_obs,
                                     hidden_tracks_pts, t, num_timesteps))

    def save_callback(payload):
        file_path = os.path.join(output_path, f"{key}.json")

        brief = [{"index": d["index"], "cx": d["centroid"][0], "cy": d["centroid"][1], "label": d["label"], "track_ids":  d["track_ids"]} for d in payload]
        with open(file_path, "w") as f:
            json.dump(brief, f, indent=2)

        print(brief)


    result = interactive_playback(
        frames,
        omg_cells_coords,
        ogm_cell_visibility_data,
        ogm_cell_label_data,
        backgrond_img.shape,
        on_save=save_callback,
        key=key,
        hidden_tracks_id_ogm_data=hidden_tracks_id_ogm_data,
    )

    # When the window is closed, you still get everything:
    print("All saves:", len(result["saved_clicks"]))
    print("Selection at close:", len(result["last_selection"]))


