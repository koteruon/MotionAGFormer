import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


# Define rotation and transformation functions
def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray):
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if isinstance(res, torch.Tensor):
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif isinstance(result, torch.Tensor):
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


# Define the 3D pose visualization function
def show3Dpose(vals, ax):
    ax.clear()
    ax.view_init(elev=15.0, azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect("auto")

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)

    for i, (x, y, z) in enumerate(vals):
        ax.text(x, y, z, str(i), color="black", fontsize=12, ha="right")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.draw()


# Load skeleton data from JSON file
with open("demo/output/bhc_left_01_20240808_00/annotation/bhc_left_01_00.json") as f:
    data = json.load(f)

skeletons = np.array(data["skeletons"])
num_frames = skeletons.shape[0]

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Create sliders for frame and rotation adjustments
axframe = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor="lightgoldenrodyellow")
axrot1 = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor="lightgoldenrodyellow")
axrot2 = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
axrot3 = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor="lightgoldenrodyellow")
axrot4 = plt.axes([0.2, 0.18, 0.65, 0.03], facecolor="lightgoldenrodyellow")

sframe = Slider(axframe, "Frame", 0, num_frames - 1, valinit=0, valfmt="%0.0f")
srot1 = Slider(axrot1, "Rot1", -1.0, 1.0, valinit=0.14)
srot2 = Slider(axrot2, "Rot2", -1.0, 1.0, valinit=-0.15)
srot3 = Slider(axrot3, "Rot3", -1.0, 1.0, valinit=-0.76)
srot4 = Slider(axrot4, "Rot4", -1.0, 1.0, valinit=0.62)


# Update function for sliders
def update(val):
    frame_idx = int(sframe.val)
    rot = [srot1.val, srot2.val, srot3.val, srot4.val]
    rot = np.array(rot, dtype="float64")
    pose = camera_to_world(skeletons[frame_idx], R=rot, t=0)
    show3Dpose(pose, ax)


# Set slider update function
sframe.on_changed(update)
srot1.on_changed(update)
srot2.on_changed(update)
srot3.on_changed(update)
srot4.on_changed(update)

# Initial plot
update(None)

plt.show()
