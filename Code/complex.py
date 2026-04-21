import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.linalg import solve_banded
from scipy.ndimage import binary_dilation

# ============================================================
# 2D FitzHugh-Nagumo spiral-wave / direct-stimulus simulation
#
# Event system:
#   All external interventions are handled by apply_events(...).
#   To change impulse/reset behavior, edit EVENT_CONFIG only.
#
# Supported event types:
#   - "wall_stimulus"
#   - "reset_region"
#   - "defib_stimulus"
#   - "point_stimulus"
#   - "teleportation_stimulus"
#   - "domain_stimulus"
#
# Stimulus behavior:
#   - one-step only
#   - direct modification of u before evolution
#   - mode = "overwrite" or "add"
#
# Diagnostics:
#   - for every defib_stimulus, teleportation_stimulus, or
#     domain_stimulus event, store and plot a post-stimulus /
#     pre-evolution diagnostic using contours from the
#     pre-stimulus state.
#
# Animation:
#   - starts at animation_start_time
#   - no pause-frame behavior
# ============================================================

# ----------------------------
# PDE parameters
# ----------------------------
alpha = 0.1
beta = 0.3
gamma = 1.0
eps = 0.010
delta = 0.0
D = 0.0003

Lx = 6.0
Ly = 6.0
Nx = 200
Ny = 200
dt = 0.1

u_th = 0.3
v_th = 0.07
g_th = 0.04

# --------------------------------
# Simulation time
# --------------------------------
T_final = 3500.0
n_steps = int(T_final / dt)

# ----------------------------
# Event configuration
# ----------------------------
EVENT_CONFIG = {
    "events": [
        {
             "name": "normal_1",
             "type": "point_stimulus",
             "time": 0.0,
             "amplitude": 1.0,
             "center": (3.0, 3.0),
             "radius": 0.25,
             "mode": "overwrite",
        },
        {
             "name": "abnormal_1",
             "type": "point_stimulus",
             "time": 400.0,
             "amplitude": 1.0,
             "center": (4.0, 4.0),
             "radius": 0.25,
             "mode": "overwrite",
        },
        {
             "name": "abnormal_2",
             "type": "point_stimulus",
             "time": 900.0,
             "amplitude": 1.0,
             "center": (2.0, 4.0),
             "radius": 0.25,
             "mode": "overwrite",
        },
        {
             "name": "abnormal_3",
             "type": "point_stimulus",
             "time": 1200.0,
             "amplitude": 1.0,
             "center": (4.0, 2.25),
             "radius": 0.25,
             "mode": "overwrite",
        },
        {
             "name": "abnormal_4",
             "type": "point_stimulus",
             "time": 1400.0,
             "amplitude": 1.0,
             "center": (2.0, 2.0),
             "radius": 0.25,
             "mode": "overwrite",
        },
        {
             "name": "abnormal_5",
             "type": "point_stimulus",
             "time": 1700.0,
             "amplitude": 1.0,
             "center": (4.0, 1.5),
             "radius": 0.25,
             "mode": "overwrite",
        },
        
    ]
}

# ----------------------------
# Animation settings
# ----------------------------
frame_every = 20
animation_interval = 1
animation_start_time = 2700.0

# ----------------------------
# Plot style
# ----------------------------
contour_lw = 1.2

# -------------------------------------
# Grid and finite-difference parameters
# -------------------------------------
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

rx = D * dt / dx**2
ry = D * dt / dy**2

x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# -----------------------------------------
# Initial condition
# -----------------------------------------
u = np.zeros((Ny, Nx), dtype=np.float64)
v = np.zeros((Ny, Nx), dtype=np.float64)

# -----------------------------------------
# Precompute banded matrices for ADI solves
# -----------------------------------------
def build_implicit_coeffs(n, r):
    lower = np.full(n - 1, -0.5 * r, dtype=np.float64)
    diag = np.full(n, 1.0 + r, dtype=np.float64)
    upper = np.full(n - 1, -0.5 * r, dtype=np.float64)
    upper[0] = -r
    lower[-1] = -r
    return lower, diag, upper

ax_lower, ax_diag, ax_upper = build_implicit_coeffs(Nx, rx)
ay_lower, ay_diag, ay_upper = build_implicit_coeffs(Ny, ry)

abx = np.zeros((3, Nx), dtype=np.float64)
abx[0, 1:] = ax_upper
abx[1, :] = ax_diag
abx[2, :-1] = ax_lower

aby = np.zeros((3, Ny), dtype=np.float64)
aby[0, 1:] = ay_upper
aby[1, :] = ay_diag
aby[2, :-1] = ay_lower

# -----------------------------------------
# Snapshot times
# -----------------------------------------
snapshot_times = [1750.0, 2000.0, 2500.0, 3000.0]
snapshot_indices = {int(round(ts / dt)): ts for ts in snapshot_times}
snapshots_u = {}
snapshots_v = {}

# -----------------------------------------
# Animation storage
# -----------------------------------------
animation_u_frames = []
animation_times = []

# -----------------------------------------
# Diagnostic event storage
# -----------------------------------------
diagnostic_events = []

# -----------------------------------------
# Preallocated work arrays
# -----------------------------------------
lap_y = np.empty_like(u)
lap_x = np.empty_like(u)
F = np.empty_like(u)
u_half = np.empty_like(u)
rhs_half = np.empty_like(u)
rhs_new = np.empty_like(u)
v_new = np.empty_like(v)

# -----------------------------------------
# Helpers
# -----------------------------------------
def build_disk_structure_grid(radius_cells):
    yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells

def build_wall_mask(rows):
    mask = np.zeros((Ny, Nx), dtype=bool)
    mask[:rows, :] = True
    return mask

def build_point_mask(center_xy, radius_phys):
    cx, cy = center_xy
    return (X - cx) ** 2 + (Y - cy) ** 2 <= radius_phys ** 2

def compute_defib_base_mask(u_arr, v_arr):
    return (u_arr < u_th) & (np.abs(v_arr - v_th) < g_th)

def dilate_mask(mask, radius_cells):
    if radius_cells <= 0:
        return mask.copy()
    structure = build_disk_structure_grid(radius_cells)
    return binary_dilation(mask, structure=structure)

def compute_defib_mask(u_arr, v_arr, radius_cells):
    base_mask = compute_defib_base_mask(u_arr, v_arr)
    return dilate_mask(base_mask, radius_cells)

def compute_teleportation_mask(u_arr, v_arr, radius_cells, center_xy, cutoff_radius):
    base_mask = compute_defib_mask(u_arr, v_arr, radius_cells)
    radial_cut = build_point_mask(center_xy, cutoff_radius)
    return base_mask & radial_cut

def apply_stimulus_to_u(u_arr, mask, amplitude, mode="overwrite"):
    if mode == "overwrite":
        u_arr[mask] = amplitude
    elif mode == "add":
        u_arr[mask] += amplitude
    else:
        raise ValueError(f"Unknown stimulus mode: {mode}")

def laplacian_y_explicit_inplace(u_arr, out):
    out[1:-1, :] = (u_arr[2:, :] - 2.0 * u_arr[1:-1, :] + u_arr[:-2, :]) / dy**2
    out[0, :] = 2.0 * (u_arr[1, :] - u_arr[0, :]) / dy**2
    out[-1, :] = 2.0 * (u_arr[-2, :] - u_arr[-1, :]) / dy**2

def laplacian_x_explicit_inplace(u_arr, out):
    out[:, 1:-1] = (u_arr[:, 2:] - 2.0 * u_arr[:, 1:-1] + u_arr[:, :-2]) / dx**2
    out[:, 0] = 2.0 * (u_arr[:, 1] - u_arr[:, 0]) / dx**2
    out[:, -1] = 2.0 * (u_arr[:, -2] - u_arr[:, -1]) / dx**2

def step_adi(u_arr, v_arr):
    global F, lap_y, lap_x, rhs_half, rhs_new, u_half, v_new

    # Reaction only; events already directly modified u
    np.multiply(u_arr, (u_arr - alpha), out=F)
    F *= (u_arr - 1.0)
    F += v_arr
    F *= -1.0

    np.multiply(beta, u_arr, out=v_new)
    v_new -= gamma * v_arr
    v_new -= delta
    v_new *= eps * dt
    v_new += v_arr

    laplacian_y_explicit_inplace(u_arr, lap_y)
    np.multiply(0.5 * dt * D, lap_y, out=rhs_half)
    rhs_half += u_arr
    rhs_half += 0.5 * dt * F

    u_half[:, :] = solve_banded((1, 1), abx, rhs_half.T).T

    laplacian_x_explicit_inplace(u_half, lap_x)
    np.multiply(0.5 * dt * D, lap_x, out=rhs_new)
    rhs_new += u_half

    u_new = solve_banded((1, 1), aby, rhs_new)
    return u_new, v_new.copy()

def add_levelset_contours(ax, u_arr, v_arr):
    f = u_arr - u_th
    g = v_arr - v_th

    ef = np.ma.masked_where(g >= 0.0, f)
    ax.contour(x, y, ef, levels=[0.0], colors='k', linewidths=contour_lw, linestyles='solid')

    eb = np.ma.masked_where(g <= 0.0, f)
    ax.contour(x, y, eb, levels=[0.0], colors='k', linewidths=contour_lw, linestyles='dashed')

    rf = np.ma.masked_where(f <= 0.0, g)
    ax.contour(x, y, rf, levels=[0.0], colors='w', linewidths=contour_lw, linestyles='solid')

    rb = np.ma.masked_where(f >= 0.0, g)
    ax.contour(x, y, rb, levels=[0.0], colors='w', linewidths=contour_lw, linestyles='dashed')

# -----------------------------------------
# Event system
# -----------------------------------------
def event_step_index(event_time):
    return int(round(event_time / dt))

def compute_event_mask(event, u_arr, v_arr):
    etype = event["type"]

    if etype == "wall_stimulus":
        return build_wall_mask(event["rows"])

    if etype == "defib_stimulus":
        return compute_defib_mask(u_arr, v_arr, event.get("radius", 0))

    if etype == "point_stimulus":
        return build_point_mask(event["center"], event["radius"])

    if etype == "teleportation_stimulus":
        return compute_teleportation_mask(
            u_arr,
            v_arr,
            event.get("radius", 0),
            event["center"],
            event["cutoff_radius"],
        )

    if etype == "domain_stimulus":
        return np.ones((Ny, Nx), dtype=bool)

    return None

def apply_events(u_arr, v_arr, n, event_config):
    """
    Applies all events scheduled for step n directly to the state.

    Returns a list of diagnostics for any defib/teleportation/domain events.
    """
    diagnostics = []

    for event in event_config["events"]:
        if n != event_step_index(event["time"]):
            continue

        etype = event["type"]

        if etype in (
            "wall_stimulus",
            "defib_stimulus",
            "point_stimulus",
            "teleportation_stimulus",
            "domain_stimulus",
        ):
            pre_u = u_arr.copy()
            pre_v = v_arr.copy()

            mask = compute_event_mask(event, u_arr, v_arr)
            apply_stimulus_to_u(
                u_arr,
                mask,
                event["amplitude"],
                event.get("mode", "overwrite"),
            )

            if etype in ("defib_stimulus", "teleportation_stimulus", "domain_stimulus"):
                diagnostics.append({
                    "name": event["name"],
                    "type": etype,
                    "time": event["time"],
                    "pre_u": pre_u,
                    "pre_v": pre_v,
                    "post_u": u_arr.copy(),
                    "mask": mask.copy(),
                })

            if etype == "defib_stimulus":
                percent_domain = 100.0 * np.count_nonzero(mask) / mask.size
                print(
                    f"[defib_stimulus] {event['name']} at t={event['time']:.1f}: "
                    f"{percent_domain:.3f}% of full domain stimulated"
                )

        elif etype == "reset_region":
            x_stop = Nx // 2 - event.get("x_stop_offset", 0)
            u_arr[:, :x_stop] = event["u_value"]

        else:
            raise ValueError(f"Unknown event type: {etype}")

    return diagnostics

# -----------------------------------------
# Main loop
# -----------------------------------------
for n in range(n_steps + 1):
    t = n * dt

    new_diags = apply_events(u, v, n, EVENT_CONFIG)
    diagnostic_events.extend(new_diags)

    if n in snapshot_indices:
        ts = snapshot_indices[n]
        snapshots_u[ts] = u.copy()
        snapshots_v[ts] = v.copy()

    if n % frame_every == 0:
        animation_u_frames.append(u.copy())
        animation_times.append(t)

    if n == n_steps:
        break

    u, v = step_adi(u, v)

# -----------------------------------------
# Build animation frames starting later
# -----------------------------------------
display_u_frames = []
display_times = []

for u_frame, t in zip(animation_u_frames, animation_times):
    if t < animation_start_time:
        continue
    display_u_frames.append(u_frame)
    display_times.append(t)

if len(display_u_frames) == 0:
    raise ValueError(
        f"No animation frames remain after applying animation_start_time={animation_start_time}. "
        f"Choose an earlier start time or smaller frame_every."
    )

# -----------------------------------------
# Shared color scale for u plots
# -----------------------------------------
all_snapshot_vals = np.concatenate([snapshots_u[t].ravel() for t in snapshot_times if t in snapshots_u])
all_anim_vals = np.concatenate([frame.ravel() for frame in display_u_frames])

vmin = min(np.min(all_snapshot_vals), np.min(all_anim_vals))
vmax = 1.0
red_mask_cmap = ListedColormap(["red"])

# -----------------------------------------
# Animation (NO contours)
# -----------------------------------------
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))

im_anim = ax_anim.imshow(
    display_u_frames[0],
    origin="lower",
    extent=[0, Lx, 0, Ly],
    aspect="equal",
    vmin=vmin,
    vmax=vmax,
    cmap="jet"
)

cbar = plt.colorbar(im_anim, ax=ax_anim, fraction=0.046, pad=0.04)
cbar.set_label("u")

title = ax_anim.set_title(f"t = {display_times[0]:.1f}")
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")

def update(frame_idx):
    im_anim.set_data(display_u_frames[frame_idx])
    title.set_text(f"t = {display_times[frame_idx]:.1f}")
    return [im_anim, title]

ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(display_u_frames),
    interval=animation_interval,
    blit=False
)

plt.show()

# Optional save lines:
# ani.save("spiral_animation.gif", writer=PillowWriter(fps=20))
# ani.save("spiral_animation.mp4", writer=FFMpegWriter(fps=20))

# -----------------------------------------
# Still frames at end (WITH contours + legend)
# -----------------------------------------
ncols = 4
nrows = int(np.ceil(len(snapshot_times) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3), constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

for ax, ts in zip(axes, snapshot_times):
    im = ax.imshow(
        snapshots_u[ts],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="jet"
    )

    add_levelset_contours(ax, snapshots_u[ts], snapshots_v[ts])

    ax.set_title(f"t = {ts:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[len(snapshot_times):]:
    ax.axis("off")

plt.show()

# -----------------------------------------
# Diagnostic plots for every defib/teleportation/domain event
# -----------------------------------------
if diagnostic_events:
    for diag in diagnostic_events:
        fig_diag, ax_diag = plt.subplots(figsize=(7, 6), constrained_layout=True)

        im_diag = ax_diag.imshow(
            diag["post_u"],
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap="jet"
        )

        ax_diag.imshow(
            np.ma.masked_where(~diag["mask"], diag["mask"].astype(float)),
            origin="lower",
            extent=[0, Lx, 0, Ly],
            aspect="equal",
            cmap=red_mask_cmap,
            alpha=0.35,
            interpolation="nearest",
            zorder=5
        )

        add_levelset_contours(ax_diag, diag["pre_u"], diag["pre_v"])

        plt.colorbar(im_diag, ax=ax_diag, fraction=0.046, pad=0.04, label="u")
        ax_diag.set_xlabel("x")
        ax_diag.set_ylabel("y")


        plt.show()
else:
    print("No defibrillation, teleportation, or domain-stimulus diagnostics were captured.")



