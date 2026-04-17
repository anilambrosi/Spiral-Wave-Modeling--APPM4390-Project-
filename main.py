import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.linalg import solve_banded
from scipy.ndimage import binary_dilation

# ============================================================
# Optimized 2D FitzHugh-Nagumo spiral-wave / defibrillation simulation
#
# Key behavior in this version:
#   - Stimuli are NOT PDE source terms
#   - A stimulus is applied for ONE timestep only
#   - A stimulus directly modifies u before evolution
#   - Current mode: overwrite stimulated cells with the amplitude
#   - Easy to change later to additive application
#
# Outputs:
#   - animation (no contours)
#   - still frames (with contours + legend)
#   - one separate diagnostic plot showing the state immediately
#     after the defib stimulus is applied, before any evolution
# ============================================================

# ----------------------------
# Parameters
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
# Simulation time and interventions
# --------------------------------
T_final = 2000.0
n_steps = int(T_final / dt)

t_break = 450.0
t_defib = 1500.0

# Initial wall stimulus: one-step direct application
wall_time = 0.0
wall_amplitude = 1.0
wall_rows = 8

# Defibrillation stimulus: one-step direct application
defib_time = t_defib
defib_amplitude = 1.0

# Additional spatial thickness for defibrillation mask
# measured in grid cells
defib_radius = 0

# How to apply a direct stimulus to u:
#   "overwrite" -> u[mask] = amplitude
#   "add"       -> u[mask] += amplitude
STIMULUS_APPLICATION_MODE = "overwrite"

# ----------------------------
# Animation settings
# ----------------------------
frame_every = 20
animation_interval = 1
pause_repeats = 0

# Contour style for stills / diagnostic
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

# Step indices for one-shot stimuli
wall_step = int(round(wall_time / dt))
defib_step = int(round(defib_time / dt))

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
# Fixed wall stimulus mask
# -----------------------------------------
wall_mask = np.zeros((Ny, Nx), dtype=bool)
wall_mask[:wall_rows, :] = True

# -----------------------------------------
# Defibrillation dilation structure
# -----------------------------------------
def build_disk_structure(radius):
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius

defib_structure = build_disk_structure(defib_radius)

# -----------------------------------------
# Snapshot times
# -----------------------------------------
snapshot_times = [200.0, 400.0, 451.0, 500.0, 750.0, 1000.0, 1500.0, 1550.0, 1650.0]
snapshot_indices = {int(round(ts / dt)): ts for ts in snapshot_times}
snapshots_u = {}
snapshots_v = {}

# -----------------------------------------
# Diagnostic direct-stimulus frame at t_defib
# -----------------------------------------
diagnostic_pre_u = None
diagnostic_pre_v = None
diagnostic_post_u = None
diagnostic_mask = None
diagnostic_time = t_defib

# -----------------------------------------
# Animation storage
# -----------------------------------------
animation_u_frames = []
animation_times = []

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
def laplacian_y_explicit_inplace(u_arr, out):
    out[1:-1, :] = (u_arr[2:, :] - 2.0 * u_arr[1:-1, :] + u_arr[:-2, :]) / dy**2
    out[0, :] = 2.0 * (u_arr[1, :] - u_arr[0, :]) / dy**2
    out[-1, :] = 2.0 * (u_arr[-2, :] - u_arr[-1, :]) / dy**2

def laplacian_x_explicit_inplace(u_arr, out):
    out[:, 1:-1] = (u_arr[:, 2:] - 2.0 * u_arr[:, 1:-1] + u_arr[:, :-2]) / dx**2
    out[:, 0] = 2.0 * (u_arr[:, 1] - u_arr[:, 0]) / dx**2
    out[:, -1] = 2.0 * (u_arr[:, -2] - u_arr[:, -1]) / dx**2

def compute_defib_mask(u_arr, v_arr):
    base_mask = (u_arr < u_th) & (np.abs(v_arr - v_th) < g_th)
    thick_mask = binary_dilation(base_mask, structure=defib_structure)
    return thick_mask

def apply_stimulus_to_u(u_arr, mask, amplitude, mode="overwrite"):
    if mode == "overwrite":
        u_arr[mask] = amplitude
    elif mode == "add":
        u_arr[mask] += amplitude
    else:
        raise ValueError(f"Unknown STIMULUS_APPLICATION_MODE: {mode}")

def step_adi(u_arr, v_arr):
    global F, lap_y, lap_x, rhs_half, rhs_new, u_half, v_new

    # Reaction only; stimulus is applied directly to u beforehand
    np.multiply(u_arr, (u_arr - alpha), out=F)
    F *= (u_arr - 1.0)
    F += v_arr
    F *= -1.0

    # v update
    np.multiply(beta, u_arr, out=v_new)
    v_new -= gamma * v_arr
    v_new -= delta
    v_new *= eps * dt
    v_new += v_arr

    # First half-step: x-implicit, y-explicit
    laplacian_y_explicit_inplace(u_arr, lap_y)
    np.multiply(0.5 * dt * D, lap_y, out=rhs_half)
    rhs_half += u_arr
    rhs_half += 0.5 * dt * F

    u_half[:, :] = solve_banded((1, 1), abx, rhs_half.T).T

    # Second half-step: y-implicit, x-explicit
    laplacian_x_explicit_inplace(u_half, lap_x)
    np.multiply(0.5 * dt * D, lap_x, out=rhs_new)
    rhs_new += u_half

    u_new = solve_banded((1, 1), aby, rhs_new)

    return u_new, v_new.copy()

def add_levelset_contours(ax, u_arr, v_arr):
    """
    Plot the four contour types:
      excited front:    f = 0, g < 0   solid black
      excited back:     f = 0, g > 0   dashed black
      refractory front: g = 0, f > 0   solid white
      refractory back:  g = 0, f < 0   dashed white
    """
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
# Main loop
# -----------------------------------------
break_applied = False

for n in range(n_steps + 1):
    t = n * dt

    # One-shot wall stimulus at the chosen step
    if n == wall_step:
        apply_stimulus_to_u(u, wall_mask, wall_amplitude, STIMULUS_APPLICATION_MODE)

    # One-shot defib stimulus at the chosen step
    if n == defib_step:
        # Save contours from BEFORE stimulus
        diagnostic_pre_u = u.copy()
        diagnostic_pre_v = v.copy()

        defib_mask = compute_defib_mask(u, v)
        apply_stimulus_to_u(u, defib_mask, defib_amplitude, STIMULUS_APPLICATION_MODE)

        # Save the frame immediately AFTER stimulus, BEFORE evolution
        diagnostic_post_u = u.copy()
        diagnostic_mask = defib_mask.copy()

    # Save exact state at this simulation time (after any instantaneous stimulus)
    if n in snapshot_indices:
        ts = snapshot_indices[n]
        snapshots_u[ts] = u.copy()
        snapshots_v[ts] = v.copy()

    if n % frame_every == 0:
        animation_u_frames.append(u.copy())
        animation_times.append(t)

    if n == n_steps:
        break

    # One-time symmetry break before evolution
    if (not break_applied) and (t >= t_break):
        u[:, :Nx // 2 - 10] = 0.0
        break_applied = True

    # Evolve to next step
    u, v = step_adi(u, v)

# -----------------------------------------
# Build animation frames with pauses
# -----------------------------------------
display_u_frames = []
display_times = []

mask_pause_pending = False
defib_pause_pending = False
mask_pause_done = False
defib_pause_done = False

for u_frame, t in zip(animation_u_frames, animation_times):
    display_u_frames.append(u_frame)
    display_times.append(t)

    if (not mask_pause_done) and (not mask_pause_pending) and (t >= t_break):
        mask_pause_pending = True
        continue

    if (not defib_pause_done) and (not defib_pause_pending) and (t >= defib_time):
        defib_pause_pending = True
        continue

    if mask_pause_pending and (not mask_pause_done):
        for _ in range(pause_repeats):
            display_u_frames.append(u_frame)
            display_times.append(t)
        mask_pause_pending = False
        mask_pause_done = True

    if defib_pause_pending and (not defib_pause_done):
        for _ in range(pause_repeats):
            display_u_frames.append(u_frame)
            display_times.append(t)
        defib_pause_pending = False
        defib_pause_done = True

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

title = ax_anim.set_title(f"u at t = {display_times[0]:.1f}")
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")

def update(frame_idx):
    im_anim.set_data(display_u_frames[frame_idx])
    title.set_text(f"u at t = {display_times[frame_idx]:.1f}")
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
fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
axes = axes.ravel()

legend_handles = [
    Line2D([0], [0], color='k', lw=contour_lw, linestyle='solid', label='Excited front'),
    Line2D([0], [0], color='k', lw=contour_lw, linestyle='dashed', label='Excited back'),
    Line2D([0], [0], color='w', lw=contour_lw, linestyle='solid', label='Refractory front'),
    Line2D([0], [0], color='w', lw=contour_lw, linestyle='dashed', label='Refractory back'),
]

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

    ax.set_title(f"u at t = {ts:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[len(snapshot_times):]:
    ax.axis("off")

axes[0].legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=8,
    framealpha=0.9
)

plt.suptitle("2D FHN: upward wave, symmetry break, direct one-step stimulation", y=1.02)
plt.show()

# -----------------------------------------
# Separate diagnostic plot:
# FULL FRAME immediately after direct defib stimulus, before evolution
# but contours taken from BEFORE stimulus
# -----------------------------------------
if diagnostic_post_u is not None:
    fig_diag, ax_diag = plt.subplots(figsize=(7, 6), constrained_layout=True)

    im_diag = ax_diag.imshow(
        diagnostic_post_u,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="jet"
    )

    ax_diag.imshow(
        np.ma.masked_where(~diagnostic_mask, diagnostic_mask.astype(float)),
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        cmap=red_mask_cmap,
        alpha=0.35,
        interpolation="nearest",
        zorder=5
    )

    # Contours from BEFORE stimulus
    add_levelset_contours(ax_diag, diagnostic_pre_u, diagnostic_pre_v)

    plt.colorbar(im_diag, ax=ax_diag, fraction=0.046, pad=0.04, label="u")
    ax_diag.set_title(
        f"Direct-stimulus frame at t = {diagnostic_time:.1f}\n"
        f"(after one-step stimulus, before evolution;\ncontours from pre-stimulus state)"
    )
    ax_diag.set_xlabel("x")
    ax_diag.set_ylabel("y")

    diag_legend = [
        Line2D([0], [0], color='k', lw=contour_lw, linestyle='solid', label='Excited front (pre-stim)'),
        Line2D([0], [0], color='k', lw=contour_lw, linestyle='dashed', label='Excited back (pre-stim)'),
        Line2D([0], [0], color='w', lw=contour_lw, linestyle='solid', label='Refractory front (pre-stim)'),
        Line2D([0], [0], color='w', lw=contour_lw, linestyle='dashed', label='Refractory back (pre-stim)'),
        Line2D([0], [0], color='red', lw=6, alpha=0.7, label='Applied stimulus region'),
    ]
    ax_diag.legend(handles=diag_legend, loc="upper right", fontsize=8, framealpha=0.9)

    plt.show()
else:
    print("Diagnostic defib frame was not captured. Check dt and t_defib alignment.")