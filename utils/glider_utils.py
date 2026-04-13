import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"  # render inline as JS
from utils.Glider1D import Glider1D
import flyvis
from flyvis.datasets.rendering import BoxEye
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import flyvis.utils.hex_utils
from flyvis.analysis.visualization import plt_utils, plots
### UTILS:
def plot_glider(S1, S2, title, figsize, fs=10, fps=40, xresol=32, tickwidth=2):    
    t_ax = np.arange(S1.shape[2])/fps
    fig, axes = plt.subplots(1,2, figsize=figsize)
    axes[0].imshow(S1[:,20,:].T, origin="upper", aspect="auto", cmap="gray",
            extent=(0, xresol, t_ax[-1], 0))
    axes[1].imshow(S2[:,20,:].T, origin="upper", aspect="auto", cmap="gray",
                extent=(0, xresol, t_ax[-1], 0))

    for ax in axes:
        ax.set_xlabel("x (pix)", fontsize=fs)
        ax.xaxis.set_label_position("top") ; ax.xaxis.set_ticks_position("top")
        ax.set_ylabel("Time (s)", fontsize=fs)
        ax.set_yticks(ticks=np.arange(0, t_ax[-1]+.1, 0.5))
        ax.tick_params(axis='both', which='major', labelsize=fs-2, width=tickwidth)
    fig.suptitle(title, y=.9, fontsize=fs)
    fig.tight_layout()
    plt.show()
    return fig, axes

def make_pair(rule, seed, parity=+1, vel=1, T=48):
    pd = Glider1D(rule, parity=parity, vel=vel, T=T,  direction="pd", seed=seed)
    nd = Glider1D(rule, parity=parity, vel=vel, T=T, direction="nd", seed=seed)
    pd.evolve(); nd.evolve()
    return pd.to_S(), nd.to_S()

# -------- Helper: draw diamond grid around each center --------
def draw_box_cells(ax, centers_yx, k_size, img_shape, step=1,
                   color='red', lw=0.5, alpha=0.8):
    """
    Draw the BoxEye sampling 'cells' (diamond squares at 45°) around centers.
    centers_yx: array of shape (N, 2) as (y, x) pixel coords.
    """
    H, W = img_shape
    r = (2 ** 0.5) * k_size / 2.0
    angles = np.deg2rad([45, 135, 225, 315, 405])  # closed diamond
    cosA, sinA = np.cos(angles), np.sin(angles)

    # optional thinning to reduce overdraw
    for (y0, x0) in centers_yx[::step]:
        vx = x0 + r * cosA   # x first
        vy = y0 + r * sinA   # y second
        # Simple visibility check (optional)
        if ((vx >= -1) & (vx <= W) & (vy >= -1) & (vy <= H)).any():
            ax.plot(vx, vy, '-', linewidth=lw, alpha=alpha, color=color)
            
def glider_gif(glider, fps=40, extent=15, k_size=1,
               line_width=0.8, line_alpha=0.8, dot_size=2,
               save=False, dest_path="movies/"):

    glider.evolve()
    movie_cart = glider.to_flyvis()      # (1, T, H, W)
    _, T, H, W = movie_cart.shape

    receptors = BoxEye(extent=extent, kernel_size=k_size)
    centers = receptors.receptor_centers.clone()
    centers[:, 0] += H // 2
    centers[:, 1] += W // 2
    c = centers.cpu().numpy()

    fig, ax_eye = plt.subplots(1, 1, figsize=(4, 4))
    im_eye = ax_eye.imshow(movie_cart[0, 0], origin="lower", cmap="gray", alpha=0.8, vmin=0, 
        vmax=1)
    ax_eye.scatter(c[:, 1], c[:, 0], s=dot_size)
    draw_box_cells(ax_eye, c, k_size, img_shape=(H, W), step=1,
                   color="red", lw=line_width, alpha=line_alpha)
    ttl_eye = ax_eye.set_title("Fly eye (t = 0)")
    ax_eye.set_xticks([]); ax_eye.set_yticks([])

    def update(t):
        im_eye.set_data(movie_cart[0, t])
        ttl_eye.set_text(f"Fly eye (t = {t})")
        return (im_eye, ttl_eye)

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)

    # --- show in notebook ---
    html = HTML(anim.to_jshtml())
                    # prevent duplicate static figure
    display(html)                  # <-- actually display it
    plt.close(fig) 
    # --- optional save ---
    if save:
        import os
        os.makedirs(dest_path, exist_ok=True)
        anim.save(os.path.join(dest_path, "glider_flyeye.gif"),
                  writer="pillow", fps=fps)

    return anim


def plot_recorded_cells(num_hexals=721, y_range_tolerance=0):
    radius = flyvis.utils.hex_utils.get_hextent(num_hexals)

    # Create integer (u, v) and pixel (x, y) coordinates
    u, v = flyvis.utils.hex_utils.get_hex_coords(radius)
    x, y = flyvis.utils.hex_utils.hex_to_pixel(u, v)


    band_indices = np.where(np.abs(y) <= y_range_tolerance)[0]

    # --- Prepare for LEFT-TO-RIGHT coloring for the selected band ---
    u_band = u[band_indices]
    min_u = u_band.min()
    max_u = u_band.max()
    range_u = max_u - min_u

    fig, ax = plt_utils.init_plot(figsize=[3, 3], fontsize=10)
    ax.scatter(x, y, s=40, color='darkgray', label='All Hexals', alpha=0.8)

    for idx in band_indices:
        neuron_x, neuron_y, neuron_u_coord = x[idx], y[idx], u[idx]

        # Calculate color_val for left-to-right gradient
        if range_u > 0:
            color_val = 1 - (neuron_u_coord - min_u) / range_u
        else:
            color_val = 0.5

        color = plt.cm.viridis(color_val)
        ax.scatter(neuron_x, neuron_y, s=40, color=color, zorder=5)

    # ax.set_title('Hexagonal Grid with Horizontal Band and L-R Coloring')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.scatter([], [], s=20, color='purple', label='Middle Band') # Proxy for legend
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    return fig, band_indices, u, v


def select_bandindices(y_range_tolerance):
    radius = flyvis.utils.hex_utils.get_hextent(721)

    # Create integer (u, v) and pixel (x, y) coordinates
    u, v = flyvis.utils.hex_utils.get_hex_coords(radius)
    x, y = flyvis.utils.hex_utils.hex_to_pixel(u, v)

    # --- Horizontal band selection ---
    band_indices = np.where(np.abs(y) <= y_range_tolerance)[0]

    # --- Exclude edge neurons (outermost ring) ---
    # Compute each neuron's radial distance from center
    r = np.sqrt(x**2 + y**2)
    max_r = r.max()

    # Define how much of the outer edge to exclude (e.g., 5–10%)
    margin = 0.25 * max_r  # tune this if needed
    valid_indices = np.where(r <= (max_r - margin))[0]

    # Combine both constraints: horizontal band & central region only
    band_indices_center = np.intersect1d(band_indices, valid_indices)

    print(f"Using {len(band_indices_center)} neurons (excluded {len(band_indices) - len(band_indices_center)} edge neurons)")

    # --- Prepare for LEFT-TO-RIGHT coloring for the selected band ---
    u_band = u[band_indices_center]
    min_u = u_band.min()
    max_u = u_band.max()
    range_u = max_u - min_u

    # Visual sanity check (optional)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(x, y, s=10, color='lightgray', alpha=0.3)
    ax.scatter(x[band_indices], y[band_indices], s=25, color='gray', label='original band')
    ax.scatter(x[band_indices_center], y[band_indices_center], s=40, color='purple', label='center band')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False)
    ax.set_title('Selected neurons (edges excluded)')
    plt.show()

    return fig, band_indices_center, u, v, min_u, max_u