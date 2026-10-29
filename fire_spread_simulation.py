"""
Forest Fire Spread Simulation using Cellular Automata (CA)
Outputs : snapshots at 1, 2, 3, 6, 12 hours  +  animated GIF

Libraries: numpy, scipy, matplotlib, pillow
Install  : pip install numpy scipy matplotlib pillow

Model (based on Alexandridis et al. 2008 style probabilistic CA):
    p_burn = p_base * f_fuel * f_moisture * f_wind * f_slope
    Each unburnt cell can be ignited by each of its 8 burning neighbours.

Grid cell states:
    0 = unburnt (burnable)   1 = burning   2 = burnt   3 = non-burnable (water / settlement)

NOTE: Yahan synthetic DEM / fuel / weather use hui hai. Asli data ke liye
      neeche load_real_data() wala section dekho.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------------
SEED = 7
rng = np.random.default_rng(SEED)

N = 200                    # grid size (N x N cells)
CELL = 30.0                # cell size in metres (30 m DEM)
DT_MIN = 10                # one CA step = 10 minutes
BURN_STEPS = 3             # a cell keeps burning for 3 steps (30 min)
TOTAL_HOURS = 12
SNAP_HOURS = [1, 2, 3, 6, 12]

# Weather (single values; real data me ERA5/IMD se grid-wise aayega)
WIND_SPEED = 6.0           # m/s
WIND_FROM_DEG = 225.0      # direction wind is blowing FROM (225 = South-West)
TEMP = 32.0                # deg C
RH = 30.0                  # % relative humidity
RAIN = 0.0                 # mm

P_BASE = 0.35              # base spread probability (calibration parameter)

# ------------------------------------------------------------------
# 2. INPUT LAYERS  (synthetic)  -> replace with real data
# ------------------------------------------------------------------
def make_synthetic_data(n):
    # DEM: smoothed random noise scaled to 300-900 m
    dem = gaussian_filter(rng.normal(size=(n, n)), sigma=18)
    dem = (dem - dem.min()) / (dem.max() - dem.min()) * 600 + 300

    # LULC -> fuel classes: 0 water/non-burnable, 1 grass, 2 shrub, 3 forest
    veg = gaussian_filter(rng.normal(size=(n, n)), sigma=10)
    veg = (veg - veg.min()) / (veg.max() - veg.min())
    fuel = np.digitize(veg, [0.12, 0.40, 0.70])      # 0,1,2,3
    return dem, fuel


def load_real_data():
    """
    Asli data use karne ke liye is function ko use karo (rasterio install karo):

        import rasterio
        with rasterio.open("dem_30m.tif") as src:
            dem = src.read(1).astype(float)
        with rasterio.open("lulc.tif") as src:
            lulc = src.read(1)
        # LULC classes ko fuel classes (0-3) me map karo:
        fuel = np.zeros_like(lulc)
        fuel[np.isin(lulc, [GRASS_CODES])] = 1
        fuel[np.isin(lulc, [SHRUB_CODES])] = 2
        fuel[np.isin(lulc, [FOREST_CODES])] = 3
        return dem, fuel
    Dono rasters ka shape/resolution same hona chahiye (resample karke).
    """
    raise NotImplementedError


dem, fuel = make_synthetic_data(N)

# Slope (degrees) and aspect (degrees) from DEM
dzdy, dzdx = np.gradient(dem, CELL)                  # dzdy: along rows
slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
aspect_deg = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360

# Fuel factor (relative flammability) for classes 0..3
FUEL_FACTOR = np.array([0.0, 1.2, 1.0, 0.8])
f_fuel = FUEL_FACTOR[fuel]

# Moisture factor from humidity / rain / temperature
f_moist = np.clip(1.3 - RH / 60.0, 0.1, 1.2) * np.exp(-RAIN) * (1 + (TEMP - 25) / 100)

# ------------------------------------------------------------------
# 3. CELLULAR AUTOMATA ENGINE
# ------------------------------------------------------------------
# 8 neighbour offsets (dy, dx): source cell = target cell + (dy, dx)
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Wind blows TOWARD this direction
wind_to = np.radians((WIND_FROM_DEG + 180) % 360)
wind_vec = np.array([np.sin(wind_to), np.cos(wind_to)])      # (east, north)

C1, C2 = 0.045, 0.131      # wind constants (Alexandridis et al.)
A_SLOPE = 0.078            # slope constant


def shift(a, dy, dx, fill=0):
    """out[i, j] = a[i+dy, j+dx]  (fill outside the grid)"""
    out = np.full_like(a, fill)
    ys, ye = max(0, -dy), a.shape[0] - max(0, dy)
    xs, xe = max(0, -dx), a.shape[1] - max(0, dx)
    out[ys:ye, xs:xe] = a[ys + dy:ye + dy, xs + dx:xe + dx]
    return out


# Pre-compute direction-dependent factors (they do not change in time)
dir_factor = {}
for dy, dx in NEIGHBOURS:
    dist = CELL * np.hypot(dy, dx)
    # direction of spread: source -> target  (east, north)
    d = np.array([-dx, dy], dtype=float)
    d /= np.linalg.norm(d)
    cos_t = float(d @ wind_vec)
    f_wind = np.exp(C1 * WIND_SPEED) * np.exp(WIND_SPEED * C2 * (cos_t - 1))

    rise = dem - shift(dem, dy, dx, fill=0)                # target - source
    slope_dir = np.degrees(np.arctan(rise / dist))         # +ve = uphill spread
    f_slope = np.exp(A_SLOPE * slope_dir)

    p = P_BASE * f_fuel * f_moist * f_wind * f_slope
    # diagonal neighbours are farther -> lower chance
    p = np.clip(p / (1.0 if dy == 0 or dx == 0 else 1.414), 0, 1)
    dir_factor[(dy, dx)] = p


def step(state, burn_t):
    burning = state == 1
    not_ignited = np.ones_like(f_fuel)
    for off in NEIGHBOURS:
        src_burning = shift(burning.astype(float), *off)
        not_ignited *= 1 - dir_factor[off] * src_burning
    p_ignite = 1 - not_ignited
    new_fire = (state == 0) & (rng.random(state.shape) < p_ignite)

    burn_t[burning] += 1
    finished = burning & (burn_t >= BURN_STEPS)
    state = state.copy()
    state[finished] = 2
    state[new_fire] = 1
    return state, burn_t


# ------------------------------------------------------------------
# 4. RUN SIMULATION
# ------------------------------------------------------------------
state = np.zeros((N, N), dtype=int)
state[fuel == 0] = 3                                   # non-burnable

# Ignition point: ek burnable cell jo SW me ho (wind NE ki taraf le jayegi)
cy, cx = int(N * 0.65), int(N * 0.35)
cand = np.argwhere((fuel > 0) & (np.abs(np.arange(N)[:, None] - cy) < 15)
                   & (np.abs(np.arange(N)[None, :] - cx) < 15))
iy, ix = cand[len(cand) // 2]
state[iy - 1:iy + 2, ix - 1:ix + 2][fuel[iy - 1:iy + 2, ix - 1:ix + 2] > 0] = 1
burn_t = np.zeros((N, N))

steps_per_hour = 60 // DT_MIN
total_steps = TOTAL_HOURS * steps_per_hour
snap_steps = {h * steps_per_hour: h for h in SNAP_HOURS}

frames = [state.copy()]
snapshots = {}
for t in range(1, total_steps + 1):
    state, burn_t = step(state, burn_t)
    frames.append(state.copy())
    if t in snap_steps:
        snapshots[snap_steps[t]] = state.copy()

# ------------------------------------------------------------------
# 5. STATISTICS
# ------------------------------------------------------------------
cell_ha = CELL * CELL / 10000.0
print("=" * 55)
print(f"Wind: {WIND_SPEED} m/s from {WIND_FROM_DEG:.0f} deg | Temp {TEMP} C | RH {RH}%")
print("=" * 55)
print(f"{'Hour':>5} | {'Burning (ha)':>12} | {'Burnt (ha)':>10} | {'Total affected (ha)':>20}")
for h in SNAP_HOURS:
    s = snapshots[h]
    b = (s == 1).sum() * cell_ha
    d = (s == 2).sum() * cell_ha
    print(f"{h:>5} | {b:>12.1f} | {d:>10.1f} | {b + d:>20.1f}")

# ------------------------------------------------------------------
# 6. PLOTS
# ------------------------------------------------------------------
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(dem, vert_exag=2, dx=CELL, dy=CELL)

# 6.1 Input layers
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
im = ax[0].imshow(dem, cmap="terrain"); ax[0].set_title("DEM (m)")
plt.colorbar(im, ax=ax[0], fraction=0.046)
im = ax[1].imshow(slope_deg, cmap="YlOrRd"); ax[1].set_title("Slope (deg)")
plt.colorbar(im, ax=ax[1], fraction=0.046)
im = ax[2].imshow(aspect_deg, cmap="hsv"); ax[2].set_title("Aspect (deg)")
plt.colorbar(im, ax=ax[2], fraction=0.046)
fuel_cmap = ListedColormap(["#4a90d9", "#e8d77a", "#9bb85b", "#2d6a3a"])
ax[3].imshow(fuel, cmap=fuel_cmap, vmin=0, vmax=3)
ax[3].set_title("Fuel (blue=none, yellow=grass,\nlight green=shrub, dark=forest)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("fire_01_input_layers.png", dpi=120)
plt.close()

# 6.2 Snapshots 1/2/3/6/12 h
STATE_COLORS = ListedColormap([(0, 0, 0, 0),            # unburnt -> transparent
                               (1.0, 0.25, 0.0, 0.95),  # burning
                               (0.15, 0.15, 0.15, 0.85),# burnt
                               (0.2, 0.5, 0.9, 0.6)])   # non-burnable

def draw(ax, s, title):
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    ax.imshow(fuel, cmap=fuel_cmap, vmin=0, vmax=3, alpha=0.35)
    ax.imshow(s, cmap=STATE_COLORS, vmin=0, vmax=3, interpolation="nearest")
    ax.plot(ix, iy, "y*", ms=9)
    # wind arrow
    ax.annotate("", xy=(N * 0.12 + wind_vec[0] * 22, N * 0.12 - wind_vec[1] * 22),
                xytext=(N * 0.12, N * 0.12),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax.text(N * 0.03, N * 0.22, "wind", color="white", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
for a, h in zip(ax, SNAP_HOURS):
    s = snapshots[h]
    area = ((s == 1) | (s == 2)).sum() * cell_ha
    draw(a, s, f"After {h} h  ({area:.0f} ha)")
plt.suptitle("Fire spread simulation  (red = burning, black = burnt, * = ignition)",
             fontsize=13)
plt.tight_layout()
plt.savefig("fire_02_snapshots.png", dpi=120)
plt.close()

# 6.3 Burned area vs time
hrs = np.arange(len(frames)) * DT_MIN / 60
area_t = [((f == 1) | (f == 2)).sum() * cell_ha for f in frames]
plt.figure(figsize=(7, 4))
plt.plot(hrs, area_t, color="firebrick", lw=2)
for h in SNAP_HOURS:
    plt.axvline(h, color="grey", ls=":", lw=0.8)
plt.xlabel("Time (hours)")
plt.ylabel("Affected area (ha)")
plt.title("Fire growth over time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fire_03_growth_curve.png", dpi=120)
plt.close()

# 6.4 Animation (GIF)
print("\nGIF ban rahi hai, thoda wait karo...")
fig, ax = plt.subplots(figsize=(6, 6))
anim_frames = list(range(0, len(frames), 2))             # every 20 minutes

def update(k):
    ax.clear()
    t = anim_frames[k]
    h = t * DT_MIN / 60
    draw(ax, frames[t], f"Fire spread - t = {h:.1f} h")

ani = FuncAnimation(fig, update, frames=len(anim_frames), interval=150)
ani.save("fire_spread_animation.gif", writer=PillowWriter(fps=8))
plt.close()

print("Done. Files saved:")
print("  fire_01_input_layers.png, fire_02_snapshots.png,")
print("  fire_03_growth_curve.png, fire_spread_animation.gif")
