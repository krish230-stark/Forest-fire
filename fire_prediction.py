"""
PART (a): NEXT-DAY FIRE PREDICTION MAP
--------------------------------------
Input  : weather (temp, RH, wind speed/direction, rain), terrain (slope, aspect),
         fuel (from LULC), past fire locations
Target : will a fire occur in pixel (i, j) on the NEXT day?  (0/1)
Models : Logistic Regression, Random Forest, Gradient Boosting (+ optional LSTM)
Output : fire probability map for next day, risk classes, metrics, plots

Libraries: numpy, pandas, scipy, matplotlib, scikit-learn  (+ torch, optional)
Install  : pip install numpy pandas scipy matplotlib scikit-learn

DATA MODE
  USE_SYNTHETIC = True   -> demo data generate hota hai (bina download ke chalta hai)
  USE_SYNTHETIC = False  -> asli data (ERA5 + FIRMS + DEM + LULC) real_data.py se aata hai
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from scipy.ndimage import gaussian_filter, uniform_filter

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. SETTINGS
# ------------------------------------------------------------------
USE_SYNTHETIC = True        # False karne par real_data.py use hoga
USE_LSTM = True             # torch installed ho to LSTM bhi train hoga
SEED = 11
rng = np.random.default_rng(SEED)

N = 100                     # grid size (synthetic)
D = 150                     # number of days (synthetic)
CELL = 100.0                # cell size in metres (synthetic)
START = 4                   # first usable day (needs 4 days of history for LSTM)
TRAIN_FRAC = 0.70           # first 70% days = train, last 30% = test (time-based split)
NEG_PER_POS = 10            # training me negatives ko downsample karna


# ------------------------------------------------------------------
# 1. DATA  (synthetic demo OR real)
# ------------------------------------------------------------------
def smooth_noise(shape, sigma):
    z = gaussian_filter(rng.normal(size=shape), sigma)
    return z / z.std()


def ar1(n, phi, std):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, std * np.sqrt(1 - phi ** 2))
    return x


def make_synthetic_dataset():
    """Demo data. Fire ka rule hidden hai (weather + dryness + fuel + slope),
    isliye model kuch seekh sakta hai. Ye REAL data nahi hai."""
    dem = gaussian_filter(rng.normal(size=(N, N)), sigma=12)
    dem = (dem - dem.min()) / (dem.max() - dem.min()) * 1000 + 400
    veg = gaussian_filter(rng.normal(size=(N, N)), sigma=8)
    veg = (veg - veg.min()) / (veg.max() - veg.min())
    fuel = np.digitize(veg, [0.10, 0.40, 0.70])           # 0 none,1 grass,2 shrub,3 forest

    days = np.arange(D)
    seasonal = 22 + 9 * np.sin(np.pi * days / D)
    reg_t = ar1(D, 0.85, 2.5)
    reg_rh = ar1(D, 0.8, 8)
    reg_w = ar1(D, 0.7, 1.2)
    wdir_reg = (225 + np.cumsum(rng.normal(0, 20, D))) % 360
    rain_event = rng.random(D) < 0.16

    temp = np.zeros((D, N, N), np.float32)
    rh = np.zeros_like(temp)
    wind = np.zeros_like(temp)
    wdir = np.zeros_like(temp)
    rain = np.zeros_like(temp)
    lapse = -6.5 * (dem - dem.mean()) / 1000
    for d in range(D):
        temp[d] = seasonal[d] + reg_t[d] + lapse + 0.8 * smooth_noise((N, N), 10)
        rh[d] = np.clip(70 - 1.6 * (temp[d] - 20) + reg_rh[d]
                        + 3 * smooth_noise((N, N), 10), 10, 100)
        wind[d] = np.clip(3 + reg_w[d] + 0.5 * smooth_noise((N, N), 10), 0.3, None)
        wdir[d] = (wdir_reg[d] + 10 * smooth_noise((N, N), 15)) % 360
        if rain_event[d]:
            rain[d] = np.clip(rng.exponential(6) * (1 + 0.5 * smooth_noise((N, N), 10)),
                              0, None)

    # slope for fire rule
    gy, gx = np.gradient(dem, CELL)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    f_fuel = np.array([0.0, 1.2, 1.0, 0.8])[fuel]

    zt = (temp - temp.mean()) / temp.std()
    zr = (rh - rh.mean()) / rh.std()
    zw = (wind - wind.mean()) / wind.std()

    fire = np.zeros((D, N, N), np.uint8)
    dry = np.zeros((N, N))
    for d in range(D):
        dry = np.where(rain[d] > 1, 0, dry + 1)
        nb = uniform_filter(fire[d - 1].astype(float), size=5) if d > 0 else 0
        z = (-7.8 + 0.9 * zt[d] - 0.9 * zr[d] + 0.5 * zw[d]
             + 0.10 * np.minimum(dry, 15) + 1.5 * (f_fuel - 1) + 0.04 * slope
             + 6 * nb - 2.5 * (rain[d] > 1))
        p = 1 / (1 + np.exp(-z)) * (fuel > 0)
        fire[d] = rng.random((N, N)) < p

    dates = pd.date_range("2024-03-01", periods=D)
    return dict(dates=dates, temp=temp, rh=rh, wind=wind, wdir=wdir, rain=rain,
                dem=dem, fuel=fuel, fire=fire, cell=CELL)


if USE_SYNTHETIC:
    data = make_synthetic_dataset()
else:
    from real_data import load_real_dataset
    data = load_real_dataset()

temp, rh, wind, wdir, rain = (data[k] for k in ["temp", "rh", "wind", "wdir", "rain"])
dem, fuel, fire, CELL = data["dem"], data["fuel"], data["fire"], data["cell"]
dates = data["dates"]
D, N, _ = temp.shape
print("=" * 65)
print(f"Days: {D} | Grid: {N}x{N} | Cell: {CELL:.0f} m | "
      f"Fire pixel-days: {int(fire.sum())} ({fire.mean() * 100:.3f}%)")
print("=" * 65)

# ------------------------------------------------------------------
# 2. TERRAIN FEATURES (slope, aspect)
# ------------------------------------------------------------------
gy, gx = np.gradient(dem, CELL)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING  (features of day d  ->  label = fire on day d+1)
# ------------------------------------------------------------------
wd_rad = np.radians(wdir)
wind_u = -wind * np.sin(wd_rad)                    # east-ward component
wind_v = -wind * np.cos(wd_rad)                    # north-ward component

rain3 = rain.copy()
temp3 = temp.copy()
for lag in (1, 2):
    rain3[lag:] += rain[:-lag]
    temp3[lag:] += temp[:-lag]
temp3 = temp3 / 3.0

dry_days = np.zeros_like(rain)
cur = np.zeros((N, N), np.float32)
for d in range(D):
    cur = np.where(rain[d] > 1, 0, cur + 1)
    dry_days[d] = np.minimum(cur, 15)

fire_recent = fire.astype(np.float32).copy()
for lag in (1, 2):
    fire_recent[lag:] = np.maximum(fire_recent[lag:], fire[:-lag])
fire_nb = uniform_filter(fire_recent, size=(1, 5, 5))      # recent fires nearby
doy = np.broadcast_to(np.asarray(dates.dayofyear, np.float32)[:, None, None], temp.shape)

static = {
    "slope": slope, "aspect_sin": np.sin(np.radians(aspect)),
    "aspect_cos": np.cos(np.radians(aspect)), "elevation": dem,
    "fuel_grass": (fuel == 1), "fuel_shrub": (fuel == 2), "fuel_forest": (fuel == 3),
}
dynamic = {
    "temp": temp, "rh": rh, "wind_speed": wind, "wind_u": wind_u, "wind_v": wind_v,
    "rain": rain, "rain_3d": rain3, "temp_3d": temp3, "dry_days": dry_days,
    "fire_today": fire.astype(np.float32), "fire_nearby_3d": fire_nb, "day_of_year": doy,
}
feature_names = list(dynamic) + list(static)


def feature_matrix(day_idx):
    """Rows = (day, row, col) for each day in day_idx."""
    cols = []
    for name in feature_names:
        if name in dynamic:
            a = dynamic[name][day_idx]
        else:
            a = np.broadcast_to(static[name], (len(day_idx), N, N))
        cols.append(a.reshape(-1).astype(np.float32))
    return np.stack(cols, axis=1)


all_days = np.arange(START, D - 1)
split = START + int((len(all_days)) * TRAIN_FRAC)
train_days = np.arange(START, split)
test_days = np.arange(split, D - 1)

X_train_full = feature_matrix(train_days)
y_train_full = fire[train_days + 1].reshape(-1)
X_test = feature_matrix(test_days)
y_test = fire[test_days + 1].reshape(-1)

# Fire sirf burnable cells par ho sakti hai -> non-burnable cells hata do
burnable_tr = np.broadcast_to(fuel > 0, (len(train_days), N, N)).reshape(-1)
burnable_te = np.broadcast_to(fuel > 0, (len(test_days), N, N)).reshape(-1)
X_train_full, y_train_full = X_train_full[burnable_tr], y_train_full[burnable_tr]
X_test, y_test = X_test[burnable_te], y_test[burnable_te]

# Class imbalance: sab positives + random negatives
pos = np.where(y_train_full == 1)[0]
neg = np.where(y_train_full == 0)[0]
neg_keep = rng.choice(neg, size=min(len(neg), NEG_PER_POS * len(pos)), replace=False)
idx = rng.permutation(np.concatenate([pos, neg_keep]))
X_train, y_train = X_train_full[idx], y_train_full[idx]
neg_rate = len(neg_keep) / len(neg)              # negatives kept fraction (for calibration)

print(f"Train days: {len(train_days)} | Test days: {len(test_days)}")
print(f"Train samples (after balancing): {len(y_train)}  (fires = {len(pos)})")
print(f"Test samples: {len(y_test)}  (fires = {int(y_test.sum())})\n")


def calibrate(p):
    """Downsampling se probabilities badh jaati hain -> prior correction."""
    return p * neg_rate / (p * neg_rate + (1 - p))


# ------------------------------------------------------------------
# 4. MODELS
# ------------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12,
                                            min_samples_leaf=5, random_state=SEED,
                                            n_jobs=1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                                        max_iter=200, random_state=SEED),
}


def evaluate(name, p):
    """Fire-risk maps ke liye ranking metrics zyada meaningful hain."""
    roc = roc_auc_score(y_test, p)
    pr = average_precision_score(y_test, p)
    thr = np.quantile(p, 0.95)                       # top 5% riskiest pixels
    top = p >= thr
    capture = y_test[top].sum() / y_test.sum()
    prec = y_test[top].mean()
    return dict(Model=name, ROC_AUC=roc, PR_AUC=pr,
                **{"Fires captured in top 5% (%)": capture * 100,
                   "Precision in top 5% (%)": prec * 100})


results, test_probs = [], {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)[:, 1]
    test_probs[name] = p
    results.append(evaluate(name, p))

# ------------------------------------------------------------------
# 4b. OPTIONAL: LSTM  (past 5 days weather sequence + static terrain)
# ------------------------------------------------------------------
lstm_predict_fn = None
if USE_LSTM:
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        SEQ = 5
        dyn_names = ["temp", "rh", "wind_speed", "rain", "fire_nearby_3d"]
        dyn_arr = np.stack([dynamic[k] for k in dyn_names], axis=-1)    # (D,N,N,5)
        mu = dyn_arr.reshape(-1, len(dyn_names)).mean(0)
        sd = dyn_arr.reshape(-1, len(dyn_names)).std(0) + 1e-6
        dyn_arr = ((dyn_arr - mu) / sd).astype(np.float32)
        st_names = ["slope", "aspect_sin", "aspect_cos", "elevation",
                    "fuel_grass", "fuel_shrub", "fuel_forest"]
        st_arr = np.stack([static[k].astype(np.float32) for k in st_names], axis=-1)
        st_arr = (st_arr - st_arr.reshape(-1, len(st_names)).mean(0)) / \
                 (st_arr.reshape(-1, len(st_names)).std(0) + 1e-6)

        def make_seq(days, rows, cols):
            """days/rows/cols: 1-D index arrays of equal length."""
            seq = np.stack([dyn_arr[days - k, rows, cols] for k in range(SEQ - 1, -1, -1)],
                           axis=1)                                    # (n, SEQ, 5)
            return seq, st_arr[rows, cols]

        class FireLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(len(dyn_names), 32, batch_first=True)
                self.head = nn.Sequential(nn.Linear(32 + len(st_names), 32), nn.ReLU(),
                                          nn.Linear(32, 1))

            def forward(self, seq, stat):
                _, (h, _) = self.lstm(seq)
                return self.head(torch.cat([h[-1], stat], dim=1)).squeeze(1)

        # training samples (same balanced scheme)
        def sample_index(days, y_all_fn):
            dd, rr, cc = np.meshgrid(days, np.arange(N), np.arange(N), indexing="ij")
            dd, rr, cc = dd.reshape(-1), rr.reshape(-1), cc.reshape(-1)
            keep = fuel[rr, cc] > 0
            return dd[keep], rr[keep], cc[keep]

        dd, rr, cc = sample_index(train_days, None)
        yy = fire[dd + 1, rr, cc]
        p_i = np.where(yy == 1)[0]
        n_i = rng.choice(np.where(yy == 0)[0], size=NEG_PER_POS * len(p_i), replace=False)
        sel = rng.permutation(np.concatenate([p_i, n_i]))
        seq_tr, st_tr = make_seq(dd[sel], rr[sel], cc[sel])
        seq_tr, st_tr = torch.tensor(seq_tr), torch.tensor(st_tr)
        y_tr = torch.tensor(yy[sel].astype(np.float32))

        net = FireLSTM()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        lossf = nn.BCEWithLogitsLoss()
        for epoch in range(8):
            perm = torch.randperm(len(y_tr))
            tot = 0.0
            for b in range(0, len(perm), 1024):
                bi = perm[b:b + 1024]
                opt.zero_grad()
                loss = lossf(net(seq_tr[bi], st_tr[bi]), y_tr[bi])
                loss.backward()
                opt.step()
                tot += loss.item() * len(bi)
            print(f"LSTM epoch {epoch + 1}/8  loss = {tot / len(perm):.4f}")

        def lstm_predict(days, rows, cols):
            net.eval()
            out = []
            with torch.no_grad():
                for b in range(0, len(days), 20000):
                    s, t = make_seq(days[b:b + 20000], rows[b:b + 20000], cols[b:b + 20000])
                    out.append(torch.sigmoid(net(torch.tensor(s), torch.tensor(t))).numpy())
            return np.concatenate(out)

        dte, rte, cte = sample_index(test_days, None)
        p_lstm = lstm_predict(dte, rte, cte)
        test_probs["LSTM"] = p_lstm
        results.append(evaluate("LSTM", p_lstm))
        lstm_predict_fn = lstm_predict
        print()
    except ImportError:
        print("PyTorch installed nahi hai -> LSTM skip. (pip install torch)\n")

# ------------------------------------------------------------------
# 5. RESULTS
# ------------------------------------------------------------------
res = pd.DataFrame(results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
print("=" * 65)
print("MODEL COMPARISON (test = last 30% days, time-based split)")
print("=" * 65)
print(res.round(3).to_string(index=False))
base_rate = y_test.mean() * 100
print(f"\nBaseline fire rate in test: {base_rate:.3f}%  "
      f"-> random top-5% pixels capture ~5% of fires")
best_name = res.iloc[0]["Model"]
print(f"Best model (PR-AUC): {best_name}")

# ROC + PR curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in test_probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
    pr_, rc_, _ = precision_recall_curve(y_test, p)
    ax[1].plot(rc_, pr_, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
ax[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8)
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
for a in ax:
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pred_01_roc_pr_curves.png", dpi=120)
plt.close()

# Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(imp.index, imp.values, color="darkorange")
plt.title("Random Forest - feature importance")
plt.tight_layout()
plt.savefig("pred_02_feature_importance.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6. NEXT-DAY FIRE PREDICTION MAP
# ------------------------------------------------------------------
# Last day ke features -> agle din ka map. (Synthetic/historical data me hum
# actual agle din se compare bhi kar sakte hain; real forecast me last day = aaj)
map_day = D - 2
map_model_name = best_name if best_name in models else "Random Forest"
Xmap = feature_matrix(np.array([map_day]))
prob_map = calibrate(models[map_model_name].predict_proba(Xmap)[:, 1]).reshape(N, N)
prob_map = np.where(fuel > 0, prob_map, np.nan)
actual_next = fire[map_day + 1]

q = np.nanquantile(prob_map, [0.85, 0.95, 0.99])
risk = np.zeros((N, N), int)                     # 0 low, 1 moderate, 2 high, 3 very high
risk[prob_map >= q[0]] = 1
risk[prob_map >= q[1]] = 2
risk[prob_map >= q[2]] = 3
risk = np.where(fuel > 0, risk, -1)

hill = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=CELL, dy=CELL)
fig, ax = plt.subplots(1, 2, figsize=(14, 6.3))
for a in ax:
    a.imshow(hill, cmap="gray", vmin=0, vmax=1)
    a.axis("off")
im = ax[0].imshow(prob_map, cmap="YlOrRd", alpha=0.8)
plt.colorbar(im, ax=ax[0], fraction=0.046, label="Fire probability (next day)")
ax[0].set_title(f"Next-day fire probability\n({dates[map_day + 1].date()}, {map_model_name})")

cmap_r = ListedColormap([(0, 0, 0, 0), (0.55, 0.8, 0.4, 0.75), (1, 0.85, 0.1, 0.8),
                         (1, 0.5, 0.0, 0.85), (0.8, 0.0, 0.0, 0.9)])
ax[1].imshow(risk + 1, cmap=cmap_r, vmin=0, vmax=4, interpolation="nearest")
fy, fx = np.where(actual_next == 1)
ax[1].scatter(fx, fy, s=22, c="black", marker="x", label="Actual fire (next day)")
ax[1].legend(loc="lower right", fontsize=8)
ax[1].set_title("Risk classes  (green=low, yellow=moderate,\norange=high, red=very high)")
plt.tight_layout()
plt.savefig("pred_03_next_day_fire_map.png", dpi=120)
plt.close()

in_high = (risk[actual_next == 1] >= 2).mean() * 100 if actual_next.sum() else float("nan")
print(f"\nMap day: {dates[map_day].date()} -> predicted for {dates[map_day + 1].date()}")
print(f"Actual fires next day: {int(actual_next.sum())} | "
      f"fraction inside High/Very-high zones: {in_high:.1f}% "
      f"(those zones cover ~15% of area)")

# Save outputs
np.save("fire_probability_next_day.npy", prob_map)
flat = [(r, c, prob_map[r, c]) for r in range(N) for c in range(N) if not np.isnan(prob_map[r, c])]
top = pd.DataFrame(sorted(flat, key=lambda t: -t[2])[:10], columns=["row", "col", "probability"])
top.to_csv("top10_risk_pixels.csv", index=False)
print("\nTop 5 highest-risk pixels (spread simulation ke liye ignition candidates):")
print(top.head(5).round(4).to_string(index=False))

print("\nDone. Files saved:")
print("  pred_01_roc_pr_curves.png, pred_02_feature_importance.png,")
print("  pred_03_next_day_fire_map.png, fire_probability_next_day.npy, top10_risk_pixels.csv")
