#metric computation and shows statistical comparisons and violin plots 

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, levene, ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

#edit folder paths based on local setup

PRO_FOLDER = 'pro_runners'
HS_FOLDER  = 'pose_results'

PRO_MEN_FOLDER   = '/Users/rishi/pro_men'
PRO_WOMEN_FOLDER = '/Users/rishi/pro_women'
HS_MEN_FOLDER    = '/Users/rishi/hs_men'
HS_WOMEN_FOLDER  = '/Users/rishi/hs_women'

SHOW_STATS      = True
P_ADJUST_METHOD = 'fdr_bh'   
ALPHA           = 0.05


def process_folder_metric(folder_path, per_file_func, verbose=False):
    """Apply per_file_func to each CSV in folder_path; return list of values."""
    if not os.path.isdir(folder_path):
        print(f"[WARN] Folder not found: {folder_path}")
        return []

    vals = []
    files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(".csv")]
    if verbose:
        print(f"[INFO] {folder_path}: {len(files)} csv(s)")

    for f in files:
        fp = os.path.join(folder_path, f)
        try:
            v = per_file_func(fp)
        except Exception as e:
            if verbose:
                print(f"[SKIP] {f}: error: {e}")
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if verbose:
                print(f"[SKIP] {f}: no usable value")
            continue
        vals.append(float(v))

    if verbose:
        print(f"[INFO] usable videos: {len(vals)}")
    return vals


def cohens_d_independent(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan

    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2.0)
    if pooled == 0:
        return np.nan
    return (m1 - m2) / pooled


def interpret_d(d):
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:  return "Very small diff"
    if ad < 0.5:  return "Small diff"
    if ad < 0.8:  return "Moderate/noticable diff"
    if ad < 1.2:  return "Big difference"
    return "Very very strong difference"


def print_stats(group1, group2, label1, label2):
    g1 = np.array([x for x in group1 if not pd.isna(x)], dtype=float)
    g2 = np.array([x for x in group2 if not pd.isna(x)], dtype=float)
    n1, n2 = len(g1), len(g2)
    mean1 = float(np.mean(g1)) if n1 else np.nan
    mean2 = float(np.mean(g2)) if n2 else np.nan

    if n1 >= 2 and n2 >= 2:
        t_stat, p_value = ttest_ind(g1, g2, equal_var=False)
        _, p_levene = levene(g1, g2)
        _, ks_p = ks_2samp(g1, g2)
    else:
        t_stat = p_value = p_levene = ks_p = np.nan

    d = cohens_d_independent(g1, g2)

    print(f"Samples: {label1}=n{n1}, {label2}=n{n2}")
    print(f"Means:   {label1}={mean1:.6f}, {label2}={mean2:.6f}")
    print(f"Welch t-statistic: {t_stat:.6f}")
    print(f"P-value: {p_value:.6g}")
    print(f"Cohen's D: {d:.6f} ({interpret_d(d)})")
    print(f"Levene's test p-value: {p_levene:.6g}")
    print(f"KS test p-value: {ks_p:.6g}")


def _angle_3d(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosv = np.dot(ba, bc) / denom
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def hip_flexion_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = [
        'left_shoulder_x','left_shoulder_y','left_shoulder_z',
        'left_hip_x','left_hip_y','left_hip_z',
        'left_knee_x','left_knee_y','left_knee_z',
        'right_shoulder_x','right_shoulder_y','right_shoulder_z',
        'right_hip_x','right_hip_y','right_hip_z',
        'right_knee_x','right_knee_y','right_knee_z',
    ]
    if any(c not in df.columns for c in req):
        return None

    df[req] = df[req].apply(pd.to_numeric, errors='coerce')
    mean_val, count = 0.0, 0

    for _, row in df.iterrows():
        if row[['left_shoulder_x','left_shoulder_y','left_shoulder_z',
                'left_hip_x','left_hip_y','left_hip_z',
                'left_knee_x','left_knee_y','left_knee_z']].isnull().any():
            la = np.nan
        else:
            la = _angle_3d(
                (row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']),
                (row['left_hip_x'],      row['left_hip_y'],      row['left_hip_z']),
                (row['left_knee_x'],     row['left_knee_y'],     row['left_knee_z'])
            )

        if row[['right_shoulder_x','right_shoulder_y','right_shoulder_z',
                'right_hip_x','right_hip_y','right_hip_z',
                'right_knee_x','right_knee_y','right_knee_z']].isnull().any():
            ra = np.nan
        else:
            ra = _angle_3d(
                (row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']),
                (row['right_hip_x'],      row['right_hip_y'],      row['right_hip_z']),
                (row['right_knee_x'],     row['right_knee_y'],     row['right_knee_z'])
            )

        if np.isnan(la) and np.isnan(ra):
            continue

        frame_val = la if np.isnan(ra) else (ra if np.isnan(la) else (la + ra)/2.0)
        if np.isnan(frame_val):
            continue

        count += 1
        mean_val += (frame_val - mean_val) / count

    return float(mean_val) if count > 0 else None


def knee_asymmetry_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = [
        'left_hip_x','left_hip_y','left_knee_x','left_knee_y','left_ankle_x','left_ankle_y',
        'right_hip_x','right_hip_y','right_knee_x','right_knee_y','right_ankle_x','right_ankle_y'
    ]
    if any(c not in df.columns for c in req):
        return None

    df[req] = df[req].apply(pd.to_numeric, errors='coerce')

    def knee_angle_2d(hip, knee, ankle):
        v1 = hip - knee
        v2 = ankle - knee
        d = np.linalg.norm(v1) * np.linalg.norm(v2)
        if d == 0:
            return np.nan
        cosv = np.dot(v1, v2) / d
        cosv = np.clip(cosv, -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    def symmetry_angle(L, R):
        if R == 0:
            return 100.0
        sa = (45 - np.degrees(np.arctan(L / R))) / 90 * 100
        return abs(sa)

    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        if r[['left_hip_x','left_hip_y','left_knee_x','left_knee_y',
              'left_ankle_x','left_ankle_y']].isnull().any():
            la = np.nan
        else:
            la = knee_angle_2d(
                np.array([r['left_hip_x'],   r['left_hip_y']]),
                np.array([r['left_knee_x'],  r['left_knee_y']]),
                np.array([r['left_ankle_x'], r['left_ankle_y']])
            )

        if r[['right_hip_x','right_hip_y','right_knee_x','right_knee_y',
              'right_ankle_x','right_ankle_y']].isnull().any():
            ra = np.nan
        else:
            ra = knee_angle_2d(
                np.array([r['right_hip_x'],   r['right_hip_y']]),
                np.array([r['right_knee_x'],  r['right_knee_y']]),
                np.array([r['right_ankle_x'], r['right_ankle_y']])
            )

        if np.isnan(la) or np.isnan(ra):
            continue

        sa = symmetry_angle(la, ra)
        if np.isnan(sa):
            continue

        count += 1
        mean_val += (sa - mean_val) / count

    return float(mean_val) if count > 0 else None


def knee_flexion_per_file(file_path, side="right"):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    def have_side(s):
        cols = [
            f"{s}_hip_x",f"{s}_hip_y",f"{s}_hip_z",
            f"{s}_knee_x",f"{s}_knee_y",f"{s}_knee_z",
            f"{s}_ankle_x",f"{s}_ankle_y",f"{s}_ankle_z"
        ]
        return all(c in df.columns for c in cols), cols

    left_ok,  left_cols  = have_side("left")
    right_ok, right_cols = have_side("right")

    if side == "left"  and not left_ok:  return None
    if side == "right" and not right_ok: return None
    if side == "both"  and not (left_ok or right_ok): return None

    use_cols = []
    if side in ("left","both") and left_ok:
        use_cols += left_cols
    if side in ("right","both") and right_ok:
        use_cols += right_cols

    df[use_cols] = df[use_cols].apply(pd.to_numeric, errors='coerce')

    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        la = ra = np.nan

        if side in ("left","both") and left_ok and not r[left_cols].isnull().any():
            la = _angle_3d(
                (r['left_hip_x'],   r['left_hip_y'],   r['left_hip_z']),
                (r['left_knee_x'],  r['left_knee_y'],  r['left_knee_z']),
                (r['left_ankle_x'], r['left_ankle_y'], r['left_ankle_z'])
            )

        if side in ("right","both") and right_ok and not r[right_cols].isnull().any():
            ra = _angle_3d(
                (r['right_hip_x'],   r['right_hip_y'],   r['right_hip_z']),
                (r['right_knee_x'],  r['right_knee_y'],  r['right_knee_z']),
                (r['right_ankle_x'], r['right_ankle_y'], r['right_ankle_z'])
            )

        if np.isnan(la) and np.isnan(ra):
            continue

        frame_val = la if np.isnan(ra) else (ra if np.isnan(la) else (la + ra)/2.0)
        if np.isnan(frame_val):
            continue

        count += 1
        mean_val += (frame_val - mean_val) / count

    return float(mean_val) if count > 0 else None


def arm_swing_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = ['left_shoulder_x','left_wrist_x','right_shoulder_x','right_wrist_x']
    if any(c not in df.columns for c in req):
        return None

    df[req] = df[req].apply(pd.to_numeric, errors='coerce')

    l_min, l_max, l_any = np.inf, -np.inf, False
    r_min, r_max, r_any = np.inf, -np.inf, False

    for lsx, lwx, rsx, rwx in zip(df['left_shoulder_x'],
                                  df['left_wrist_x'],
                                  df['right_shoulder_x'],
                                  df['right_wrist_x']):

        if pd.notna(lsx) and pd.notna(lwx):
            l_off = float(lwx) - float(lsx)
            if np.isfinite(l_off):
                l_any = True
                l_min = min(l_min, l_off)
                l_max = max(l_max, l_off)

        if pd.notna(rsx) and pd.notna(rwx):
            r_off = float(rwx) - float(rsx)
            if np.isfinite(r_off):
                r_any = True
                r_min = min(r_min, r_off)
                r_max = max(r_max, r_off)

    if not (l_any or r_any):
        return None

    left_amp  = (l_max - l_min) if l_any else np.nan
    right_amp = (r_max - r_min) if r_any else np.nan

    if np.isnan(left_amp) and np.isnan(right_amp):
        return None
    if np.isnan(left_amp):
        return float(right_amp)
    if np.isnan(right_amp):
        return float(left_amp)
    return float((left_amp + right_amp)/2.0)


def normalized_overstride_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    left  = ['left_hip_x','left_hip_y','left_hip_z',
             'left_ankle_x','left_ankle_y','left_ankle_z']
    right = ['right_hip_x','right_hip_y','right_hip_z',
             'right_ankle_x','right_ankle_y','right_ankle_z']

    if all(c in df.columns for c in left):
        use = left
        hipx,hipy,hipz, ankx,anky,ankz = (
            'left_hip_x','left_hip_y','left_hip_z',
            'left_ankle_x','left_ankle_y','left_ankle_z'
        )
    elif all(c in df.columns for c in right):
        use = right
        hipx,hipy,hipz, ankx,anky,ankz = (
            'right_hip_x','right_hip_y','right_hip_z',
            'right_ankle_x','right_ankle_y','right_ankle_z'
        )
    else:
        return None

    df[use] = df[use].apply(pd.to_numeric, errors='coerce')
    sub = df[use].dropna()
    if sub.empty:
        return None

    hip   = sub[[hipx,hipy,hipz]].to_numpy(float)
    ankle = sub[[ankx,anky,ankz]].to_numpy(float)

    over = ankle[:,0] - hip[:,0]   
    leg  = np.linalg.norm(ankle - hip, axis=1)
    leg  = np.where(leg == 0.0, np.nan, leg)

    vals = np.abs(over) / leg
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    return float(np.mean(vals))


def dorsiflexion_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    left = [
        'left_knee_x','left_knee_y','left_knee_z',
        'left_ankle_x','left_ankle_y','left_ankle_z',
        'left_foot_index_x','left_foot_index_y','left_foot_index_z'
    ]
    right = [
        'right_knee_x','right_knee_y','right_knee_z',
        'right_ankle_x','right_ankle_y','right_ankle_z',
        'right_foot_index_x','right_foot_index_y','right_foot_index_z'
    ]

    left_ok  = all(c in df.columns for c in left)
    right_ok = all(c in df.columns for c in right)
    if not (left_ok or right_ok):
        return None

    use = (left if left_ok else []) + (right if right_ok else [])
    df[use] = df[use].apply(pd.to_numeric, errors='coerce')

    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        la = ra = np.nan

        if left_ok and not r[left].isnull().any():
            la = _angle_3d(
                (r['left_knee_x'],  r['left_knee_y'],  r['left_knee_z']),
                (r['left_ankle_x'], r['left_ankle_y'], r['left_ankle_z']),
                (r['left_foot_index_x'], r['left_foot_index_y'], r['left_foot_index_z'])
            )

        if right_ok and not r[right].isnull().any():
            ra = _angle_3d(
                (r['right_knee_x'],  r['right_knee_y'],  r['right_knee_z']),
                (r['right_ankle_x'], r['right_ankle_y'], r['right_ankle_z']),
                (r['right_foot_index_x'], r['right_foot_index_y'], r['right_foot_index_z'])
            )

        if np.isnan(la) and np.isnan(ra):
            continue

        frame_val = la if np.isnan(ra) else (ra if np.isnan(la) else (la + ra)/2.0)
        if np.isnan(frame_val):
            continue

        count += 1
        mean_val += (frame_val - mean_val) / count

    return float(mean_val) if count > 0 else None


def _to_numeric(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")


def _angle_between_vecs(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    c = np.dot(v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def upperbody_verticality_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = [
        'left_shoulder_x','left_shoulder_y','left_shoulder_z',
        'right_shoulder_x','right_shoulder_y','right_shoulder_z',
        'left_hip_x','left_hip_y','left_hip_z',
        'right_hip_x','right_hip_y','right_hip_z'
    ]
    if any(c not in df.columns for c in req):
        return None

    _to_numeric(df, req)
    vertical_axis = np.array([0.0, 1.0, 0.0])
    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        if r[req].isnull().any():
            continue

        shoulder_mid = np.array([
            (r['left_shoulder_x'] + r['right_shoulder_x']) / 2.0,
            (r['left_shoulder_y'] + r['right_shoulder_y']) / 2.0,
            (r['left_shoulder_z'] + r['right_shoulder_z']) / 2.0
        ])
        hip_mid = np.array([
            (r['left_hip_x'] + r['right_hip_x']) / 2.0,
            (r['left_hip_y'] + r['right_hip_y']) / 2.0,
            (r['left_hip_z'] + r['right_hip_z']) / 2.0
        ])

        torso = shoulder_mid - hip_mid
        n = np.linalg.norm(torso)
        if n == 0:
            continue

        u = torso / n
        cosang = np.abs(np.dot(u, vertical_axis))
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))   

        count += 1
        mean_val += (angle - mean_val) / count

    return float(mean_val) if count > 0 else None


def upperbody_yaw_shoulders_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = [
        'left_shoulder_x','left_shoulder_y','left_shoulder_z',
        'right_shoulder_x','right_shoulder_y','right_shoulder_z'
    ]
    if any(c not in df.columns for c in req):
        return None

    _to_numeric(df, req)
    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        if r[req].isnull().any():
            continue

        S = np.array([
            r['right_shoulder_x'] - r['left_shoulder_x'],
            r['right_shoulder_y'] - r['left_shoulder_y'],
            r['right_shoulder_z'] - r['left_shoulder_z']
        ])
        S_h = np.array([S[0], 0.0, S[2]])  
        n = np.linalg.norm(S_h)
        if n == 0:
            continue

        yaw = np.degrees(np.arctan2(S_h[2], S_h[0]))
        yaw = abs(yaw)

        count += 1
        mean_val += (yaw - mean_val) / count

    return float(mean_val) if count > 0 else None


def upperbody_twist_per_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    req = [
        'left_shoulder_x','left_shoulder_y','left_shoulder_z',
        'right_shoulder_x','right_shoulder_y','right_shoulder_z',
        'left_hip_x','left_hip_y','left_hip_z',
        'right_hip_x','right_hip_y','right_hip_z'
    ]
    if any(c not in df.columns for c in req):
        return None

    _to_numeric(df, req)
    mean_val, count = 0.0, 0

    for _, r in df.iterrows():
        if r[req].isnull().any():
            continue

        S = np.array([
            r['right_shoulder_x'] - r['left_shoulder_x'],
            r['right_shoulder_y'] - r['left_shoulder_y'],
            r['right_shoulder_z'] - r['left_shoulder_z']
        ])
        H = np.array([
            r['right_hip_x'] - r['left_hip_x'],
            r['right_hip_y'] - r['left_hip_y'],
            r['right_hip_z'] - r['left_hip_z']
        ])

        S_h = np.array([S[0], 0.0, S[2]])
        H_h = np.array([H[0], 0.0, H[2]])

        ang = _angle_between_vecs(S_h, H_h)
        if np.isnan(ang):
            continue

        count += 1
        mean_val += (ang - mean_val) / count

    return float(mean_val) if count > 0 else None


def compute_metric_arrays(folder_a, folder_b):
    """Compute all metrics for group A vs group B (per CSV -> per runner)."""
    return {
        "Arm Swing":              (process_folder_metric(folder_a, arm_swing_per_file),
                                   process_folder_metric(folder_b, arm_swing_per_file)),
        "Hip Flexion":            (process_folder_metric(folder_a, hip_flexion_per_file),
                                   process_folder_metric(folder_b, hip_flexion_per_file)),
        "Overstriding":           (process_folder_metric(folder_a, normalized_overstride_per_file),
                                   process_folder_metric(folder_b, normalized_overstride_per_file)),
        "Knee Asymmetry":         (process_folder_metric(folder_a, knee_asymmetry_per_file),
                                   process_folder_metric(folder_b, knee_asymmetry_per_file)),
        "Ankle Dorsiflexion":     (process_folder_metric(folder_a, dorsiflexion_per_file),
                                   process_folder_metric(folder_b, dorsiflexion_per_file)),
        "Knee Flexion":           (process_folder_metric(folder_a, lambda p: knee_flexion_per_file(p, side="right")),
                                   process_folder_metric(folder_b, lambda p: knee_flexion_per_file(p, side="right"))),
        "Upper-Body Verticality": (process_folder_metric(folder_a, upperbody_verticality_per_file),
                                   process_folder_metric(folder_b, upperbody_verticality_per_file)),
        "Shoulder Yaw":           (process_folder_metric(folder_a, upperbody_yaw_shoulders_per_file),
                                   process_folder_metric(folder_b, upperbody_yaw_shoulders_per_file)),
        "Trunk Twist":            (process_folder_metric(folder_a, upperbody_twist_per_file),
                                   process_folder_metric(folder_b, upperbody_twist_per_file)),
    }


def metric_dict_to_long_df(metric_dict, label_a, label_b):
    rows = []
    for metric, (a_vals, b_vals) in metric_dict.items():
        for v in a_vals:
            if pd.notna(v):
                rows.append((metric, float(v), label_a))
        for v in b_vals:
            if pd.notna(v):
                rows.append((metric, float(v), label_b))
    return pd.DataFrame(rows, columns=["Metric", "Value", "Group"])


def ttest_per_metric(metric_dict):
    out = {}
    for metric, (a_vals, b_vals) in metric_dict.items():
        g1 = pd.Series(a_vals, dtype=float).dropna()
        g2 = pd.Series(b_vals, dtype=float).dropna()
        if len(g1) > 1 and len(g2) > 1:
            _, p = ttest_ind(g1, g2, equal_var=False)
            out[metric] = float(p)
        else:
            out[metric] = np.nan
    return out

def adjust_pvalues_in_family(p_dict, method):
    metrics = [m for m, p in p_dict.items() if np.isfinite(p)]
    if not metrics:
        return {m: np.nan for m in p_dict}
    raw = [p_dict[m] for m in metrics]
    _, padj, _, _ = multipletests(raw, method=method)
    adj_map = {m: float(q) for m, q in zip(metrics, padj)}
    return {m: adj_map.get(m, np.nan) for m in p_dict}


def print_family_summary(title, p_raw, p_adj, method):
    rows = []
    for m in sorted(p_raw, key=lambda k: (np.inf if np.isnan(p_adj[k]) else p_adj[k])):
        rows.append({
            "Metric": m,
            "p_raw": p_raw[m],
            f"p_adj_{method}": p_adj[m],
            "sig@0.05": (p_adj[m] < ALPHA) if np.isfinite(p_adj[m]) else False
        })
    df_sum = pd.DataFrame(rows)
    print(f"\n=== {title} | {method} | m = {np.isfinite(list(p_raw.values())).sum()} tests ===")
    print(df_sum.to_string(index=False))


def annotate_facet_axes(g, p_raw, p_adj, method):
    for ax, metric in zip(g.axes.flat, g.col_names):
        pr = p_raw.get(metric, np.nan)
        pa = p_adj.get(metric, np.nan)

        def fmt(p):
            if np.isnan(p): return "n/a"
            if p < 0.001:  return "< 0.001 ***"
            if p < 0.01:   return f"= {p:.3f} **"
            if p < 0.05:   return f"= {p:.3f} *"
            return f"= {p:.3f}"

        line1 = f"adj p ({method}) {fmt(pa)}"
        line2 = f"raw p {fmt(pr)}"

        y0, y1 = ax.get_ylim()
        pad = (y1 - y0) * 0.10 if np.isfinite(y1 - y0) else 1.0
        ax.set_ylim(y0, y1 + pad)
        ax.text(0.5, y1 + pad*0.18, line1,
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                transform=ax.transData)
        ax.text(0.5, y1 + pad*0.05, line2,
                ha="center", va="bottom",
                fontsize=10, color="gray",
                transform=ax.transData)


CONTRASTS = [
    ("Pro vs HS",        ("Pro Runners", "HS Runners"), (PRO_FOLDER, HS_FOLDER)),
    ("Pro Men vs Women", ("Pro Men",     "Pro Women"),  (PRO_MEN_FOLDER, PRO_WOMEN_FOLDER)),
    ("HS Men vs Women",  ("HS Men",      "HS Women"),   (HS_MEN_FOLDER,  HS_WOMEN_FOLDER)),
]


for title, (label_a, label_b), (folder_a, folder_b) in CONTRASTS:
    if not (os.path.isdir(folder_a) and os.path.isdir(folder_b)):
        print(f"[SKIP] {title}: missing folder(s) -> '{folder_a}' or '{folder_b}'")
        continue

    metric_dict = compute_metric_arrays(folder_a, folder_b)

    if SHOW_STATS:
        print(f"\n=== {title}: Per-metric descriptive stats ===")
        for metric, (a_vals, b_vals) in metric_dict.items():
            print(f"\n--- {metric} ---")
            print_stats(a_vals, b_vals, label_a, label_b)

    p_raw = ttest_per_metric(metric_dict)
    p_adj = adjust_pvalues_in_family(p_raw, P_ADJUST_METHOD)

    print_family_summary(title, p_raw, p_adj, P_ADJUST_METHOD)

    df_long = metric_dict_to_long_df(metric_dict, label_a, label_b)
    if df_long.empty:
        print(f"[NOTE] No usable data for {title}.")
        continue

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=df_long,
        kind="violin",
        x="Group", y="Value",
        col="Metric", col_wrap=3,
        inner="quartile",
        cut=0, bw_adjust=0.7,
        sharey=False, height=5, aspect=0.9, scale="width"
    )
    annotate_facet_axes(g, p_raw, p_adj, P_ADJUST_METHOD)
    g.fig.suptitle(title, y=1.02, fontsize=16)
    g.set_axis_labels("", "Value (raw units)")
    g.fig.subplots_adjust(top=0.92)
    plt.show()
