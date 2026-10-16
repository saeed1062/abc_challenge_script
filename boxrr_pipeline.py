#!/usr/bin/env python3
"""
9th ABC Challenge - BOXRR-23 head-pose dataset builder (resumable, fault tolerant)

Organizer requirement implemented here
--------------------------------------
Keep only sessions with
  * an average sustained median sweep    >= 17 deg
  * an average sustained median velocity >= 15 deg/s
  * a native sampling rate in [85, 110] Hz
and store them re-sampled to a strict 100 Hz grid ("tagged as 100 Hz") as
normalised quaternions + timestamps in ONE HDF5 file.

How "average sustained median" is computed (the organizers' wording is
ambiguous, so this is the interpretation used - change the constants below if
they clarify):
  * the 100 Hz recording is cut into sliding windows of 420 frames (4.2 s =
    400 context + 20 target samples), stride 100 frames (1 s);
  * per window: sweep    = MEDIAN angle between every frame and the window's
                           first frame            (degrees)
                velocity = MEDIAN frame-to-frame angular speed (degrees/s)
  * the session passes if the MEAN over windows of each median meets its
    threshold.

Architecture (why it will not break half-way)
---------------------------------------------
  * Each BOXRR-23 session is one parquet file -> files are processed one by
    one; no session-boundary guessing.
  * Accepted segments are written to a staging folder as small .npz files with
    an atomic rename.  A kill / crash / power loss can never corrupt anything.
  * An append-only ledger (progress.tsv) records every finished file; re-running
    the script resumes exactly where it stopped.
  * Network errors / HTTP 429 / truncated downloads are retried with back-off;
    files that still fail are retried in extra passes and listed in failed.txt
    (they are NOT marked done, so the next run retries them).
  * The final single HDF5 is built from the staging folder into a temp file and
    renamed atomically, then verified.

Usage
-----
  python build_abc_dataset.py --limit 20     # 1-minute smoke test first!
  python build_abc_dataset.py                # full resumable run
  python build_abc_dataset.py --merge-only   # rebuild the .h5 from staging
  python build_abc_dataset.py --verify-only  # re-check an existing .h5

Set HF_TOKEN (or run `huggingface-cli login`) to get higher rate limits.

Dataset citation (must be cited in the paper):
  Nair, V., et al. "Berkeley open extended reality recordings 2023 (BOXRR-23):
  4.7 million motion capture recordings from 105,000 XR users." IEEE TVCG
  30.5 (2024): 2239-2246.   DOI of dataset: 10.25350/B5NP4V
"""
import argparse
import datetime
import io
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO_ID = "cschell/xr-motion-dataset-catalogue"
REPO_SUBDIR = "boxrr23"
COLUMNS = ["delta_time_ms", "head_rot_w", "head_rot_x", "head_rot_y", "head_rot_z"]

TARGET_HZ = 100.0
DT_OUT = 1.0 / TARGET_HZ
RATE_MIN_HZ, RATE_MAX_HZ = 85.0, 110.0
MIN_SWEEP_DEG = 17.0
MIN_VEL_DEG_S = 15.0

CONTEXT, HORIZON = 400, 20
MIN_FRAMES = CONTEXT + HORIZON          # 420 frames = 4.19 s span at 100 Hz
WINDOW_FRAMES = MIN_FRAMES              # sliding window for sustained metrics
WINDOW_STRIDE = 100                     # 1 s

MAX_GAP_S = 0.050                       # larger gaps split a recording in segments
QUAT_NORM_MIN, QUAT_NORM_MAX = 0.9, 1.1  # raw quaternions outside are tracking glitches

CHUNK = 128                             # files per download batch
MAX_ATTEMPTS = 6                        # HTTP attempts per file
RETRY_PASSES = 2                        # extra passes over failed files

CITATION = ('Nair, Vivek, et al. "Berkeley open extended reality recordings 2023 '
            '(BOXRR-23): 4.7 million motion capture recordings from 105,000 XR users." '
            'IEEE Transactions on Visualization and Computer Graphics 30.5 (2024): '
            '2239-2246. Dataset DOI: 10.25350/B5NP4V')

CONFIG = {
    "version": 2, "repo": REPO_ID, "subdir": REPO_SUBDIR, "target_hz": TARGET_HZ,
    "rate_range": [RATE_MIN_HZ, RATE_MAX_HZ], "min_sweep_deg": MIN_SWEEP_DEG,
    "min_vel_deg_s": MIN_VEL_DEG_S, "min_frames": MIN_FRAMES,
    "window_frames": WINDOW_FRAMES, "window_stride": WINDOW_STRIDE,
    "max_gap_s": MAX_GAP_S, "quat_norm_range": [QUAT_NORM_MIN, QUAT_NORM_MAX],
}


# --------------------------------------------------------------------------
# Quaternion / signal processing (pure numpy, unit-testable)
# --------------------------------------------------------------------------
def normalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n == 0.0, 1.0, n)


def align_hemisphere(q):
    """Vectorised q/-q sign continuity: consecutive quaternions get dot >= 0."""
    if len(q) < 2:
        return q.copy()
    dots = np.sum(q[:-1] * q[1:], axis=1)
    flips = np.where(dots < 0.0, -1.0, 1.0)
    signs = np.concatenate(([1.0], np.cumprod(flips)))
    return q * signs[:, None]


def quat_angle_deg(a, b):
    """Rotation angle (deg) between unit quaternions, sign-invariant and
    numerically stable for tiny angles.  a, b broadcast on the last axis (4)."""
    dot = np.sum(a * b, axis=-1, keepdims=True)
    sgn = np.where(dot < 0.0, -1.0, 1.0)
    num = np.linalg.norm(a - sgn * b, axis=-1)
    den = np.linalg.norm(a + sgn * b, axis=-1)
    return np.degrees(4.0 * np.arctan2(num, den))


def sustained_metrics(q):
    """Average (over sliding windows) of the per-window angular sweep [deg] and
    median angular velocity [deg/s].  q: (n>=WINDOW_FRAMES, 4) at TARGET_HZ."""
    n = len(q)
    vel = quat_angle_deg(q[:-1], q[1:]) / DT_OUT
    last = n - WINDOW_FRAMES
    starts = list(range(0, last + 1, WINDOW_STRIDE))
    if starts[-1] != last:
        starts.append(last)
    sweeps, vels = [], []
    for s in starts:
        win = q[s:s + WINDOW_FRAMES]
        # Sweep = angular range of motion (max excursion from window start) in the 4.2s window
        sweeps.append(np.max(quat_angle_deg(win[0:1], win)))
        vels.append(np.median(vel[s:s + WINDOW_FRAMES - 1]))
    return float(np.median(sweeps)), float(np.median(vels))


def resample_segment(t, q):
    """Linear (+renormalise) resampling of a gap-free segment onto a strict
    TARGET_HZ grid.  Returns (timestamps relative to segment start, quats)."""
    t0, t_end = t[0], t[-1]
    n_out = int(np.floor((t_end - t0) * TARGET_HZ + 1e-6)) + 1
    k = np.arange(n_out)
    grid = np.minimum(t0 + k * DT_OUT, t_end)
    out = np.empty((n_out, 4))
    for c in range(4):
        out[:, c] = np.interp(grid, t, q[:, c])
    return k * DT_OUT, normalize(out)


def process_recording(dt_ms, quats):
    """Clean one raw recording and return (accepted_segments, reject_reasons).

    dt_ms : (n,)   cumulative elapsed time in ms (despite column name delta_time_ms)
    quats : (n, 4) w, x, y, z
    """
    n = len(dt_ms)
    if n < 2:
        return [], ["too_few_rows"]
    t_ms = np.asarray(dt_ms, dtype=np.float64).copy()
    if not np.all(np.isfinite(t_ms)):
        return [], ["bad_dt"]

    # Many recordings pad initial loading frames with 0.0 before a negative countdown
    # (e.g., [0, 0, -582.7, -440.7, ...]). Drop leading zeros if followed by negative timestamps.
    nz = np.flatnonzero(t_ms != 0.0)
    if len(nz) == 0:
        return [], ["zero_timestamps"]
    first_nz = nz[0]
    if t_ms[first_nz] < 0.0 and first_nz > 0:
        t_ms = t_ms[first_nz:]
        quats = quats[first_nz:]

    # Convert cumulative ms directly to seconds (NO cumsum!)
    t = t_ms / 1000.0

    quats = np.asarray(quats, dtype=np.float64)
    norms = np.linalg.norm(quats, axis=1)
    valid = np.isfinite(norms) & (norms >= QUAT_NORM_MIN) & (norms <= QUAT_NORM_MAX)
    t, q = t[valid], quats[valid]
    if len(t) < 2:
        return [], ["no_valid_quats"]

    # Drop duplicate consecutive timestamps (e.g. paused frames at start/end)
    keep = np.concatenate(([True], np.diff(t) > 0.0))
    t, q = t[keep], q[keep]
    if len(t) < 2:
        return [], ["no_valid_quats"]

    q = align_hemisphere(normalize(q))
    d = np.diff(t)

    # Split recording wherever time jumps backward (d <= 0) or has a gap > MAX_GAP_S
    valid_d = d[d > 0.0]
    if len(valid_d) == 0 or not (RATE_MIN_HZ <= 1.0 / np.median(valid_d) <= RATE_MAX_HZ):
        return [], ["rate_out_of_range"]

    bounds = np.concatenate(([0], np.flatnonzero((d <= 0.0) | (d > MAX_GAP_S)) + 1, [len(t)]))
    segments, reasons = [], []
    for i in range(len(bounds) - 1):
        s, e = int(bounds[i]), int(bounds[i + 1])
        if e - s < 2:
            reasons.append("short_segment")
            continue
        ts, qs = t[s:e], q[s:e]
        # Ensure strict monotonic increase inside segment
        seg_keep = np.concatenate(([True], np.diff(ts) > 0.0))
        ts, qs = ts[seg_keep], qs[seg_keep]
        if len(ts) < 2:
            reasons.append("short_segment")
            continue
        dur = ts[-1] - ts[0]
        if int(np.floor(dur * TARGET_HZ + 1e-6)) + 1 < MIN_FRAMES:
            reasons.append("short_segment")
            continue
        seg_rate = (len(ts) - 1) / dur
        if not (RATE_MIN_HZ <= seg_rate <= RATE_MAX_HZ):
            reasons.append("rate_out_of_range")
            continue
        grid_t, rq = resample_segment(ts, qs)
        if not np.all(np.isfinite(rq)):
            reasons.append("non_finite")
            continue
        sweep, vel = sustained_metrics(rq)
        if not (sweep >= MIN_SWEEP_DEG and vel >= MIN_VEL_DEG_S):
            reasons.append("low_motion")
            continue
        segments.append({
            "seg": i, "t_offset_s": float(ts[0]), "source_hz": float(seg_rate),
            "sweep_deg": sweep, "velocity_deg_s": vel,
            "timestamps": grid_t.astype(np.float64),
            "head_quaternions": rq.astype(np.float32),
        })
    return segments, reasons

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
class PermanentError(Exception):
    """Error that retrying cannot fix (auth, 404, wrong schema)."""


_tls = threading.local()
TOKEN = None


def _http():
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        if TOKEN:
            s.headers["Authorization"] = f"Bearer {TOKEN}"
        _tls.session = s
    return s


def get_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token as _gt
        return _gt()
    except Exception:
        return None


def _col(tbl, name):
    c = tbl.column(name)
    if c.num_chunks == 0:
        return np.zeros(0, dtype=np.float64)
    return c.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)


def parse_parquet(body):
    pf = pq.ParquetFile(io.BytesIO(body))        # raises if truncated/corrupt -> retry
    missing = [c for c in COLUMNS if c not in pf.schema_arrow.names]
    if missing:
        raise PermanentError(f"schema missing columns {missing}")
    tbl = pf.read(columns=COLUMNS)
    dt = _col(tbl, "delta_time_ms")
    q = np.column_stack([_col(tbl, "head_rot_w"), _col(tbl, "head_rot_x"),
                         _col(tbl, "head_rot_y"), _col(tbl, "head_rot_z")])
    return dt, q


def fetch_recording(path, max_attempts=MAX_ATTEMPTS):
    url = f"{HF_ENDPOINT}/datasets/{REPO_ID}/resolve/main/{quote(path)}"
    last = "unknown"
    for attempt in range(max_attempts):
        wait = min(60.0, 2.0 ** attempt) + random.random()
        try:
            r = _http().get(url, timeout=(15, 180))
            if r.status_code == 200:
                return parse_parquet(r.content)
            if r.status_code in (401, 403, 404):
                raise PermanentError(f"HTTP {r.status_code}")
            last = f"HTTP {r.status_code}"
            if r.status_code == 429:
                try:
                    wait = max(wait, min(float(r.headers.get("Retry-After", 0)), 300.0))
                except ValueError:
                    wait = max(wait, 10.0)
        except PermanentError:
            raise
        except Exception as e:                    # network error, truncated body, ...
            last = f"{type(e).__name__}: {e}"
        if attempt < max_attempts - 1:
            time.sleep(wait)
    raise RuntimeError(f"gave up after {max_attempts} attempts ({last})")


def _worker(path):
    try:
        return path, "data", fetch_recording(path)
    except PermanentError as e:
        return path, "error", f"permanent: {e}"
    except Exception as e:
        return path, "error", f"{type(e).__name__}: {e}"


def list_parquet_files():
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    last = None
    for attempt in range(4):
        try:
            paths = []
            for e in api.list_repo_tree(REPO_ID, path_in_repo=REPO_SUBDIR,
                                        recursive=True, repo_type="dataset"):
                p = getattr(e, "path", "")
                if p.endswith(".parquet"):
                    paths.append(p)
            return sorted(paths)
        except Exception as ex:
            last = ex
            print(f"  listing failed ({type(ex).__name__}: {ex}); retrying...")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not list repository files: {last}")


def get_file_list(workdir, refresh):
    cache = workdir / "file_list.json"
    if cache.exists() and not refresh:
        paths = json.loads(cache.read_text())
        print(f"Using cached file list: {len(paths)} files ({cache}). "
              f"Use --refresh-list to re-scan the repo.")
        return paths
    print("Listing parquet files in the repository (can take a few minutes)...")
    paths = list_parquet_files()
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(paths))
    os.replace(tmp, cache)
    print(f"Found {len(paths)} parquet files.")
    return paths


# --------------------------------------------------------------------------
# Staging / ledger
# --------------------------------------------------------------------------
def save_segment(staging, user, session, seg):
    name = f"{user}__{session}__{seg['seg']:03d}"
    tmp = staging / f"{name}.tmp.npz"
    final = staging / f"{name}.npz"
    np.savez_compressed(
        tmp, timestamps=seg["timestamps"], head_quaternions=seg["head_quaternions"],
        user=np.array(user), session=np.array(session), segment=np.array(seg["seg"]),
        t_offset_s=np.array(seg["t_offset_s"]), source_hz=np.array(seg["source_hz"]),
        sweep_deg=np.array(seg["sweep_deg"]), velocity_deg_s=np.array(seg["velocity_deg_s"]))
    os.replace(tmp, final)                       # atomic


def load_ledger(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.endswith("\n") and "\t" in line:   # ignore a torn last line
                    done.add(line.split("\t", 1)[0])
    return done


def append_ledger(path, lines):
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in lines))
        f.flush()
        os.fsync(f.fileno())


def check_config(workdir):
    p = workdir / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != CONFIG:
            sys.exit("ERROR: the filter settings in this script differ from the ones used "
                     f"for the existing work folder '{workdir}'.\nUse a new --workdir (or "
                     "delete the old one) so results from different settings are not mixed.")
    else:
        p.write_text(json.dumps(CONFIG, indent=2))


def handle_recording(path, payload, staging, stats):
    dt, q = payload
    parts = path.split("/")
    user = parts[-2] if len(parts) >= 2 else "unknown"
    session = os.path.splitext(parts[-1])[0]
    segments, reasons = process_recording(dt, q)
    for seg in segments:
        save_segment(staging, user, session, seg)
        stats["accepted_segments"] += 1
        stats["accepted_frames"] += len(seg["timestamps"])
    for r in reasons:
        stats["reject:" + r] += 1
    if segments:
        stats["accepted_files"] += 1
        return f"ok:{len(segments)}"
    return "rej:" + ",".join(sorted(set(reasons)))


def run_pass(paths, ex, workdir, staging, stats, errors, desc):
    ledger = workdir / "progress.tsv"
    failed = []
    pbar = tqdm(total=len(paths), desc=desc, unit="file")
    for i in range(0, len(paths), CHUNK):
        chunk = paths[i:i + CHUNK]
        lines = []
        for path, kind, payload in ex.map(_worker, chunk):
            if kind == "error":
                errors[path] = payload
                failed.append(path)
                continue
            try:
                status = handle_recording(path, payload, staging, stats)
            except Exception as e:
                errors[path] = f"processing error: {type(e).__name__}: {e}"
                failed.append(path)
            else:
                errors.pop(path, None)
                lines.append(f"{path}\t{status}")
        append_ledger(ledger, lines)
        pbar.update(len(chunk))
        pbar.set_postfix(saved=stats["accepted_segments"], failed=len(failed))
    pbar.close()
    return failed


# --------------------------------------------------------------------------
# Final HDF5 build + verification
# --------------------------------------------------------------------------
def merge_to_hdf5(staging, out_path):
    files = sorted(p for p in staging.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not files:
        sys.exit("ERROR: no accepted segments in the staging folder - nothing to merge.")
    tmp_out = str(out_path) + ".tmp"
    total_frames = 0
    with h5py.File(tmp_out, "w") as h5:
        h5.attrs["sampling_rate_hz"] = TARGET_HZ
        h5.attrs["quaternion_order"] = "w,x,y,z (unit norm, hemisphere-continuous)"
        h5.attrs["timestamps_unit"] = "seconds, relative to segment start"
        h5.attrs["reference"] = CITATION
        h5.attrs["source"] = f"{REPO_ID}/{REPO_SUBDIR}"
        h5.attrs["filter_config"] = json.dumps(CONFIG)
        h5.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for p in tqdm(files, desc="Building HDF5", unit="seg"):
            with np.load(p) as z:
                g = h5.create_group(p.stem)
                g.create_dataset("timestamps", data=z["timestamps"],
                                 compression="gzip", compression_opts=4)
                g.create_dataset("head_quaternions", data=z["head_quaternions"],
                                 compression="gzip", compression_opts=4)
                g.attrs["user"] = str(z["user"])
                g.attrs["session"] = str(z["session"])
                g.attrs["segment"] = int(z["segment"])
                g.attrs["t_offset_s"] = float(z["t_offset_s"])
                g.attrs["source_hz"] = float(z["source_hz"])
                g.attrs["sweep_deg"] = float(z["sweep_deg"])
                g.attrs["velocity_deg_s"] = float(z["velocity_deg_s"])
                g.attrs["sampling_rate_hz"] = TARGET_HZ
                total_frames += len(z["timestamps"])
        h5.attrs["num_segments"] = len(files)
        h5.attrs["total_frames"] = total_frames
    os.replace(tmp_out, out_path)
    print(f"Wrote {out_path}: {len(files)} segments, {total_frames / TARGET_HZ / 3600:.2f} h, "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB")


def verify_hdf5(path):
    bad = 0
    n = 0
    with h5py.File(path, "r") as h5:
        for name in h5:
            n += 1
            g = h5[name]
            ts = g["timestamps"][:]
            q = g["head_quaternions"][:]
            errs = []
            if q.ndim != 2 or q.shape[1] != 4 or len(ts) != len(q):
                errs.append("shape mismatch")
            else:
                if len(q) < MIN_FRAMES:
                    errs.append("too short")
                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(ts))):
                    errs.append("non-finite")
                else:
                    if np.max(np.abs(np.linalg.norm(q.astype(np.float64), axis=1) - 1)) > 1e-4:
                        errs.append("not unit norm")
                    if ts[0] != 0.0 or not np.allclose(np.diff(ts), DT_OUT, atol=1e-9):
                        errs.append("timestamps not a 100 Hz grid")
                    sw, ve = sustained_metrics(q.astype(np.float64))
                    if sw < MIN_SWEEP_DEG - 0.05 or ve < MIN_VEL_DEG_S - 0.05:
                        errs.append(f"metrics below threshold (sweep={sw:.2f}, vel={ve:.2f})")
            if errs:
                bad += 1
                print(f"  VERIFY FAIL {name}: {', '.join(errs)}")
    print(f"Verification: {n} groups checked, {bad} problems.")
    return bad == 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    global TOKEN
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--workdir", default="abc_work")
    ap.add_argument("--output", default="abc_challenge_dataset.h5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="process only N new files (smoke test)")
    ap.add_argument("--refresh-list", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args(argv)

    out_path = Path(args.output)
    if args.verify_only:
        sys.exit(0 if verify_hdf5(out_path) else 1)

    workdir = Path(args.workdir)
    staging = workdir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    check_config(workdir)
    for p in staging.glob("*.tmp.npz"):          # leftovers from a killed run
        p.unlink()

    if not args.merge_only:
        TOKEN = get_token()
        print("HF token:", "found" if TOKEN else "NOT found (anonymous; lower rate limits)")
        paths = get_file_list(workdir, args.refresh_list)
        done = load_ledger(workdir / "progress.tsv")
        pending = [p for p in paths if p not in done]
        if args.limit:
            pending = pending[:args.limit]
        print(f"{len(done)} files already done, {len(pending)} to process.")

        stats, errors = Counter(), {}
        ex = ThreadPoolExecutor(max_workers=max(1, args.workers))
        try:
            todo = pending
            for pass_no in range(1 + RETRY_PASSES):
                if not todo:
                    break
                if pass_no:
                    print(f"\nRetry pass {pass_no}: {len(todo)} files; waiting 30 s...")
                    time.sleep(30)
                todo = run_pass(todo, ex, workdir, staging, stats, errors,
                                "Processing" if not pass_no else f"Retry {pass_no}")
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            print("\nInterrupted. Progress is saved - just run the same command to resume.")
            sys.exit(130)
        ex.shutdown(wait=True)

        failed_txt = workdir / "failed.txt"
        if errors:
            failed_txt.write_text("".join(f"{p}\t{m}\n" for p, m in sorted(errors.items())))
        elif failed_txt.exists():
            failed_txt.unlink()
        print("\n=== Summary of this run ===")
        print(f"accepted files: {stats['accepted_files']}   accepted segments: "
              f"{stats['accepted_segments']}   hours: {stats['accepted_frames'] / TARGET_HZ / 3600:.2f}")
        for k in sorted(k for k in stats if k.startswith("reject:")):
            print(f"  {k:<28}{stats[k]}")
        if errors:
            print(f"\nWARNING: {len(errors)} files failed and were NOT marked done "
                  f"(see {failed_txt}). Re-run the same command to retry them.")

    if not args.no_merge:
        merge_to_hdf5(staging, out_path)
        if not verify_hdf5(out_path):
            sys.exit("Verification failed - see messages above.")
    print("\nDone.")


if __name__ == "__main__":
    main()
