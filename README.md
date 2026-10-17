# 9th ABC Challenge: BOXRR-23 Dataset Builder

This pipeline automates the downloading, filtering, and resampling of the BOXRR-23 dataset (4.7 million XR head-tracking recordings) for the 9th ABC Challenge. It enforces the exact challenge thresholds ( $\ge 17^\circ$ sweep, $\ge 15^\circ$/s velocity, 85–110 Hz native sampling) and outputs a mathematically verified, 100 Hz resampled HDF5 file.

The script is heavily fault-tolerant. It processes files incrementally, writes to atomic temporary files, and maintains a progress ledger. If your internet disconnects or your computer reboots, the script resumes exactly where it left off without re-downloading failed files.

## 1. Install Dependencies

Ensure you have Python 3.8 or higher installed. Install the required libraries using pip:

```bash
pip install numpy h5py pyarrow requests tqdm huggingface_hub

```

## 2. Link Hugging Face (Required for Rate Limits)

The dataset contains over 4.7 million individual Parquet files. Downloading them anonymously will trigger severe server-side rate limits (HTTP 429) from Hugging Face.

1. Create a free Hugging Face account at huggingface.co.
2. Navigate to **Settings > Access Tokens** and create a new **Read** token.
3. Link your machine by running the CLI login in your terminal and pasting your token:

```bash
huggingface-cli login

```

Alternatively, you can export the token directly in your terminal session before running the script:

```bash
export HF_TOKEN="hf_your_token_here"

```

## 3. Run the Pipeline

Save the provided script as `build_abc_dataset.py`.

**Step A: The Smoke Test (Highly Recommended)**
Before committing to a multi-day download, run a 200-file test to verify your Hugging Face authentication, network stability, and mathematical filtering.

```bash
python build_abc_dataset.py --limit 200

```

*Note: The very first time you run the script, it will pause for several minutes to build a local map of all 531,483 repository paths. Do not interrupt this process. Future runs will load this map instantly.*

**Step B: The Full Run**
Once the smoke test succeeds and writes segments to `abc_challenge_dataset.h5`, start the full extraction. The script uses 8 parallel workers by default.

```bash
python build_abc_dataset.py --workers 8

```

*(If your specific router or connection struggles with timeouts, reduce the worker count to `--workers 4`.)*

**Step C: Resuming a Stopped Run**
If the script is interrupted (Ctrl+C, power loss, or network drop), simply run the exact same command again:

```bash
python build_abc_dataset.py --workers 8

```

The script reads `abc_work/progress.tsv` and skips all previously processed files.

## 4. Storage & Architecture Notes

* **Staging Phase:** Accepted segments are saved as compressed `.npz` files inside `abc_work/staging/`.
* **Merge Phase:** Once all files are processed, the script merges the staging files into `abc_challenge_dataset.h5`.
* **Disk Space Warning:** During the final merge, both the `staging` folder and the `.h5` file exist simultaneously. Ensure you have at least 100 GB of free space before starting the full run, or pass the `--no-merge` flag to pause before the final assembly.

## 5. Dataset Citation

If you use this pipeline for a publication or challenge submission, the BOXRR-23 dataset creators require the following citation:

> Nair, Vivek, et al. "Berkeley open extended reality recordings 2023 (BOXRR-23): 4.7 million motion capture recordings from 105,000 XR users." *IEEE Transactions on Visualization and Computer Graphics* 30.5 (2024): 2239-2246. Dataset DOI: 10.25350/B5NP4V
