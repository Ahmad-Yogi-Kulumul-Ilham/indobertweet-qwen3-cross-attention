"""Kaggle kernel: jalankan daftar eksperimen (PLAN) lalu simpan results/ ke output.

Dua fold dijalankan bersamaan, masing-masing di satu GPU T4 (CUDA_VISIBLE_DEVICES), sehingga
kedua GPU terpakai penuh. Fold yang hasilnya sudah ada di dataset results sebelumnya
(RESULTS_DATASET) dilewati, jadi eksperimen bisa dicicil lintas sesi Kaggle.
"""
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

# Sesi 2: fusion v2 (aux loss + branch dropout, tanpa gate kolaps) + ulang model tunggal
# agar probabilitas tersimpan untuk baseline ensemble.
# (task, run_name, argumen train.py, folds)
PLAN = [
    ("depresi", "fusion_cross_v2", "--model fusion --fusion cross_attention", "all"),
    ("distortion", "fusion_cross_v2", "--model fusion --fusion cross_attention", "all"),
    ("depresi", "indobertweet", "--model encoder", "all"),
    ("distortion", "indobertweet", "--model encoder", "all"),
    ("depresi", "qwen3_lora", "--model llm", "all"),
    ("distortion", "qwen3_lora", "--model llm", "all"),
]
# Batasi ke run tertentu (mis. untuk kernel paralel); None = semua isi PLAN
ONLY_RUNS = None
# Tanpa gradient checkpointing (~30% lebih cepat); otomatis diulang dengan checkpointing jika gagal/OOM
EXTRA_ARGS = "--no_grad_ckpt"
TIME_BUDGET_HOURS = 10.5

WORK = "/tmp/proj"  # di luar /kaggle/working agar salinan data tidak ikut jadi output
OUT = "/kaggle/working/results"
T0 = time.time()
LOG_LOCK = threading.Lock()


def log(msg):
    with LOG_LOCK:
        print(msg, flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/run_log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def sh(cmd, check=True, env=None, log_file=None):
    """Jalankan perintah; output ke log_file (job paralel) atau ke log utama."""
    log(f"[{(time.time() - T0) / 60:.1f} mnt] $ {cmd}")
    full_env = dict(os.environ, TQDM_MININTERVAL="120", **(env or {}))
    proc = subprocess.Popen(cmd, shell=True, cwd=WORK if os.path.isdir(WORK) else None,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", env=full_env)
    sink = open(log_file, "a", encoding="utf-8") if log_file else None
    for line in proc.stdout:
        if sink:
            sink.write(line)
            sink.flush()
            if "TEST acc" in line or "Error" in line or "error:" in line:
                log(line.rstrip())
        else:
            log(line.rstrip())
    rc = proc.wait()
    if sink:
        sink.close()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def setup():
    src = glob.glob("/kaggle/input/**/src/build_datasets.py", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(src)), WORK, dirs_exist_ok=True)
    for prev in glob.glob("/kaggle/input/**/results/*/*/fold*.json", recursive=True):
        rel = prev.split("/results/", 1)[1]
        os.makedirs(os.path.dirname(f"{WORK}/results/{rel}"), exist_ok=True)
        shutil.copy(prev, f"{WORK}/results/{rel}")
    sh(f"{sys.executable} -m pip install -q -U 'transformers>=4.51' peft bitsandbytes accelerate openpyxl")
    sh(f"{sys.executable} src/build_datasets.py")
    for t in sorted({p[0] for p in PLAN}):
        extra = " --min_per_class 30" if t == "distortion_type" else ""
        sh(f"{sys.executable} src/prepare_data.py --input data/raw_std/{t}.csv --out_dir data/{t}{extra}")


def jobs():
    out = []
    for task, run, args, folds in PLAN:
        if ONLY_RUNS and run not in ONLY_RUNS:
            continue
        wanted = range(5) if folds == "all" else [int(f) for f in folds.split(",")]
        out += [(task, run, args, f) for f in wanted
                if not os.path.exists(f"{WORK}/results/{task}/{run}/fold{f}.json")]
    return out


def sync():
    with LOG_LOCK:
        shutil.copytree(f"{WORK}/results", OUT, dirs_exist_ok=True)


def worker(gpu, q):
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    while True:
        try:
            task, run, args, fold = q.get_nowait()
        except queue.Empty:
            return
        if (time.time() - T0) / 3600 > TIME_BUDGET_HOURS:
            log(f"[GPU {gpu}] waktu habis, {task}/{run} fold {fold} dilanjutkan di sesi berikutnya")
            continue
        base = f"{sys.executable} src/train.py --task {task} --run_name {run} {args} --folds {fold}"
        logf = f"{OUT}/logs/{task}_{run}_fold{fold}.txt"
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
        rc = sh(f"{base} {EXTRA_ARGS}", check=False, env=env, log_file=logf)
        if rc != 0 and EXTRA_ARGS:
            log(f"[GPU {gpu}] {task}/{run} fold {fold} gagal (exit {rc}), ulang dengan gradient checkpointing")
            rc = sh(base, check=False, env=env, log_file=logf)
        if rc != 0:
            log(f"GAGAL: {task}/{run} fold {fold} (exit {rc}), lihat {logf}")
        sync()


def main():
    sh("nvidia-smi")
    setup()
    q = queue.Queue()
    for j in jobs():
        q.put(j)
    n_gpu = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True,
                                      text=True).stdout.strip() or 1))
    log(f"{q.qsize()} job, {n_gpu} GPU paralel")
    threads = [threading.Thread(target=worker, args=(g, q)) for g in range(n_gpu)]
    for t in threads:
        t.start()
        time.sleep(30)  # jeda agar unduhan model pertama tidak bertabrakan
    for t in threads:
        t.join()
    sh(f"for t in $(ls results); do {sys.executable} src/aggregate.py --task $t; done", check=False)
    sync()


if __name__ == "__main__":
    main()
