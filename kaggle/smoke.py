"""Kaggle kernel: setup + data prep + uji cepat fusion (fold 0, 500 sampel, 1 epoch)."""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

WORK = "/kaggle/working/proj"


def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=WORK if os.path.isdir(WORK) else None)


def setup_project():
    hits = glob.glob("/kaggle/input/**/build_datasets.py", recursive=True)
    if hits:
        root = os.path.dirname(os.path.dirname(hits[0]))
        shutil.copytree(root, WORK, dirs_exist_ok=True)
    else:  # dataset diunggah sebagai zip per folder
        os.makedirs(WORK, exist_ok=True)
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            name = os.path.splitext(os.path.basename(z))[0]
            target = os.path.join(WORK, name if name in ("src", "data") else "")
            with zipfile.ZipFile(z) as f:
                f.extractall(target)
        for f in glob.glob("/kaggle/input/**/*.*", recursive=True):
            if not f.endswith(".zip") and os.path.isfile(f):
                shutil.copy(f, WORK)
    print("Isi proyek:", sorted(os.listdir(WORK)), flush=True)


def main():
    sh("nvidia-smi")
    setup_project()
    sh("ls -R | head -40")
    sh(f"{sys.executable} -m pip install -q -U 'transformers>=4.51' peft bitsandbytes accelerate openpyxl")
    sh(f"{sys.executable} -c \"import torch, transformers, peft, bitsandbytes; "
       f"print('torch', torch.__version__, 'transformers', transformers.__version__, "
       f"'peft', peft.__version__, 'bnb', bitsandbytes.__version__, 'gpus', torch.cuda.device_count())\"")
    sh(f"{sys.executable} src/build_datasets.py")
    for t in ["depresi", "suicide", "depresi_anxiety", "distortion", "multiclass"]:
        sh(f"{sys.executable} src/prepare_data.py --input data/raw_std/{t}.csv --out_dir data/{t}")
    sh(f"{sys.executable} src/prepare_data.py --input data/raw_std/distortion_type.csv "
       f"--out_dir data/distortion_type --min_per_class 30")
    sh(f"{sys.executable} src/train.py --task depresi --model fusion --run_name debug_fusion "
       f"--folds 0 --epochs 1 --max_train_samples 500")
    sh("cat results/depresi/debug_fusion/fold0.json | head -60")
    shutil.copytree(f"{WORK}/results", "/kaggle/working/results", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
