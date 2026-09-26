"""
Download and extract the additional Loghub datasets (src/multi_dataset_registry.py).

Subcommands:
  verify-url  HEAD-check the dataset's Zenodo URL before downloading.
  download    Stream the archive to disk and extract it to data/raw/<name>_raw/, preserving its
              internal structure (single file or directory tree).
  sample-raw  Print the first raw lines via the dataset's parser, to check the field layout.

On failure, verify-url and download write a short note to results/multi_dataset/<name>/acquisition.md
and exit nonzero.

Usage:
    python src/multi_dataset_acquire.py verify-url --dataset openstack
    python src/multi_dataset_acquire.py download --dataset openstack
    python src/multi_dataset_acquire.py sample-raw --dataset openstack --lines 20
"""

import argparse
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multi_dataset_registry import DATASETS, dataset_url, raw_root, require_known, results_dir

DOWNLOAD_CHUNK = 1 << 20
HEAD_TIMEOUT_S = 30
DOWNLOAD_TIMEOUT_S = 300

PARSER_MODULES = {
    "spark": "src.parser_spark",
    "hadoop": "src.parser_hadoop",
    "openstack": "src.parser_openstack",
    "windows": "src.parser_windows",
    "zookeeper": "src.parser_zookeeper",
    "android": "src.parser_android",
}


def write_note(name: str, filename: str, text: str):
    out_dir = results_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(text)
    print(text)


def cmd_verify_url(args):
    url = dataset_url(args.dataset)
    print(f"[verify-url] HEAD {url}", flush=True)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=HEAD_TIMEOUT_S) as resp:
            status = resp.status
            size = resp.headers.get("Content-Length", "unknown")
    except urllib.error.HTTPError as e:
        write_note(
            args.dataset, "acquisition.md",
            f"# {args.dataset} acquisition\n\nHEAD {url} -> HTTP {e.code}. "
            "URL does not resolve -- SKIPPING this dataset.\n",
        )
        sys.exit(2)
    except Exception:
        print("FATAL: HEAD request failed (network error, not a 404).", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if status >= 400:
        write_note(
            args.dataset, "acquisition.md",
            f"# {args.dataset} acquisition\n\nHEAD {url} -> HTTP {status}. SKIPPING this dataset.\n",
        )
        sys.exit(2)

    print(f"[verify-url] OK -- HTTP {status}, Content-Length={size}")


def _extract(archive_path: Path, dest_root: Path):
    """Extracts every regular file in the archive under dest_root, preserving its internal
    structure; src/parser_common.py's iter_raw_lines walks either a single file or a tree."""
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            raise ValueError("archive contains no regular files")
        print(f"[download] extracting {len(members):,} file(s) from archive")
        tar.extractall(path=dest_root, members=members)


def cmd_download(args):
    dest_root = raw_root(args.dataset)
    if dest_root.exists() and not args.force:
        print(f"{dest_root} already exists -- skipping download (pass --force to redo).")
        return

    url = dataset_url(args.dataset)
    tmp_dir = Path(f"data/raw/.{args.dataset}_download_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / f"{args.dataset}.tar.gz"

    print(f"[download] fetching {url}", flush=True)
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as response, open(archive_path, "wb") as out:
            shutil.copyfileobj(response, out, length=DOWNLOAD_CHUNK)
    except Exception:
        write_note(
            args.dataset, "acquisition.md",
            f"# {args.dataset} acquisition\n\nDownload of {url} failed. See stderr for the traceback. "
            "SKIPPING this dataset.\n",
        )
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(2)
    print(f"[download] wrote {archive_path} ({archive_path.stat().st_size:,} bytes) in {time.time() - t0:.0f}s", flush=True)

    try:
        _extract(archive_path, dest_root)
    except Exception:
        write_note(
            args.dataset, "acquisition.md",
            f"# {args.dataset} acquisition\n\nDownloaded {url} but extraction failed (unexpected archive "
            "shape or corrupt download). See stderr for the traceback. SKIPPING this dataset.\n",
        )
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(dest_root, ignore_errors=True)
        sys.exit(2)

    archive_path.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    n_files = sum(1 for p in dest_root.rglob("*") if p.is_file()) if dest_root.is_dir() else 1
    print(f"[download] {dest_root} ready -- {n_files:,} file(s)")


def cmd_sample_raw(args):
    module_name = PARSER_MODULES[args.dataset]
    __import__(module_name)
    mod = sys.modules[module_name]
    from src.parser_common import sample_raw_lines

    sample_raw_lines(mod.RAW_ROOT, args.lines)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    for name, fn, extra in [
        ("verify-url", cmd_verify_url, None),
        ("download", cmd_download, "force"),
        ("sample-raw", cmd_sample_raw, "lines"),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
        if extra == "force":
            p.add_argument("--force", action="store_true")
        if extra == "lines":
            p.add_argument("--lines", type=int, default=20)
        p.set_defaults(func=fn)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    require_known(args.dataset)
    args.func(args)


if __name__ == "__main__":
    main()
