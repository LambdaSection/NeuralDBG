"""Shared helper for Panos/TrainProof reproducible bundle.

Problem it fixes (disclosed to Panos):
- TensorDiskCache.save() returns an absolute path with a random UUID.
- NeuralDbg.__exit__ calls disk_cache.cleanup() which DELETES every .pt
  file just referenced. Exported JSONs therefore point to deleted files.
- outputs/* and artifacts/* are gitignored, so a plain archive is missing.

What this helper does:
- preserve_cache(dbg): called INSIDE the `with NeuralDbg(...)` block,
  before __exit__ runs. Copies every cached .pt into
  examples/panos_bundle/tensor_cache/ (tracked) and returns a
  {absolute-posix -> relative-posix} mapping.
- patch_package(json_path, cache_mapping, extra): rewrites every
  metadata.tensor_cache_path to the bundled relative path where a copy
  exists; otherwise sets tensor_cache_path=None + tensor_cache_status=
  "NOT-CHECKED ..." (TrainProof vocabulary) and keeps the original in
  tensor_cache_requested_path. Merges `extra` dict under top-level
  "repro" key and mirrors the patched JSON into examples/panos_bundle/.
"""

import json
import shutil
from pathlib import Path

BUNDLE_DIR = Path("examples/panos_bundle")
BUNDLE_TC = BUNDLE_DIR / "tensor_cache"

NOT_CHECKED = (
    "NOT-CHECKED (tensor evicted by disk_cache.cleanup on context exit; "
    "original absolute path kept in tensor_cache_requested_path; see REPRO.md)"
)


def preserve_cache(dbg):
    """Copy cached tensors to the tracked bundle dir. Call INSIDE with-block."""
    BUNDLE_TC.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for src in list(getattr(getattr(dbg, "disk_cache", None), "_files", [])):
        try:
            src_p = Path(src)
            if not src_p.exists():
                continue
            dst = BUNDLE_TC / src_p.name
            if not dst.exists():
                shutil.copy2(src_p, dst)
            rel = dst.as_posix()
            mapping[str(src_p)] = rel
            try:
                mapping[str(src_p.absolute().as_posix())] = rel
            except Exception:
                pass
            # Also map plain posix without drive quirks
            mapping[src_p.as_posix()] = rel
        except Exception:
            continue
    return mapping


def _patch_event_list(events, cache_mapping):
    n_patched = 0
    n_missing = 0
    for ev in events or []:
        md = ev.get("metadata") or {}
        if "tensor_cache_path" in md and md["tensor_cache_path"]:
            orig = md["tensor_cache_path"]
            rel = cache_mapping.get(orig)
            if rel is None:
                # Fallback: basename match inside bundle dir
                try:
                    cand = BUNDLE_TC / Path(orig).name
                    if cand.exists():
                        rel = cand.as_posix()
                        cache_mapping[orig] = rel
                except Exception:
                    pass
            if rel:
                md["tensor_cache_path"] = rel
                md["tensor_cache_status"] = "bundled"
                n_patched += 1
            else:
                md["tensor_cache_requested_path"] = orig
                md["tensor_cache_path"] = None
                md["tensor_cache_status"] = NOT_CHECKED
                n_missing += 1
    return n_patched, n_missing


def patch_package(json_path, cache_mapping, extra):
    """Patch exported JSON in place + mirror into bundle dir."""
    p = Path(json_path)
    pkg = json.loads(p.read_text())
    np1, nm1 = _patch_event_list(pkg.get("events", []), cache_mapping)
    np2, nm2 = 0, 0
    for hyp in pkg.get("hypotheses", []):
        a, b = _patch_event_list(hyp.get("evidence", []), cache_mapping)
        np2 += a
        nm2 += b
    pkg["repro"] = extra
    p.write_text(json.dumps(pkg, indent=2, default=str))
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / p.name).write_text(json.dumps(pkg, indent=2, default=str))
    return (np1 + np2), (nm1 + nm2)
