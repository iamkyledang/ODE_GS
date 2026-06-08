#
# gpu.py — Unified GPU configuration for 4D Gaussian Splatting experiments.
#
# Three hardware tiers are detected automatically once at import time:
#
#   LOW   – NVIDIA GeForce RTX 3060 Laptop  ( 6 GB VRAM, Ampere CC 8.6)
#   HIGH  – NVIDIA GeForce RTX 4090         (24 GB VRAM, Ada   CC 8.9)
#   ULTRA – NVIDIA RTX PRO 6000             (48 GB VRAM, Ada/Blackwell)
#
# Every downstream module (train.py, render.py, metrics.py, full_eval.py)
# imports the single GPU_CFG instance and apply_torch_global_settings().
# There is NO duplicate hardware-detection logic anywhere else.
#
# Usage (add near the very top of each entry-point, before first CUDA alloc):
#
#   from gpu import GPU_CFG, apply_torch_global_settings, log_gpu_info
#   apply_torch_global_settings()   # sets PYTORCH_CUDA_ALLOC_CONF + torch flags
#   log_gpu_info()                  # print one-line hardware summary
#
# Then read config values:
#   GPU_CFG.batch_size, GPU_CFG.num_workers, GPU_CFG.max_gaussians, …
#

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Hardware tiers
# ─────────────────────────────────────────────────────────────────────────────

class _Tier:
    LOW   = "LOW"    # RTX 3060 Laptop  –  6 GB
    HIGH  = "HIGH"   # RTX 4090         – 24 GB
    ULTRA = "ULTRA"  # RTX PRO 6000     – 48 GB+


# ─────────────────────────────────────────────────────────────────────────────
# Internal detection helpers (nvidia-smi only — no CUDA init at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _nvidia_smi_query(fields: str):
    """Run nvidia-smi for device 0 and return a list of field values.
    Returns an empty list if nvidia-smi is unavailable or fails."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().splitlines()
            return [v.strip() for v in lines[0].split(',')]
        return []
    except Exception:
        return []


def _get_gpu_name() -> str:
    vals = _nvidia_smi_query("name")
    return vals[0] if vals else ""


def _get_vram_gb() -> float:
    """Total VRAM in GiB for device 0 (nvidia-smi reports in MiB)."""
    vals = _nvidia_smi_query("memory.total")
    try:
        return int(vals[0]) / 1024.0 if vals else 0.0
    except (ValueError, IndexError):
        return 0.0


def _detect_tier(name: str, vram_gb: float) -> str:
    """
    Determine GPU tier by name substrings first, then fall back to VRAM.
    Name matching is preferred because different GPUs can share the same VRAM.
    """
    n = name.lower()

    # ── ULTRA: RTX PRO 6000 / RTX 6000 Ada / any GPU with >= 40 GB ────────
    if "rtx pro 6000" in n or "rtx 6000 ada" in n:
        return _Tier.ULTRA
    if vram_gb >= 40.0:
        return _Tier.ULTRA

    # ── HIGH: RTX 4090 / any GPU with >= 20 GB ────────────────────────────
    if "4090" in n:
        return _Tier.HIGH
    if vram_gb >= 20.0:
        return _Tier.HIGH

    # ── LOW: RTX 3060 Laptop / any GPU with < 20 GB ───────────────────────
    return _Tier.LOW


def _is_ampere_or_newer() -> bool:
    """True for Ampere (CC >= 8) or newer, inferred from GPU name.
    Uses nvidia-smi so no CUDA initialisation happens at import time."""
    name = _get_gpu_name().lower()
    # RTX 30xx = Ampere, RTX 40xx = Ada Lovelace, RTX PRO 6000 / RTX 6000 Ada,
    # A100, H100, H200, B100 are all Ampere or newer.
    return any(x in name for x in (
        "rtx 30", "rtx 40", "rtx pro 6000", "rtx 6000 ada",
        "a100", "h100", "h200", "b100",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# GPUConfig dataclass — one instance per process, built once at import
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    gpu_name:  str   = ""
    vram_gb:   float = 0.0
    tier:      str   = _Tier.LOW
    is_ampere: bool  = False   # Ampere CC>=8 (RTX 3060 / 4090 / RTX PRO 6000)

    # ── Convenience tier booleans ─────────────────────────────────────────────
    is_low_vram:   bool = False   # RTX 3060 Laptop
    is_high_vram:  bool = False   # RTX 4090
    is_ultra_vram: bool = False   # RTX PRO 6000

    # ── CUDA memory allocator ─────────────────────────────────────────────────
    # Written to PYTORCH_CUDA_ALLOC_CONF by apply_torch_global_settings().
    # Must be applied before the first CUDA tensor allocation.
    cuda_alloc_conf: str = ""

    # ── DataLoader ────────────────────────────────────────────────────────────
    # batch_size: cameras rendered per gradient step (Gaussians are shared).
    batch_size:         int  = 1
    num_workers:        int  = 8
    pin_memory:         bool = False
    persistent_workers: bool = False

    # ── Gaussian densification ────────────────────────────────────────────────
    # max_gaussians caps VRAM + CPU bookkeeping (grad_accum, max_radii2D).
    # densify_grad_threshold_override, when set, replaces coarse/fine_init/after
    # thresholds — applied in train.py after argument parsing.
    max_gaussians:                   int            = 360_000
    densify_grad_threshold_override: Optional[float] = None

    # ── SSIM loss ─────────────────────────────────────────────────────────────
    # < 1.0 → bilinear downsample before SSIM, reducing its VRAM footprint ~4×
    # at 0.5 (halving each spatial dimension).
    ssim_downsample_factor: float = 1.0

    # ── Metrics ───────────────────────────────────────────────────────────────
    # lpips_on_gpu: run LPIPS networks on CUDA (True) or CPU (False).
    # metrics_cache_clear: "per_image" | "per_camera" | "end_only"
    lpips_on_gpu:        bool = True
    metrics_cache_clear: str  = "per_image"

    # ── Garbage collection / cache flush ─────────────────────────────────────
    # gc_interval: iterations between gc.collect() + torch.cuda.empty_cache().
    # aggressive_cache_clear: also flush after every densify/prune/grow call.
    # cache_clear_before_backward: flush allocator before loss.backward() to
    #   avoid fragmentation-induced OOM on small-VRAM GPUs.
    gc_interval:                 int  = 200
    aggressive_cache_clear:      bool = False
    cache_clear_before_backward: bool = False

    # ── full_eval.py CLI helpers ──────────────────────────────────────────────
    # Appended verbatim to the train.py subprocess command line in full_eval.py.
    # cli_num_workers_flag: e.g. "--num_workers 2" to limit DataLoader RAM use.
    # cli_mem_flags: densification threshold overrides (empty = use defaults).
    cli_num_workers_flag: str = ""
    cli_mem_flags:        str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Factory — constructs the right GPUConfig for the detected hardware
# ─────────────────────────────────────────────────────────────────────────────

def _build_config() -> GPUConfig:
    gpu_name  = _get_gpu_name()
    vram_gb   = _get_vram_gb()
    tier      = _detect_tier(gpu_name, vram_gb)
    is_ampere = _is_ampere_or_newer()

    # ─────────────────────────────────────────────────────────────────────────
    if tier == _Tier.LOW:
        # ── NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM) ───────────────────────
        #
        # Aggressive memory conservation is the primary goal.  Every parameter
        # is chosen to keep peak VRAM + system RAM within the 6 GB envelope.
        #
        # Allocator — max_split_size_mb:64
        #   Caps each cached block at 64 MB so the allocator recycles smaller
        #   fragments instead of holding one large slab per tensor.  This
        #   dramatically reduces fragmentation during densification where tensor
        #   sizes change every ~100 iterations.
        # roundup_power2_divisions:4
        #   Aligns allocations to 1/4 of the next power-of-2, reducing wasted
        #   padding on heterogeneous tensor sizes produced by Gaussian growth.
        #
        # batch_size=1: single camera per gradient step — the minimum footprint.
        # num_workers=2: only 2 prefetch buffers alive at once instead of 8–16,
        #   halving the shared-memory pressure from DataLoader workers.
        # pin_memory=False: host-pinned memory eats RAM; skip on 6 GB configs.
        # persistent_workers=False: workers are re-spawned each epoch to release
        #   their shared-memory segments between passes.
        # max_gaussians=150 000: caps VRAM growth from densification and limits
        #   CPU-side bookkeeping (grad_accum, max_radii2D arrays).
        # densify_grad_threshold_override=0.0004: higher threshold → fewer
        #   split candidates per interval → slower Gaussian growth → lower VRAM.
        # ssim_downsample_factor=0.5: compute SSIM at 25% area (0.5× each dim),
        #   cutting its activation-buffer VRAM requirement ~4×.
        # lpips_on_gpu=False: compute LPIPS perceptual losses on CPU to spare
        #   the already-tight 6 GB for the Gaussians and render buffers.
        # gc_interval=50: frequent GC + cache flush frees Python cyclic garbage
        #   that holds CUDA tensors before densification spikes VRAM.
        # aggressive_cache_clear=True: flush after every densify/prune/grow.
        # cache_clear_before_backward=True: flush before backward() to give
        #   the gradient computation a clean unfragmented allocator state.
        return GPUConfig(
            gpu_name  = gpu_name,
            vram_gb   = vram_gb,
            tier      = _Tier.LOW,
            is_ampere = is_ampere,
            is_low_vram   = True,
            is_high_vram  = False,
            is_ultra_vram = False,

            cuda_alloc_conf = "max_split_size_mb:64,roundup_power2_divisions:4",

            batch_size         = 1,
            num_workers        = 2,
            pin_memory         = False,
            persistent_workers = False,

            max_gaussians                   = 150_000,
            densify_grad_threshold_override = 0.0004,

            ssim_downsample_factor = 0.5,

            lpips_on_gpu        = False,
            metrics_cache_clear = "per_image",

            gc_interval                 = 50,
            aggressive_cache_clear      = True,
            cache_clear_before_backward = True,

            cli_num_workers_flag = "--num_workers 2",
            cli_mem_flags = (
                "--densify_grad_threshold_coarse 0.0004 "
                "--densify_grad_threshold_fine_init 0.0004 "
                "--densify_grad_threshold_after 0.0004 "
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    elif tier == _Tier.HIGH:
        # ── NVIDIA GeForce RTX 4090 (24 GB VRAM) ─────────────────────────────
        #
        # Maximise throughput.  VRAM is plentiful but not unlimited; settings
        # are tuned for best wall-clock speed with no quality compromise.
        #
        # Allocator — max_split_size_mb:512
        #   Larger split ceiling lets the allocator reuse big blocks across the
        #   large rendering passes produced by 24 GB of VRAM.
        # garbage_collection_threshold:0.9
        #   Defer internal GC until 90% of the reserved pool is used, keeping
        #   most fragments available for rapid re-allocation without manual
        #   torch.cuda.empty_cache() calls between every operation.
        #
        # batch_size=2: render 2 cameras per gradient step — doubles data
        #   throughput with Gaussians shared across the batch (no extra VRAM
        #   for the model weights), effectively 2× faster gradient signal.
        # num_workers=16: maximise CPU↔GPU prefetch parallelism; 16 is the
        #   sweet spot before the prefetch overhead exceeds the benefit on a
        #   typical 8–16 core workstation.
        # pin_memory=True / persistent_workers=True: pre-pinned host memory
        #   enables zero-copy DMA transfers; persistent workers avoid worker
        #   spawn/teardown overhead between epochs.
        # max_gaussians=360 000: the established quality/VRAM balance for 4090.
        # densify_grad_threshold_override=None: use method-config defaults.
        # ssim_downsample_factor=1.0: full-resolution SSIM.
        # gc_interval=200: infrequent GC; VRAM is ample.
        return GPUConfig(
            gpu_name  = gpu_name,
            vram_gb   = vram_gb,
            tier      = _Tier.HIGH,
            is_ampere = is_ampere,
            is_low_vram   = False,
            is_high_vram  = True,
            is_ultra_vram = False,

            cuda_alloc_conf = "max_split_size_mb:512,garbage_collection_threshold:0.9",

            batch_size         = 2,
            num_workers        = 16,
            pin_memory         = True,
            persistent_workers = True,

            max_gaussians                   = 360_000,
            densify_grad_threshold_override = None,

            ssim_downsample_factor = 1.0,

            lpips_on_gpu        = True,
            metrics_cache_clear = "per_camera",

            gc_interval                 = 200,
            aggressive_cache_clear      = False,
            cache_clear_before_backward = False,

            cli_num_workers_flag = "",
            cli_mem_flags        = "",
        )

    # ─────────────────────────────────────────────────────────────────────────
    else:  # _Tier.ULTRA
        # ── NVIDIA RTX PRO 6000 (48 GB VRAM, Ada / Blackwell) ────────────────
        #
        # Maximum quality and throughput.  With 48 GB VRAM the experiment can
        # sustain very large Gaussian counts, larger batches, and finer
        # densification thresholds without any memory-saving workarounds.
        #
        # Allocator — max_split_size_mb:1024
        #   1 GB block ceiling maximises allocator reuse across the large render
        #   passes that come with 600k Gaussians and batch_size=4.
        # garbage_collection_threshold:0.95
        #   Defer internal GC until 95% of the reserved pool is used.  With
        #   48 GB headroom, fragments stay in the pool for rapid re-allocation
        #   during densification bursts without wasting any rendering time.
        #
        # batch_size=4: 4 cameras per gradient step — 4× the gradient signal
        #   per optimizer step vs batch_size=1, improving convergence speed and
        #   final quality when training budgets are fixed in iterations.
        # num_workers=16: same ceiling as the 4090; bound by CPU core count.
        # pin_memory=True / persistent_workers=True: same benefits as HIGH.
        # max_gaussians=600 000: RTX PRO 6000 has enough VRAM and memory
        #   bandwidth to sustain a much denser scene representation, producing
        #   noticeably sharper reconstructions on high-frequency detail.
        # densify_grad_threshold_override=0.0001: lower threshold → more
        #   aggressive densification → denser point cloud within VRAM budget.
        # ssim_downsample_factor=1.0: full-resolution SSIM; no compromise.
        # metrics_cache_clear="end_only": minimal overhead; VRAM is plentiful.
        # gc_interval=500: very infrequent; no fragmentation pressure.
        return GPUConfig(
            gpu_name  = gpu_name,
            vram_gb   = vram_gb,
            tier      = _Tier.ULTRA,
            is_ampere = is_ampere,
            is_low_vram   = False,
            is_high_vram  = False,
            is_ultra_vram = True,

            cuda_alloc_conf = "max_split_size_mb:1024,garbage_collection_threshold:0.95",

            batch_size         = 4,
            num_workers        = 16,
            pin_memory         = True,
            persistent_workers = True,

            max_gaussians                   = 600_000,
            densify_grad_threshold_override = 0.0001,

            ssim_downsample_factor = 1.0,

            lpips_on_gpu        = True,
            metrics_cache_clear = "end_only",

            gc_interval                 = 500,
            aggressive_cache_clear      = False,
            cache_clear_before_backward = False,

            cli_num_workers_flag = "",
            cli_mem_flags = (
                "--densify_grad_threshold_coarse 0.0001 "
                "--densify_grad_threshold_fine_init 0.0001 "
                "--densify_grad_threshold_after 0.0001 "
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — evaluated once at import time.
# ─────────────────────────────────────────────────────────────────────────────

GPU_CFG: GPUConfig = _build_config()


# ─────────────────────────────────────────────────────────────────────────────
# Global PyTorch / CUDA settings — call once before the first CUDA allocation.
# ─────────────────────────────────────────────────────────────────────────────

def apply_torch_global_settings() -> None:
    """
    Apply all hardware-specific PyTorch and CUDA global settings.

    MUST be called before the first torch.cuda tensor allocation so that
    PYTORCH_CUDA_ALLOC_CONF is read by the caching allocator at its first
    initialisation.  Idempotent — safe to call more than once.

    Settings applied:
      • PYTORCH_CUDA_ALLOC_CONF  (via os.environ.setdefault)
      • torch.backends.cudnn.benchmark
      • torch.set_float32_matmul_precision("high")  [Ampere / Ada / Blackwell]
      • torch.backends.cuda.matmul.allow_tf32        [same]
      • torch.backends.cudnn.allow_tf32              [same]
    """
    if GPU_CFG.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", GPU_CFG.cuda_alloc_conf)

    # These are attribute writes only — they do NOT call cudaGetDeviceCount() or
    # any other CUDA API, so they are safe to set before safe_state() /
    # torch.cuda.set_device().  Skip silently on old PyTorch that lacks an attr.
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if GPU_CFG.is_ampere:
        # TF32 matmul: ~1.5–3× faster on Ampere / Ada / Blackwell vs full FP32,
        # with < 0.1% numerical error.  Safe for Gaussian splatting training.
        try:
            torch.set_float32_matmul_precision("high")  # added PyTorch 1.12
        except AttributeError:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

_TIER_LABEL = {
    _Tier.LOW:   "LOW   — NVIDIA GeForce RTX 3060 Laptop  ( 6 GB)",
    _Tier.HIGH:  "HIGH  — NVIDIA GeForce RTX 4090         (24 GB)",
    _Tier.ULTRA: "ULTRA — NVIDIA RTX PRO 6000             (48 GB+)",
}


def log_gpu_info() -> None:
    """Print a one-line hardware summary to stdout."""
    if not GPU_CFG.gpu_name:
        print("[gpu] No CUDA device — running on CPU.")
        return
    cfg = GPU_CFG
    tier_label = _TIER_LABEL.get(cfg.tier, cfg.tier)
    tf32_str   = "ON" if cfg.is_ampere else "OFF"
    print(
        f"[gpu] {cfg.gpu_name}  |  {cfg.vram_gb:.1f} GB VRAM  |  "
        f"tier={tier_label}  |  "
        f"batch={cfg.batch_size}  workers={cfg.num_workers}  "
        f"max_gaussians={cfg.max_gaussians:,}  "
        f"TF32={tf32_str}  pin_mem={cfg.pin_memory}"
    )
    if cfg.cuda_alloc_conf:
        print(f"[gpu] PYTORCH_CUDA_ALLOC_CONF={cfg.cuda_alloc_conf}")
    if cfg.densify_grad_threshold_override is not None:
        print(f"[gpu] densify_grad_threshold override={cfg.densify_grad_threshold_override}")
