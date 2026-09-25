"""HW1: run the CNN measurement grid on a Kaggle T4 and fit PySR equations.

Kaggle notebook cell (enable GPU and Internet in notebook settings):

    !python /kaggle/input/YOUR_DATASET/hw1/kaggle_measure.py

Or upload this file to /kaggle/working and run it there. Outputs go to
/kaggle/working/hw1/results by default. PySR and nvidia-ml-py are installed
automatically if absent; PySR's first import also downloads Julia dependencies.
Each point warms up, then uses separate forward passes for peak memory,
latency, and energy. Three independent trials are aggregated. Diagnostics are
written to measurements.csv and diagnostics.json.

The PySR fits are empirical suggestions. The homework's handwritten FLOP,
memory, latency, and energy derivations are still required separately.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path


BASE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIELDS = (
    "S", "B", "latency_s", "peak_memory_bytes", "energy_j", "status",
    "energy_method", "energy_j_from_counter", "energy_j_from_power",
    "is_validation", "repeats", "latency_repeats", "latency_window_s",
    "warmup_repeats", "warmup_window_s",
    "oom_retries", "latency_mad_s",
    "power_samples", "distinct_power_readings", "power_window_s",
    "free_memory_bytes_before", "gpu_temp_c_before", "gpu_temp_c_after",
    "sm_clock_mhz_before", "sm_clock_mhz_after", "power_w_before",
    "power_w_after", "measurement_warning",
    "trial_count", "latency_trial_mad_s", "energy_trial_mad_j",
    "peak_memory_range_bytes",
)
TRIAL_FIELDS = ("trial_index", *FIELDS)


def ensure_package(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        print(f"Installing {package} with {sys.executable} -m pip ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True)


def make_grid(seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    extra_s = rng.sample(sorted(set(range(32, 513, 16)) - set(BASE_SIZES)), 4)
    extra_b = rng.sample(sorted(set(range(1, 257)) - set(BASE_BATCHES)), 3)
    return sorted((*BASE_SIZES, *extra_s)), sorted((*BASE_BATCHES, *extra_b))


def build_model(torch):
    nn = torch.nn
    # No BatchNorm placement is specified in the layer table, so none is added.
    layers = [
        ("conv7_s2", nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)),
        ("relu1", nn.ReLU(inplace=True)),
        ("maxpool", nn.MaxPool2d(3, stride=2, padding=1)),
        ("conv5", nn.Conv2d(32, 64, 5, padding=2, bias=False)),
        ("relu2", nn.ReLU(inplace=True)),
        ("conv3_s2_a", nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)),
        ("relu3", nn.ReLU(inplace=True)),
        ("conv1_a", nn.Conv2d(128, 256, 1, bias=False)),
        ("relu4", nn.ReLU(inplace=True)),
        ("conv3_s2_b", nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)),
        ("relu5", nn.ReLU(inplace=True)),
        ("conv1_b", nn.Conv2d(256, 512, 1, bias=False)),
        ("relu6", nn.ReLU(inplace=True)),
        ("gap", nn.AdaptiveAvgPool2d(1)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(512, 256)),
        ("relu7", nn.ReLU(inplace=True)),
        ("fc2", nn.Linear(256, 100)),
    ]
    return nn.Sequential(OrderedDict(layers))


class PowerSampler:
    """Sample whole-GPU NVML power during a burst of synchronized forwards."""

    def __init__(self, pynvml, handle, interval_s: float = 0.02):
        self.nvml = pynvml
        self.handle = handle
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            watts = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            self.samples.append((time.perf_counter(), watts))
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def start(self) -> None:
        self._sample()

        def loop() -> None:
            while not self.stop_event.wait(self.interval_s):
                self._sample()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        if self.error is not None:
            raise RuntimeError("NVML power sampling failed") from self.error

    def energy_j(self, start: float, end: float) -> float:
        samples = self.samples
        if len(samples) < 2:
            raise RuntimeError("NVML returned fewer than two power samples")

        def interpolate(t: float) -> float:
            if t <= samples[0][0]:
                return samples[0][1]
            for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
                if t0 <= t <= t1:
                    return p0 + (p1 - p0) * (t - t0) / max(t1 - t0, 1e-12)
            return samples[-1][1]

        points = [(start, interpolate(start))]
        points.extend((t, p) for t, p in samples if start < t < end)
        points.append((end, interpolate(end)))
        return sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(points, points[1:]))


def gpu_state(pynvml, handle, suffix: str) -> dict:
    """Diagnostic readings outside the timed region; unsupported fields stay blank."""
    def read(fn):
        try:
            return fn()
        except Exception:
            return ""

    return {
        f"gpu_temp_c_{suffix}": read(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
        f"sm_clock_mhz_{suffix}": read(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
        f"power_w_{suffix}": read(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0),
    }


def select_nvml_handle(pynvml, torch, cuda_index: int):
    """Match CUDA and NVML by UUID; GPU ordinals are not always identical."""
    def normalized(value) -> str:
        if isinstance(value, bytes):
            value = value.decode()
        return str(value).lower().removeprefix("gpu-").replace("-", "")

    count = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
    gpu_uuid = getattr(torch.cuda.get_device_properties(cuda_index), "uuid", None)
    if gpu_uuid is not None:
        matches = [h for h in handles if normalized(pynvml.nvmlDeviceGetUUID(h)) == normalized(gpu_uuid)]
        if len(matches) == 1:
            return matches[0]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [item.strip() for item in visible.split(",") if item.strip()]
    if len(tokens) > cuda_index:
        token = tokens[cuda_index]
        if token.startswith("GPU-"):
            matches = [h for h in handles if normalized(pynvml.nvmlDeviceGetUUID(h)).startswith(normalized(token))]
            if len(matches) == 1:
                return matches[0]
        elif token.isdigit() and int(token) < count:
            return handles[int(token)]
    if count == 1 and torch.cuda.device_count() == 1:
        return handles[0]
    raise RuntimeError("Cannot match CUDA GPU to NVML GPU; set CUDA_VISIBLE_DEVICES to a GPU UUID")


def measure_one(torch, model, pynvml, handle, s: int, b: int, args) -> dict:
    device = torch.device(f"cuda:{args.device}")
    x = None
    sampler = None
    oom = False
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before = gpu_state(pynvml, handle, "before")
    free_memory, _ = torch.cuda.mem_get_info(device)
    try:
        x = torch.randn((b, 3, s, s), device=device, dtype=torch.float32)
        torch.cuda.synchronize(device)
        with torch.inference_mode():
            warmup_started = time.perf_counter()
            warmup_repeats = 0
            while warmup_repeats < args.warmup or (
                time.perf_counter() - warmup_started < args.warmup_seconds
                and warmup_repeats < args.max_warmup
            ):
                y = model(x)
                torch.cuda.synchronize(device)
                del y
                warmup_repeats += 1
            warmup_window = time.perf_counter() - warmup_started

            # Measure a single isolated forward after warmup. Repeated forwards
            # below are for latency and power, not for the memory definition.
            torch.cuda.reset_peak_memory_stats(device)
            y = model(x)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            del y

            # Time forwards without the NVML polling thread. Synchronization
            # makes this an end-to-end wall-clock measurement per forward.
            latency_times = []
            latency_started = time.perf_counter()
            while len(latency_times) < args.min_repeats or (
                time.perf_counter() - latency_started < args.latency_seconds
                and len(latency_times) < args.max_repeats
            ):
                t0 = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize(device)
                latency_times.append(time.perf_counter() - t0)
                del y
            latency_ended = time.perf_counter()

            # Energy is measured in a separate burst. The repeated calls use
            # the same synchronized-forward protocol as the latency phase.
            sampler = PowerSampler(pynvml, handle)
            sampler.start()
            try:
                counter_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                counter_start_mj = None
            energy_repeats = 0
            start = time.perf_counter()
            while energy_repeats < args.min_repeats or (
                time.perf_counter() - start < args.target_seconds
                and energy_repeats < args.max_repeats
            ):
                y = model(x)
                torch.cuda.synchronize(device)
                del y
                energy_repeats += 1
            end = time.perf_counter()
            try:
                counter_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                counter_end_mj = None
            sampler.stop()
        latency = statistics.median(latency_times)
        mad = statistics.median(abs(t - latency) for t in latency_times)
        distinct_power = len({round(p, 1) for _, p in sampler.samples})
        energy_from_power = sampler.energy_j(start, end) / energy_repeats
        energy_from_counter = (
            (counter_end_mj - counter_start_mj) / 1000.0 / energy_repeats
            if counter_start_mj is not None and counter_end_mj is not None
            and counter_end_mj > counter_start_mj else None
        )
        energy = energy_from_counter if energy_from_counter is not None else energy_from_power
        energy_method = "nvml_counter" if energy_from_counter is not None else "nvml_power_integral"
        warnings = []
        if warmup_window < args.warmup_seconds:
            warnings.append("warmup_short")
        if latency_ended - latency_started < args.latency_seconds:
            warnings.append("latency_window_short")
        if end - start < args.target_seconds:
            warnings.append("power_window_short")
        if mad / max(latency, 1e-12) > 0.1:
            warnings.append("latency_unstable")
        if energy_from_counter is not None and abs(energy_from_counter - energy_from_power) / energy_from_counter > 0.2:
            warnings.append("energy_estimators_disagree")
        return {
            "latency_s": latency,
            "peak_memory_bytes": peak,
            "energy_j": energy,
            "status": "ok",
            "energy_method": energy_method,
            "energy_j_from_counter": energy_from_counter if energy_from_counter is not None else "",
            "energy_j_from_power": energy_from_power,
            "repeats": energy_repeats,
            "latency_repeats": len(latency_times),
            "latency_window_s": latency_ended - latency_started,
            "warmup_repeats": warmup_repeats,
            "warmup_window_s": warmup_window,
            "oom_retries": 0,
            "latency_mad_s": mad,
            "power_samples": len(sampler.samples),
            "distinct_power_readings": distinct_power,
            "power_window_s": end - start,
            "free_memory_bytes_before": free_memory,
            "measurement_warning": ";".join(warnings),
            "trial_count": 1,
            "latency_trial_mad_s": "",
            "energy_trial_mad_j": "",
            "peak_memory_range_bytes": "",
            **before,
            **gpu_state(pynvml, handle, "after"),
        }
    except torch.cuda.OutOfMemoryError:
        oom = True
        if sampler is not None and sampler.thread is not None and sampler.thread.is_alive():
            sampler.stop()
        return {
            "latency_s": "", "peak_memory_bytes": "", "energy_j": "",
            "status": "OOM", "repeats": 0, "warmup_repeats": "", "warmup_window_s": "",
            "latency_repeats": 0, "latency_window_s": "",
            "energy_method": "", "energy_j_from_counter": "", "energy_j_from_power": "",
            "oom_retries": 0,
            "latency_mad_s": "", "power_samples": 0,
            "distinct_power_readings": "", "power_window_s": "",
            "free_memory_bytes_before": free_memory,
            "measurement_warning": "OOM",
            "trial_count": 1,
            "latency_trial_mad_s": "",
            "energy_trial_mad_j": "",
            "peak_memory_range_bytes": "",
            **before,
            **gpu_state(pynvml, handle, "after"),
        }
    finally:
        del x
        if oom:
            gc.collect()
            torch.cuda.empty_cache()


def profile_kernels(torch, model, s: int, b: int, device: int) -> list[dict]:
    """Profile each layer separately so CUDA kernel names retain layer labels."""
    from torch.profiler import ProfilerActivity, profile

    rows = []
    x = torch.randn((b, 3, s, s), device=f"cuda:{device}", dtype=torch.float32)
    with torch.inference_mode():
        for layer_name, layer in model.named_children():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                y = layer(x)
                torch.cuda.synchronize(device)
            for event in prof.events():
                if "CUDA" in str(event.device_type).upper():
                    rows.append({"S": s, "B": b, "layer": layer_name, "kernel_name": event.name})
            x = y
    return rows


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def check_csv_header(path: Path, expected: tuple[str, ...]) -> None:
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as file:
            actual = tuple(next(csv.reader(file)))
        if actual != expected:
            raise RuntimeError(f"{path} has different columns; choose another --output-dir")


def aggregate_trials(trials: list[dict]) -> dict:
    if not trials:
        raise ValueError("No trials to aggregate")
    success = [r for r in trials if r["status"] == "ok"]
    all_warnings = {part for r in trials for part in str(r["measurement_warning"]).split(";") if part}
    if not success:
        row = dict(trials[-1])
        row.pop("trial_index", None)
        row["trial_count"] = len(trials)
        row["oom_retries"] = sum(int(r["oom_retries"]) for r in trials)
        row["measurement_warning"] = ";".join(sorted(all_warnings))
        return row

    row = dict(success[0])
    row.pop("trial_index", None)
    optional_numeric = (
        "latency_s", "peak_memory_bytes", "energy_j", "energy_j_from_counter",
        "energy_j_from_power", "latency_mad_s", "warmup_window_s",
        "latency_window_s", "power_window_s", "free_memory_bytes_before",
        "gpu_temp_c_before", "gpu_temp_c_after", "sm_clock_mhz_before",
        "sm_clock_mhz_after", "power_w_before", "power_w_after",
    )
    for field in optional_numeric:
        values = [float(r[field]) for r in success if r[field] != "" and r[field] is not None]
        row[field] = statistics.median(values) if values else ""
    row["peak_memory_bytes"] = int(round(row["peak_memory_bytes"]))
    row["free_memory_bytes_before"] = int(round(row["free_memory_bytes_before"]))
    for field in ("repeats", "latency_repeats", "warmup_repeats", "power_samples", "oom_retries"):
        row[field] = sum(int(r[field]) for r in trials if r[field] != "")
    row["distinct_power_readings"] = min(int(r["distinct_power_readings"]) for r in success)
    row["trial_count"] = len(trials)
    row["energy_method"] = (
        success[0]["energy_method"]
        if all(r["energy_method"] == success[0]["energy_method"] for r in success)
        else "mixed"
    )
    if row["energy_method"] == "mixed":
        all_warnings.add("mixed_energy_methods")
    latencies = [float(r["latency_s"]) for r in success]
    energies = [float(r["energy_j"]) for r in success]
    peaks = [int(r["peak_memory_bytes"]) for r in success]
    row["latency_trial_mad_s"] = statistics.median(abs(x - row["latency_s"]) for x in latencies)
    row["energy_trial_mad_j"] = statistics.median(abs(x - row["energy_j"]) for x in energies)
    row["peak_memory_range_bytes"] = max(peaks) - min(peaks)
    if len(success) != len(trials):
        row["status"] = "unstable_oom"
        all_warnings.add("inconsistent_oom")
    if row["latency_trial_mad_s"] / max(row["latency_s"], 1e-12) > 0.05:
        all_warnings.add("latency_between_trials_unstable")
    if row["energy_trial_mad_j"] / max(row["energy_j"], 1e-12) > 0.10:
        all_warnings.add("energy_between_trials_unstable")
    if row["peak_memory_range_bytes"] != 0:
        all_warnings.add("memory_peak_changed")
    row["measurement_warning"] = ";".join(sorted(all_warnings))
    return row


def measure_grid(args, out: Path) -> None:
    ensure_package("pynvml", "nvidia-ml-py")
    import pynvml
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: enable GPU in Kaggle notebook settings")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} unavailable")
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    gpu_name = torch.cuda.get_device_name(args.device)
    if "T4" not in gpu_name:
        print(f"Warning: selected GPU is {gpu_name}, not a T4", flush=True)

    sizes, batches = make_grid(args.seed)
    configurations = [(s, b) for s in sizes for b in batches]
    random.Random(args.seed + 1).shuffle(configurations)
    metadata = {
        "protocol_version": 11,
        "seed": args.seed, "sizes": sizes, "batches": batches,
        "order_seed": args.seed + 1,
        "trial_input_seed_rule": "(seed*1000003 + S*1009 + B*17 + trial_index) modulo 2^32",
        "gpu": gpu_name, "gpu_index": args.device,
        "cuda_gpu_uuid": str(getattr(torch.cuda.get_device_properties(args.device), "uuid", "")),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(), "platform": platform.platform(),
        "cudnn_benchmark": False, "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False, "dtype": "float32",
        "warmup_min_repeats": args.warmup,
        "warmup_seconds": args.warmup_seconds,
        "max_warmup": args.max_warmup,
        "trials": args.trials,
        "min_repeats": args.min_repeats,
        "latency_seconds": args.latency_seconds,
        "target_seconds": args.target_seconds,
        "max_repeats": args.max_repeats,
        "energy_method": "NVML whole-GPU energy counter if supported, otherwise power integral; divided by repeated synchronized forwards; no idle subtraction",
        "latency_method": "median perf_counter wall time for one forward plus CUDA synchronization",
        "memory_method": "torch.cuda.max_memory_allocated for one forward after warmup; model and input resident",
        "allocator_clean_between_trials": True,
    }
    meta_path = out / "metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if any(old.get(k) != value for k, value in metadata.items()):
            raise RuntimeError("Existing output uses a different measurement protocol; choose another --output-dir")
    else:
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        metadata["nvidia_driver"] = str(driver)
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            if saved.get("nvidia_driver", metadata["nvidia_driver"]) != metadata["nvidia_driver"]:
                raise RuntimeError("NVIDIA driver changed since the previous measurement; choose another --output-dir")
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        handle = select_nvml_handle(pynvml, torch, args.device)
        pynvml.nvmlDeviceGetPowerUsage(handle)  # fail early if power is unavailable
        model = build_model(torch).to(f"cuda:{args.device}").eval()
        with torch.inference_mode():
            smoke = torch.randn((1, 3, 32, 32), device=f"cuda:{args.device}", dtype=torch.float32)
            smoke_output = model(smoke)
            torch.cuda.synchronize(args.device)
            if tuple(smoke_output.shape) != (1, 100) or smoke_output.dtype != torch.float32:
                raise RuntimeError("Model output must have shape (1, 100) and dtype float32")
            del smoke_output, smoke
        csv_path = out / "measurements.csv"
        check_csv_header(csv_path, FIELDS)
        previous = load_rows(csv_path) if csv_path.exists() else []
        if len(previous) != len({(int(r["S"]), int(r["B"])) for r in previous}):
            raise RuntimeError("Duplicate configuration in measurements.csv")
        if any((int(r["S"]), int(r["B"])) not in configurations for r in previous):
            raise RuntimeError("measurements.csv contains a point outside this grid")
        done = {(int(r["S"]), int(r["B"])) for r in previous}
        trial_path = out / "trials.csv"
        check_csv_header(trial_path, TRIAL_FIELDS)
        prior_trials = load_rows(trial_path) if trial_path.exists() else []
        trial_map = {}
        for r in prior_trials:
            key = (int(r["S"]), int(r["B"]), int(r["trial_index"]))
            if (key[0], key[1]) not in configurations or not 0 <= key[2] < args.trials:
                raise RuntimeError(f"Trial outside this grid: {key}")
            if key in trial_map:
                raise RuntimeError(f"Duplicate trial in trials.csv: {key}")
            trial_map[key] = r
        for s, b in done:
            if any((s, b, j) not in trial_map for j in range(args.trials)):
                raise RuntimeError(f"Aggregated row S={s}, B={b} has missing raw trials")

        with csv_path.open("a", newline="") as file, trial_path.open("a", newline="") as trial_file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            trial_writer = csv.DictWriter(trial_file, fieldnames=TRIAL_FIELDS)
            if file.tell() == 0:
                writer.writeheader()
            if trial_file.tell() == 0:
                trial_writer.writeheader()
            for s, b in configurations:
                if (s, b) in done:
                    continue
                for j in range(args.trials):
                    key = (s, b, j)
                    if key in trial_map:
                        continue
                    trial_seed = (args.seed * 1_000_003 + s * 1009 + b * 17 + j) % (2**32)
                    torch.manual_seed(trial_seed)
                    result = measure_one(torch, model, pynvml, handle, s, b, args)
                    if result["status"] == "OOM":
                        torch.manual_seed(trial_seed)
                        result = measure_one(torch, model, pynvml, handle, s, b, args)
                        result["oom_retries"] = 1
                        if result["status"] == "ok":
                            result["measurement_warning"] = ";".join(
                                x for x in (result["measurement_warning"], "recovered_after_oom") if x
                            )
                    trial_row = {
                        "trial_index": j, "S": s, "B": b,
                        "is_validation": int(s not in BASE_SIZES or b not in BASE_BATCHES),
                        **result,
                    }
                    trial_writer.writerow(trial_row)
                    trial_file.flush()
                    trial_map[key] = trial_row
                    print(f"S={s:3d} B={b:3d} trial={j + 1}/{args.trials} "
                          f"{result['status']:>3s} latency={result['latency_s']} "
                          f"energy={result['energy_j']} warning={result['measurement_warning']}", flush=True)
                row = aggregate_trials([trial_map[(s, b, j)] for j in range(args.trials)])
                writer.writerow(row)
                file.flush()  # retain partial progress if Kaggle stops the job
                print(f"S={s:3d} B={b:3d} summary={row['status']} "
                      f"latency={row['latency_s']} energy={row['energy_j']}", flush=True)

        rows = load_rows(csv_path)
        if len(rows) != len(configurations):
            raise RuntimeError(f"Incomplete grid: {len(rows)} of {len(configurations)} configurations")
        diagnostic = {
            "total": len(rows),
            "raw_trials": len(trial_map),
            "expected_raw_trials": len(configurations) * args.trials,
            "ok": sum(r["status"] == "ok" for r in rows),
            "oom": sum(r["status"] == "OOM" for r in rows),
            "unstable_oom": sum(r["status"] == "unstable_oom" for r in rows),
            "recovered_after_oom": sum("recovered_after_oom" in r["measurement_warning"] for r in rows),
            "short_power_windows": sum("power_window_short" in r["measurement_warning"] for r in rows),
            "short_warmups": sum("warmup_short" in r["measurement_warning"] for r in rows),
            "unstable_latency": sum("latency_unstable" in r["measurement_warning"] for r in rows),
            "unstable_between_trials_latency": sum("latency_between_trials_unstable" in r["measurement_warning"] for r in rows),
            "unstable_between_trials_energy": sum("energy_between_trials_unstable" in r["measurement_warning"] for r in rows),
            "memory_peak_changed": sum("memory_peak_changed" in r["measurement_warning"] for r in rows),
            "short_latency_windows": sum("latency_window_short" in r["measurement_warning"] for r in rows),
            "few_distinct_power_readings": sum(r["status"] == "ok" and r["distinct_power_readings"] != "" and int(r["distinct_power_readings"]) < 3 for r in rows),
            "energy_estimators_disagree": sum("energy_estimators_disagree" in r["measurement_warning"] for r in rows),
        }
        (out / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(f"Diagnostics: {diagnostic}", flush=True)

        if not args.no_kernels:
            ok = {(int(r["S"]), int(r["B"])) for r in rows if r["status"] == "ok"}
            choices = ([(32, 1), (224, 16), (512, 16)]
                       if args.representative_kernels else configurations)
            kernel_path = out / "kernels.csv"
            kernel_dir = out / "kernel_profiles"
            kernel_dir.mkdir(exist_ok=True)
            kernel_fields = ("S", "B", "layer", "kernel_name")
            def profile_done(path: Path) -> bool:
                return path.exists() and bool(load_rows(path))

            for s, b in choices:
                if (s, b) not in ok:
                    continue
                point_path = kernel_dir / f"S{s}_B{b}.csv"
                if profile_done(point_path):
                    continue
                try:
                    items = profile_kernels(torch, model, s, b, args.device)
                    if not items:
                        raise RuntimeError("Profiler returned no CUDA kernel events")
                    temp_path = point_path.with_suffix(".tmp")
                    with temp_path.open("w", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=kernel_fields)
                        writer.writeheader()
                        writer.writerows(items)
                    os.replace(temp_path, point_path)
                    print(f"Kernels profiled: S={s}, B={b}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"Kernel profiling OOM: S={s}, B={b}", flush=True)
                except (RuntimeError, OSError) as exc:
                    print(f"Kernel profiling unavailable at S={s}, B={b}: {exc}", flush=True)
            temp_path = kernel_path.with_suffix(".tmp")
            with temp_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=kernel_fields)
                writer.writeheader()
                for s, b in choices:
                    point_path = kernel_dir / f"S{s}_B{b}.csv"
                    if (s, b) in ok and profile_done(point_path):
                        writer.writerows(load_rows(point_path))
            os.replace(temp_path, kernel_path)
            diagnostic["kernel_profiles_missing"] = sum(
                (s, b) in ok and not profile_done(kernel_dir / f"S{s}_B{b}.csv")
                for s, b in choices
            )
            (out / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    finally:
        pynvml.nvmlShutdown()


def fit_pysr(args, out: Path) -> None:
    import numpy as np

    path = out / "measurements.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run measurements first")
    all_rows = load_rows(path)
    metadata_path = out / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {(s, b) for s in metadata["sizes"] for b in metadata["batches"]}
    observed = {(int(r["S"]), int(r["B"])) for r in all_rows}
    if observed != expected or len(all_rows) != len(expected):
        raise RuntimeError("Measurement grid is incomplete or duplicated; finish measuring before PySR")
    rows = [r for r in all_rows if r["status"] == "ok"]
    if len(rows) < 10:
        raise RuntimeError("Too few successful measurements to fit PySR")

    # Delayed install leaves measurements usable if Julia setup fails.
    ensure_package("pysr", "pysr")
    from pysr import PySRRegressor

    X = np.array([[float(r["S"]), float(r["B"])] for r in rows], dtype=np.float64)
    train = np.array([r["is_validation"] == "0" for r in rows])
    valid = ~train
    if train.sum() < 10 or valid.sum() < 1:
        raise RuntimeError("Need at least 10 train points and one held-out point")

    fit_dir = out / "pysr"
    fig_dir = out / "figures"
    fit_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    for field, scale, unit in (
            ("latency_s", 1000.0, "ms"),
            ("peak_memory_bytes", 1 / (1024 * 1024), "MiB"),
            ("energy_j", 1000.0, "mJ"),
        ):
            available = np.array([r[field] != "" and "warmup_short" not in r["measurement_warning"] for r in rows])
            if field == "latency_s":
                available &= np.array([
                    "latency_unstable" not in r["measurement_warning"]
                    and "latency_window_short" not in r["measurement_warning"]
                    and "latency_between_trials_unstable" not in r["measurement_warning"]
                    for r in rows
                ])
            elif field == "energy_j":
                available &= np.array([
                    "power_window_short" not in r["measurement_warning"]
                    and "energy_between_trials_unstable" not in r["measurement_warning"]
                    and "mixed_energy_methods" not in r["measurement_warning"]
                    for r in rows
                ])
            fit_mask = train & available
            val_mask = valid & available
            if fit_mask.sum() < 10 or val_mask.sum() < 1:
                raise RuntimeError(
                    f"Cannot fit {field}: only {fit_mask.sum()} clean train and "
                    f"{val_mask.sum()} clean validation points"
                )
            y = np.array([float(r[field]) * scale if r[field] else np.nan for r in rows])
            print(f"PySR fitting {field}: {fit_mask.sum()} train, {val_mask.sum()} validation", flush=True)
            model = PySRRegressor(
                niterations=args.iterations,
                populations=4,
                population_size=30,
                ncycles_per_iteration=100,
                maxsize=18,
                binary_operators=["+", "-", "*", "/"],
                model_selection="best",
                verbosity=0,
                progress=False,
                random_state=args.seed,
                deterministic=True,
                parallelism="serial",
                output_directory=str(fit_dir),
            )
            # `S` is reserved by SymPy, which PySR uses to export equations.
            # Keep the dataset columns S/B, but use safe names in expressions.
            model.fit(X[fit_mask], y[fit_mask], variable_names=["image_size", "batch_size"])
            name = field.removesuffix("_s").removesuffix("_bytes").removesuffix("_j")
            model.equations_.to_csv(fit_dir / f"{name}_equations.csv", index=False)
            pred_train = np.asarray(model.predict(X[fit_mask]), dtype=float).reshape(-1)
            pred_valid = np.asarray(model.predict(X[val_mask]), dtype=float).reshape(-1)

            def metrics(actual, pred):
                finite = np.isfinite(pred)
                if not finite.any():
                    return {"mae": None, "median_relative_error": None, "invalid_predictions": int(len(pred))}
                return {
                    "mae": float(np.mean(np.abs(actual[finite] - pred[finite]))),
                    "median_relative_error": float(np.median(np.abs(actual[finite] - pred[finite]) / np.maximum(np.abs(actual[finite]), 1e-12))),
                    "invalid_predictions": int((~finite).sum()),
                }

            summary = {
                "target": field, "unit": unit,
                "features": ["image_size", "batch_size"],
                "feature_columns": ["S", "B"],
                "equation": str(model.sympy()), "latex": str(model.latex()),
                "train_count": int(fit_mask.sum()), "validation_count": int(val_mask.sum()),
                "train": metrics(y[fit_mask], pred_train),
                "validation": metrics(y[val_mask], pred_valid),
            }
            (fit_dir / f"{name}_selected.json").write_text(json.dumps(summary, indent=2) + "\n")

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 5))
            train_finite = np.isfinite(pred_train)
            valid_finite = np.isfinite(pred_valid)
            ax.scatter(y[fit_mask][train_finite], pred_train[train_finite], s=20, alpha=0.65, label="fit")
            ax.scatter(y[val_mask][valid_finite], pred_valid[valid_finite], s=28, alpha=0.85, label="held out")
            plotted = np.r_[y[fit_mask], y[val_mask], pred_train[train_finite], pred_valid[valid_finite]]
            lo = float(np.min(plotted))
            hi = float(np.max(plotted))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
            ax.set(xlabel=f"Measured {field} [{unit}]", ylabel=f"PySR prediction [{unit}]")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"pysr_{name}.png", dpi=170)
            plt.close(fig)
            print(f"{field}: {summary['equation']}; validation {summary['validation']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    default_out = Path("/kaggle/working/hw1/results") if Path("/kaggle/working").exists() else Path("hw1/results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=20, help="Minimum warmup forwards per trial")
    parser.add_argument("--warmup-seconds", type=float, default=0.5)
    parser.add_argument("--max-warmup", type=int, default=100000)
    parser.add_argument("--trials", type=int, default=3, help="Independent measurements per configuration")
    parser.add_argument("--min-repeats", type=int, default=20, help="Minimum timed forwards per trial")
    parser.add_argument("--latency-seconds", type=float, default=0.5)
    parser.add_argument("--target-seconds", type=float, default=3.0, help="Minimum energy measurement window per trial")
    parser.add_argument("--max-repeats", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=40, help="PySR iterations for each of 3 targets")
    parser.add_argument("--fit-only", action="store_true", help="Fit existing measurements without GPU")
    parser.add_argument("--measure-only", action="store_true", help="Skip PySR installation and fitting")
    parser.add_argument("--no-kernels", action="store_true", help="Skip kernel profiling")
    parser.add_argument("--representative-kernels", action="store_true", help="Profile only three representative configurations")
    if argv is None and Path(sys.argv[0]).name in {
        "colab_kernel_launcher.py", "ipykernel_launcher.py",
    }:
        # Code pasted into a Colab/Jupyter cell inherits the kernel's -f and
        # HistoryManager arguments. They are not arguments for this script.
        argv = []
    args = parser.parse_args(argv)
    if args.fit_only and args.measure_only:
        parser.error("--fit-only and --measure-only are mutually exclusive")
    if (args.warmup < 1 or args.max_warmup < args.warmup or args.trials < 1
            or args.min_repeats < 1 or args.max_repeats < args.min_repeats
            or args.warmup_seconds < 0 or args.latency_seconds <= 0
            or args.target_seconds <= 0 or args.iterations < 1):
        parser.error("invalid warmup, repeat, target time, or iteration settings")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.fit_only:
        measure_grid(args, out)
    if not args.measure_only:
        if args.fit_only:
            fit_pysr(args, out)
        else:
            # A clean process avoids Julia loading into the already imported
            # PyTorch runtime when this code is run from a .py file.
            script_path = globals().get("__file__")
            if script_path and Path(script_path).name == "kaggle_measure.py":
                subprocess.run(
                    [sys.executable, str(Path(script_path).resolve()), "--fit-only",
                     "--output-dir", str(out), "--seed", str(args.seed),
                     "--iterations", str(args.iterations)],
                    check=True,
                )
            else:
                # A pasted notebook cell has no importable script path.
                fit_pysr(args, out)
    print(f"Done. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
