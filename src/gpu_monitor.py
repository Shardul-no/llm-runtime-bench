"""Background GPU sampler (power, temperature, clocks, VRAM, throttle flags) via NVML."""
import threading
import time

import pynvml

# NVML clock-throttle bit masks (hard-coded to avoid pynvml version differences)
THR_SW_POWER_CAP = 0x04
THR_HW_SLOWDOWN = 0x08
THR_SW_THERMAL = 0x20
THR_HW_THERMAL = 0x40


class GPUMonitor:
    def __init__(self, index=0, interval=0.1):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.interval = interval
        self.samples = []  # (t, power_w, temp_c, sm_mhz, vram_gb, throttle_mask)
        self._stop = threading.Event()
        self._thread = None

    # ---- static info -------------------------------------------------
    def name(self):
        n = pynvml.nvmlDeviceGetName(self.h)
        return n.decode() if isinstance(n, bytes) else n

    def power_limit_w(self):
        # Some laptop drivers return N/A for one call but not another, so try several.
        for fn in ("nvmlDeviceGetPowerManagementLimit", "nvmlDeviceGetEnforcedPowerLimit",
                   "nvmlDeviceGetPowerManagementDefaultLimit"):
            try:
                return getattr(pynvml, fn)(self.h) / 1000.0
            except (pynvml.NVMLError, AttributeError):
                continue
        return float("nan")

    def temp(self):
        return pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)

    # ---- sampling ----------------------------------------------------
    def _read(self):
        t = time.perf_counter()
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        except pynvml.NVMLError:
            power = float("nan")
        temp = self.temp()
        sm = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
        vram = pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 1024 ** 3
        try:
            try:
                thr = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h)
            except AttributeError:
                thr = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)
        except pynvml.NVMLError:
            thr = 0
        return (t, power, temp, sm, vram, thr)

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._read())
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summarize(self, t0, t1):
        """Stats over the window [t0, t1] (perf_counter seconds)."""
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if len(s) < 2:
            return {}
        energy = 0.0
        for a, b in zip(s[:-1], s[1:]):  # trapezoidal integration of power
            energy += 0.5 * (a[1] + b[1]) * (b[0] - a[0])
        thr_all = 0
        for x in s:
            thr_all |= x[5]
        return {
            "energy_j": energy,
            "mean_power_w": sum(x[1] for x in s) / len(s),
            "max_temp_c": max(x[2] for x in s),
            "mean_sm_clock_mhz": sum(x[3] for x in s) / len(s),
            "vram_start_gb": s[0][4],
            "peak_vram_gb": max(x[4] for x in s),
            "thermal_throttle": int(bool(thr_all & (THR_SW_THERMAL | THR_HW_THERMAL | THR_HW_SLOWDOWN))),
            "power_throttle": int(bool(thr_all & THR_SW_POWER_CAP)),
            "n_samples": len(s),
        }

    def close(self):
        self.stop()
        pynvml.nvmlShutdown()
