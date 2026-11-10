"""
Hardware Monitoring Utility for Raspberry Pi 5.
Reads CPU utilization, RAM usage, CPU temperature via sysfs/vcgencmd,
and logs empirical telemetry during computer vision inference.
"""

import os
import time
import psutil
from typing import Dict, Any, Optional


class HardwareMonitor:
    """Monitors system resource utilization on Raspberry Pi 5."""

    def __init__(self):
        self.process = psutil.Process()
        # Prime psutil CPU calculation
        psutil.cpu_percent(interval=None)

    @staticmethod
    def get_cpu_temperature() -> float:
        """
        Reads SoC temperature in degrees Celsius from sysfs thermal zone.
        Falls back to vcgencmd or psutil sensors if needed.
        """
        # Primary method: Linux sysfs
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            try:
                with open(thermal_path, "r") as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
            except Exception:
                pass

        # Secondary method: psutil thermal sensors
        try:
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps and temps["cpu_thermal"]:
                return float(temps["cpu_thermal"][0].current)
        except Exception:
            pass

        return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Captures instantaneous hardware metrics.
        Returns:
            Dict containing cpu_percent, ram_used_mb, ram_percent,
            temperature_c, and process_memory_mb.
        """
        vm = psutil.virtual_memory()
        proc_mem = self.process.memory_info().rss / (1024 * 1024)

        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": vm.used / (1024 * 1024),
            "ram_total_mb": vm.total / (1024 * 1024),
            "ram_percent": vm.percent,
            "process_ram_mb": proc_mem,
            "cpu_temperature_c": self.get_cpu_temperature(),
        }
