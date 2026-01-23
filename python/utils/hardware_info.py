#!/usr/bin/env python3
"""
ハードウェア情報取得ユーティリティ

PyTorchのデバイス情報と詳細なハードウェア情報を取得する。
設計書の要件に基づき、GPU名、VRAM量、CPU名、メモリ量を含む。
"""

import json
import platform
import re
import subprocess
import torch
from typing import Dict, List, Optional


def _run_command(cmd: str) -> str:
    """安全にコマンドを実行して結果を取得"""
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return ""


def get_cpu_info() -> Dict[str, str]:
    """CPU情報を取得"""
    info = {"name": "Unknown CPU", "cores": "Unknown"}
    
    # CPU名の取得
    name = ""
    if hasattr(platform, "processor") and platform.processor():
        name = platform.processor()
    
    if not name:
        if platform.system() == "Windows":
            name = _run_command('wmic cpu get name /value').split("=")[-1].strip()
        elif platform.system() == "Darwin":  # macOS
            name = _run_command("sysctl -n machdep.cpu.brand_string")
        else:  # Linux
            cpu_output = _run_command("grep 'model name' /proc/cpuinfo | head -1")
            if cpu_output:
                name = cpu_output.split(":", 1)[-1].strip()
    
    info["name"] = name or "Unknown CPU"
    
    # CPUコア数の取得
    try:
        import psutil
        info["cores"] = str(psutil.cpu_count(logical=False))
        info["threads"] = str(psutil.cpu_count(logical=True))
    except ImportError:
        # psutilがない場合はplatformで取得
        try:
            import os
            info["cores"] = str(os.cpu_count() or "Unknown")
        except:
            info["cores"] = "Unknown"
    
    return info


def get_memory_info() -> Dict[str, str]:
    """メモリ情報を取得"""
    info = {"total": "Unknown", "available": "Unknown"}
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["total"] = f"{memory.total // (1024**3)}GB"
        info["available"] = f"{memory.available // (1024**3)}GB"
    except ImportError:
        # psutilがない場合はOSコマンドで取得
        if platform.system() == "Darwin":  # macOS
            total_bytes = _run_command("sysctl -n hw.memsize")
            if total_bytes:
                try:
                    info["total"] = f"{int(total_bytes) // (1024**3)}GB"
                except:
                    pass
        elif platform.system() == "Linux":
            meminfo = _run_command("grep MemTotal /proc/meminfo")
            if meminfo:
                match = re.search(r"(\d+)", meminfo)
                if match:
                    # /proc/meminfoはkB単位
                    info["total"] = f"{int(match.group(1)) // (1024**2)}GB"
    
    return info


def get_pytorch_device_info() -> Dict[str, any]:
    """PyTorchのデバイス情報を取得"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "device_count": 0,
        "current_device": "cpu",
        "devices": []
    }
    
    # 現在のデバイス決定
    if device_info["mps_available"]:
        device_info["current_device"] = "mps"
    elif device_info["cuda_available"]:
        device_info["current_device"] = "cuda"
    
    # CUDAデバイス情報
    if device_info["cuda_available"]:
        device_info["device_count"] = torch.cuda.device_count()
        for i in range(device_info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_info["devices"].append({
                "index": i,
                "name": props.name,
                "memory_total": f"{props.total_memory // (1024**3)}GB",
                "compute_capability": f"{props.major}.{props.minor}",
                "type": "cuda"
            })
    
    # MPS情報（Apple Silicon）
    if device_info["mps_available"]:
        # MPSのメモリ情報は直接取得が困難なので、system_profilerを使用
        memory_info = "Unknown"
        if platform.system() == "Darwin":
            try:
                # Apple Siliconのメモリ情報を取得
                memory_output = _run_command("system_profiler SPHardwareDataType | grep 'Memory:'")
                if memory_output:
                    # "Memory: 16 GB" のような形式から抽出
                    match = re.search(r"Memory:\s*(\d+\s*GB)", memory_output)
                    if match:
                        memory_info = match.group(1)
            except:
                pass
        
        device_info["devices"].append({
            "index": 0,
            "name": "Apple MPS",
            "memory_total": memory_info,
            "type": "mps"
        })
        device_info["device_count"] = 1
    
    return device_info


def get_gpu_info() -> List[Dict[str, str]]:
    """GPU情報を詳細に取得"""
    gpus = []
    
    # NVIDIA GPU
    nvidia_output = _run_command('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits')
    if nvidia_output:
        for line in nvidia_output.splitlines():
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpus.append({
                        "name": parts[0].strip(),
                        "memory": f"{parts[1].strip()}MB",
                        "vendor": "NVIDIA"
                    })
    
    # macOS (Apple GPU)
    if platform.system() == "Darwin" and not gpus:
        sp_output = _run_command("system_profiler SPDisplaysDataType | grep 'Chipset Model'")
        if sp_output:
            for line in sp_output.splitlines():
                if "Chipset Model" in line:
                    gpu_name = line.split(":")[-1].strip()
                    gpus.append({
                        "name": gpu_name,
                        "memory": "Shared",
                        "vendor": "Apple"
                    })
    
    # AMD GPU (ROCm)
    amd_output = _run_command('rocm-smi --showproductname --showmeminfo vram --csv')
    if amd_output and "Card series" in amd_output:
        # AMD GPUの詳細処理は必要に応じて実装
        pass
    
    return gpus or [{"name": "No discrete GPU detected", "memory": "N/A", "vendor": "Unknown"}]


def get_comprehensive_hardware_info() -> str:
    """包括的なハードウェア情報を文字列で取得"""
    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    pytorch_info = get_pytorch_device_info()
    gpu_info = get_gpu_info()
    
    # メイン情報の構築
    parts = []
    
    # デバイス情報
    if pytorch_info["current_device"] == "mps":
        parts.append("MPS")
    elif pytorch_info["current_device"] == "cuda":
        parts.append("CUDA")
    else:
        parts.append("CPU")
    
    # GPU情報
    if pytorch_info["devices"]:
        for device in pytorch_info["devices"]:
            gpu_part = device["name"]
            if device["memory_total"] != "Unknown":
                gpu_part += f" ({device['memory_total']})"
            parts.append(gpu_part)
    
    # CPU情報
    cpu_part = cpu_info["name"]
    if cpu_info["cores"] != "Unknown":
        cpu_part += f" ({cpu_info['cores']} cores)"
    parts.append(cpu_part)
    
    # メモリ情報
    if memory_info["total"] != "Unknown":
        parts.append(f"RAM: {memory_info['total']}")
    
    return " | ".join(parts)


def get_detailed_hardware_info() -> Dict[str, any]:
    """詳細なハードウェア情報を辞書で取得"""
    return {
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "pytorch": get_pytorch_device_info(),
        "gpu": get_gpu_info(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__
        }
    }


def get_hardware_info() -> str:
    """
    シンプルなハードウェア情報文字列を取得
    
    設計書の要件に基づき、GPU名、VRAM量、CPU名、メモリ量を含む。
    既存コードとの互換性のため、この関数名を維持。
    
    Returns:
        str: ハードウェア情報の文字列
    """
    return get_comprehensive_hardware_info()


if __name__ == "__main__":
    # 詳細情報の表示
    detailed_info = get_detailed_hardware_info()
    print("=== Detailed Hardware Information ===")
    print(json.dumps(detailed_info, ensure_ascii=False, indent=2))
    
    print("\n=== Summary ===")
    print(get_hardware_info())