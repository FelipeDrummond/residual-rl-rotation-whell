"""
Quick script to check which device will be used for training.

Usage:
    python -m simulation.check_device
"""

import torch


def main():
    print("=" * 60)
    print("PyTorch Device Detection")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")

    # Check MPS
    mps_available = torch.backends.mps.is_available()
    print(f"\nMPS (Metal) available: {mps_available}")
    if mps_available:
        print("  Running on Apple Silicon with Metal acceleration")

    # Determine best device
    if cuda_available:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif mps_available:
        device = "mps"
        device_name = "Apple Silicon GPU (Metal)"
    else:
        device = "cpu"
        device_name = "CPU"

    print("\n" + "=" * 60)
    print(f"Selected device: {device}")
    print(f"Device name: {device_name}")
    print("=" * 60)

    # Quick performance test
    print("\nRunning quick performance test...")
    size = 1000
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    import time
    start = time.time()
    for _ in range(10):
        z = torch.mm(x, y)
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"10 matrix multiplications ({size}x{size}): {elapsed:.3f}s")
    print(f"Performance: {10/elapsed:.1f} ops/sec")

    print("\n" + "=" * 60)
    print("READY FOR TRAINING!")
    print("=" * 60)
    print(f"\nTo train with this device, run:")
    print(f"  python -m simulation.train")
    print(f"\nOr force a specific device:")
    print(f"  python -m simulation.train --device {device}")


if __name__ == "__main__":
    main()
