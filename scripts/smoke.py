"""No CUDA driver required: verify wheel identity and compiled target flags."""
import argparse
import faulthandler
import json
from pathlib import Path


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    print("[smoke] import torch", flush=True)
    import torch
    print("[smoke] import torchvision", flush=True)
    import torchvision
    print("[smoke] import torchaudio", flush=True)
    import torchaudio
    print("[smoke] verify versions and CUDA targets", flush=True)
    assert torch.__version__ == spec["torch"] + "+" + spec["local"], torch.__version__
    assert torchvision.__version__.split("+")[0] == spec["vision"], torchvision.__version__
    assert torchaudio.__version__.split("+")[0] == spec["audio"], torchaudio.__version__
    assert torch.version.cuda, "CPU-only wheel must never be published"
    assert torch.version.cuda.split(".")[:2] == spec["cuda"].split(".")[:2]
    # get_arch_list() returns [] without a driver in some upstream releases.
    flags = torch._C._cuda_getArchFlags()
    for arch in spec["architectures"]:
        assert "sm_" + arch.split("+")[0].replace(".", "") in flags.split(), (arch, flags)
    print("[smoke] torch CPU operations", flush=True)
    assert torch.ones(2).sum().item() == 2
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
    print("[smoke] torchvision CPU NMS", flush=True)
    assert torchvision.ops.nms(boxes, torch.tensor([1., .5]), .5).tolist() == [0]
    # No GPU test is claimed: standard hosted runners have no NVIDIA GPU.
    Path(args.report).write_text(json.dumps({"imports": True, "cpu_ops": True, "compiled_arch_flags": flags, "gpu_execution": "not tested; no GPU on hosted runners"}, indent=2))


if __name__ == "__main__":
    main()
