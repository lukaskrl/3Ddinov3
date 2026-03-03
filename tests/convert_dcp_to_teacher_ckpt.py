"""Convert a DCP checkpoint directory to a consolidated teacher checkpoint (.pth).

Usage (single GPU):
  python tests/convert_dcp_to_teacher_ckpt.py \
    --config-file dinov3/configs/ssl_mri3d_stage1.yaml \
    --dcp-dir /path/to/ckpt/107799 \
    --output /path/to/teacher_checkpoint.pth
"""

import os, sys
from pathlib import Path
# append parent directory so we access dinov3
print(f"Adding parent directory to sys.path: {Path(__file__).parent.parent}")
sys.path.append(str(Path(__file__).parent.parent))


import torch
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import load_checkpoint
from dinov3.configs import DinoV3SetupArgs, setup_config
from dinov3.train.ssl_meta_arch import SSLMetaArch

# Hardcoded DCP checkpoint directory (example)
# example path /home/lukas/3Ddinov3/work_dir/mri_full_training_centering/ckpt/90999
DCP_DIR = Path("/home/lukas/3Ddinov3/work_dir/big_batch/ckpt/23399")

def main() -> None:
    # Derive config and output paths from the hardcoded checkpoint dir
    dcp_dir = DCP_DIR
    base_dir = dcp_dir.parents[1]
    config_file = base_dir / "config.yaml"
    ckpt_name = dcp_dir.name
    output_path = base_dir / f"checkpoint_{ckpt_name}.pth"

    if not distributed.is_enabled():
        distributed.enable()

    setup_args = DinoV3SetupArgs(
        config_file=str(config_file),
        pretrained_weights="",
        shard_unsharded_model=False,
        output_dir="",
        opts=[],
    )
    cfg = setup_config(setup_args, strict_cfg=False)

    with torch.device("meta"):
        model = SSLMetaArch(cfg)
    model.prepare_for_distributed_training()

    process_group = distributed.get_process_subgroup()
    load_checkpoint(
        ckpt_dir=str(dcp_dir),
        model=model,
        strict_loading=True,
        process_group=process_group,
    )

    if not distributed.is_main_process():
        return

    state_dict = model.model_ema.state_dict()
    for k, tensor in list(state_dict.items()):
        if isinstance(tensor, DTensor):
            state_dict[k] = tensor.full_tensor()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": state_dict}, str(output_path))
    print(f"Saved consolidated teacher checkpoint: {output_path}")


if __name__ == "__main__":
    main()
