import argparse
import json

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare model tensors from Lightning or pure PyTorch checkpoints"
    )
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_state(checkpoint, path):
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"{path} has no model or state_dict entry")
    return state


def compare_models(reference, candidate, atol, rtol):
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common_keys = sorted(reference_keys & candidate_keys)
    missing = sorted(reference_keys - candidate_keys)
    unexpected = sorted(candidate_keys - reference_keys)

    mismatched_shapes = []
    failed = []
    largest = {"name": None, "max_abs": 0.0, "max_rel": 0.0}
    for name in common_keys:
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape:
            mismatched_shapes.append(
                {"name": name, "reference": list(left.shape), "candidate": list(right.shape)}
            )
            continue
        if not (torch.is_floating_point(left) or torch.is_complex(left)):
            if not torch.equal(left, right):
                failed.append(name)
            continue

        left = left.float()
        right = right.float()
        absolute = (left - right).abs()
        max_abs = float(absolute.max()) if absolute.numel() else 0.0
        denominator = left.abs().clamp_min(torch.finfo(left.dtype).eps)
        max_rel = float((absolute / denominator).max()) if absolute.numel() else 0.0
        if max_abs > largest["max_abs"]:
            largest = {"name": name, "max_abs": max_abs, "max_rel": max_rel}
        if not torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True):
            failed.append(name)

    return {
        "common_tensor_count": len(common_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched_shapes,
        "failed_tensor_count": len(failed),
        "first_failed_tensors": failed[:20],
        "largest_error": largest,
        "allclose": not missing
        and not unexpected
        and not mismatched_shapes
        and not failed,
    }


def optimizer_summary(checkpoint):
    summaries = []
    for index, optimizer in enumerate(checkpoint.get("optimizer_states", [])):
        groups = optimizer.get("param_groups", [])
        summaries.append(
            {
                "index": index,
                "state_count": len(optimizer.get("state", {})),
                "learning_rates": [group.get("lr") for group in groups],
                "betas": [group.get("betas") for group in groups],
            }
        )
    return summaries


def main():
    args = parse_args()
    reference_checkpoint = torch_load(args.reference)
    candidate_checkpoint = torch_load(args.candidate)
    report = {
        "reference": {
            "global_step": reference_checkpoint.get("global_step"),
            "epoch": reference_checkpoint.get("epoch"),
            "optimizers": optimizer_summary(reference_checkpoint),
        },
        "candidate": {
            "global_step": candidate_checkpoint.get("global_step"),
            "epoch": candidate_checkpoint.get("epoch"),
            "optimizers": optimizer_summary(candidate_checkpoint),
        },
        "model": compare_models(
            model_state(reference_checkpoint, args.reference),
            model_state(candidate_checkpoint, args.candidate),
            args.atol,
            args.rtol,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["model"]["allclose"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
