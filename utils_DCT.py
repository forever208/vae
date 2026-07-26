import torch_dct as dct
import torch


def split_into_blocks_torch(image: torch.Tensor, block_sz: int):
    """
    Split a 2D tensor (H, W) or batched 3D/4D tensor (B, H, W) into non-overlapping (block_sz x block_sz) blocks.

    Args:
        image (Tensor): shape (H, W) or (B, H, W) or (B, C, H, W)
        block_sz (int): block size

    Returns:
        Tensor:
            - (N_blocks, block_sz, block_sz) if input is (H, W)
            - (B, N_blocks, block_sz, block_sz) if input is (B, H, W)
    """
    if image.dim() == 2:  # (H, W)
        H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(0, block_sz, block_sz).unfold(1, block_sz, block_sz)  # (H/b, W/b, b, b)
        return blocks.contiguous().view(-1, block_sz, block_sz)  # (N_blocks, b, b)

    elif image.dim() == 3:  # (B, H, W)
        B, H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(1, block_sz, block_sz).unfold(2, block_sz, block_sz)  # (B, H/b, W/b, b, b)
        blocks = blocks.contiguous().view(B, -1, block_sz, block_sz)  # (B, N_blocks, b, b)
        return blocks

    elif image.dim() == 4:  # (B, C, H, W)
        B, C, H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(2, block_sz, block_sz).unfold(3, block_sz, block_sz)  # (B, C, H/b, W/b, b, b)
        blocks = blocks.contiguous().view(B, C, -1, block_sz, block_sz)  # (B, C, N_blocks, b, b)
        return blocks

    else:
        raise ValueError(f"Input tensor must be 2D or 3D or 4D, got shape {image.shape}")


def combine_blocks_torch(blocks: torch.Tensor, height: int, width: int, block_sz: int):
    """
    Combine non-overlapping blocks into full image.

    Args:
        blocks:
            - (N, B, B) tensor (single image)
            - (batch, N, B, B) tensor (batched images)
            - (batch, C, N, B, B) tensor (batched multi-channel images)
        height: original image height
        width: original image width
        block_sz: size of each block (B)

    Returns:
        image:
            - (height, width) if input is 3D
            - (batch, height, width) if input is 4D
            - (batch, C, height, width) if input is 5D
    """
    blocks_per_row = width // block_sz
    blocks_per_col = height // block_sz

    if blocks.dim() == 3:  # (N, B, B)
        image = blocks.view(blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 2, 1, 3).reshape(height, width)
        return image
    elif blocks.dim() == 4:  # (batch, N, B, B)
        B = blocks.size(0)
        image = blocks.view(B, blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 1, 3, 2, 4).reshape(B, height, width)
        return image
    elif blocks.dim() == 5:  # (batch, C, N, B, B)
        B = blocks.size(0)
        C = blocks.size(1)
        image = blocks.view(B, C, blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 1, 2, 4, 3, 5).reshape(B, C, height, width)
        return image
    else:
        raise ValueError(f"Expected input of shape (N, B, B) or (batch, N, B, B) or (batch, C, N, B, B), but got {blocks.shape}")


def idct_2d_torch_unified(X: torch.Tensor, center: str = "none", mean: torch.Tensor = None):
    """
    Inverse of dct_2d_torch_unified using 2D IDCT (Type-III) with ortho norm.

    X: (..., H, W)  DCT coefficients from dct_2d_torch_unified
    center:
      - "none": exact inverse if forward used center="none"
      - "mean": exact inverse if forward used center="mean" AND you pass `mean`
    mean:
      - required when center="mean": the mean that was subtracted in forward,
        shape broadcastable to (..., 1, 1)

    returns: (..., H, W)
    """
    X = X.float()

    # Invert separable 2D DCT:
    # Forward: dct(last) -> transpose -> dct(last) -> transpose back
    # Inverse: idct(last) -> transpose -> idct(last) -> transpose back
    x = dct.idct(X, norm="ortho")                          # inverse along last dim
    x = dct.idct(x.transpose(-2, -1), norm="ortho")        # inverse along second-last dim
    x = x.transpose(-2, -1)

    if center == "none":
        return x
    elif center == "mean":
        if mean is None:
            raise ValueError("center='mean' requires `mean` (the value subtracted in forward).")
        return x + mean
    else:
        raise ValueError("center must be 'mean' or 'none'")


# ============================================================
# 1) Unified 2D DCT for inputs roughly in [-1, 1]
# ============================================================
def dct_2d_torch_unified(x: torch.Tensor, center: str = "mean"):
    """
    2D DCT-II (ortho) for inputs in roughly [-1, 1], works for RGB or latents.

    x: (..., H, W)
    center:
      - "mean": subtract spatial mean per sample (recommended)
      - "none": no centering (DC can dominate)
    returns: (..., H, W)
    """
    x = x.float()
    if center == "mean":
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    elif center == "none":
        pass
    else:
        raise ValueError("center must be 'mean' or 'none'")

    x = dct.dct(x, norm="ortho")                       # last dim
    x = dct.dct(x.transpose(-2, -1), norm="ortho")     # second-last
    return x.transpose(-2, -1)
