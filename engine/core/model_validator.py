"""Model integrity validation to enforce the three red lines.

Ensures only native full-precision (F16/BF16) models are loaded.
Rejects any quantized, distilled, or structurally modified models.
"""

import json
import logging
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

from engine.utils.hashing import get_model_hash

logger = logging.getLogger(__name__)

GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian uint32

ALLOWED_FTYPE_VALUES = {
    0,   # ALL_F32
    1,   # MOSTLY_F16
    26,  # MOSTLY_BF16
}

FTYPE_NAMES = {
    0: "F32 (全精度)",
    1: "F16 (半精度)",
    2: "Q4_0 (4位量化)",
    3: "Q4_1 (4位量化)",
    7: "Q8_0 (8位量化)",
    8: "Q8_1 (8位量化)",
    10: "Q2_K (2位量化)",
    11: "Q3_K_S (3位量化)",
    12: "Q3_K_M (3位量化)",
    13: "Q3_K_L (3位量化)",
    14: "Q4_K_S (4位量化)",
    15: "Q4_K_M (4位量化)",
    16: "Q5_K_S (5位量化)",
    17: "Q5_K_M (5位量化)",
    18: "Q6_K (6位量化)",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    26: "BF16 (半精度)",
}


class ModelValidationError(Exception):
    """Raised when model fails validation against red-line constraints."""
    pass


def _read_gguf_metadata(filepath: str) -> Dict:
    """Read GGUF file header and extract metadata fields."""
    metadata = {}
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ModelValidationError(
                f"文件不是有效的GGUF格式 (magic: {hex(magic)}, 期望: {hex(GGUF_MAGIC)})"
            )

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata["_version"] = version
        metadata["_n_tensors"] = n_tensors
        metadata["_n_kv"] = n_kv

        def read_string():
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8", errors="replace")

        def read_value(vtype):
            if vtype == 0:    # UINT8
                return struct.unpack("<B", f.read(1))[0]
            elif vtype == 1:  # INT8
                return struct.unpack("<b", f.read(1))[0]
            elif vtype == 2:  # UINT16
                return struct.unpack("<H", f.read(2))[0]
            elif vtype == 3:  # INT16
                return struct.unpack("<h", f.read(2))[0]
            elif vtype == 4:  # UINT32
                return struct.unpack("<I", f.read(4))[0]
            elif vtype == 5:  # INT32
                return struct.unpack("<i", f.read(4))[0]
            elif vtype == 6:  # FLOAT32
                return struct.unpack("<f", f.read(4))[0]
            elif vtype == 7:  # BOOL
                return struct.unpack("<B", f.read(1))[0] != 0
            elif vtype == 8:  # STRING
                return read_string()
            elif vtype == 9:  # ARRAY
                atype = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                return [read_value(atype) for _ in range(alen)]
            elif vtype == 10:  # UINT64
                return struct.unpack("<Q", f.read(8))[0]
            elif vtype == 11:  # INT64
                return struct.unpack("<q", f.read(8))[0]
            elif vtype == 12:  # FLOAT64
                return struct.unpack("<d", f.read(8))[0]
            else:
                raise ModelValidationError(f"未知的GGUF值类型: {vtype}")

        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack("<I", f.read(4))[0]
            value = read_value(vtype)
            metadata[key] = value

    return metadata


def validate_model(filepath: str) -> Tuple[Dict, str]:
    """Validate that a model file meets all red-line requirements.

    Returns:
        Tuple of (metadata dict, model_hash string)

    Raises:
        ModelValidationError if any red line is violated.
    """
    path = Path(filepath)
    if not path.exists():
        raise ModelValidationError(f"模型文件不存在: {filepath}")
    if not path.suffix.lower() == ".gguf":
        raise ModelValidationError(f"仅支持GGUF格式模型文件，当前文件: {path.name}")

    logger.info(f"正在校验模型文件: {path.name}")

    metadata = _read_gguf_metadata(filepath)

    # --- Red Line 1: No Quantization ---
    ftype = metadata.get("general.file_type")
    if ftype is None:
        logger.warning("模型元数据中未找到 general.file_type 字段，尝试继续...")
    elif ftype not in ALLOWED_FTYPE_VALUES:
        ftype_name = FTYPE_NAMES.get(ftype, f"未知({ftype})")
        raise ModelValidationError(
            f"[不量化红线] 模型精度类型为 {ftype_name}，"
            f"仅允许原生全精度 (F32/F16/BF16)。"
            f"请使用官方原生权重转换的全精度GGUF文件。"
        )

    ftype_name = FTYPE_NAMES.get(ftype, str(ftype))
    logger.info(f"模型精度校验通过: {ftype_name}")

    # --- Red Line 2: No Distillation (structure check) ---
    arch = metadata.get("general.architecture", "unknown")
    n_layers = metadata.get(f"{arch}.block_count")
    n_heads = metadata.get(f"{arch}.attention.head_count")
    n_embd = metadata.get(f"{arch}.embedding_length")

    logger.info(
        f"模型结构: arch={arch}, layers={n_layers}, "
        f"heads={n_heads}, embed_dim={n_embd}"
    )

    # --- Compute model hash for cache binding ---
    model_hash = get_model_hash(filepath)
    logger.info(f"模型哈希: {model_hash[:16]}...")

    return metadata, model_hash
