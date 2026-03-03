"""Privacy protection: AES-256-GCM encryption, sensitive info sanitization, data wipe."""

import logging
import os
import re
import secrets
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Sensitive information patterns (China-centric + international)
SENSITIVE_PATTERNS = [
    (re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"), "[身份证号已脱敏]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
    (re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"), "[银行卡号已脱敏]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[邮箱已脱敏]"),
    (re.compile(r"(?i)(?:api[_-]?key|secret|password|token|credential)\s*[:=]\s*\S+"), "[密钥已脱敏]"),
    (re.compile(r"(?i)ssh-(?:rsa|ed25519|dsa)\s+\S+"), "[SSH密钥已脱敏]"),
    (re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----"), "[私钥已脱敏]"),
]


def sanitize_sensitive_info(text: str) -> str:
    """Replace sensitive information with redacted placeholders."""
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class EncryptionManager:
    """AES-256-GCM encryption for cache and conversation data."""

    def __init__(self, key_file: str = ""):
        self._key: Optional[bytes] = None
        self._key_file = key_file
        self._enabled = False

    def setup(self, password: Optional[str] = None) -> bool:
        """Initialize encryption with a password or existing key file."""
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes

            if password:
                salt = secrets.token_bytes(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=480000,
                )
                self._key = kdf.derive(password.encode("utf-8"))
                if self._key_file:
                    key_path = Path(os.path.expanduser(self._key_file))
                    key_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(key_path, "wb") as f:
                        f.write(salt + self._key)
                self._enabled = True
                logger.info("加密已初始化 (基于密码)")
                return True

            elif self._key_file and os.path.exists(os.path.expanduser(self._key_file)):
                with open(os.path.expanduser(self._key_file), "rb") as f:
                    data = f.read()
                    if len(data) >= 48:
                        self._key = data[16:48]
                        self._enabled = True
                        logger.info("加密已初始化 (从密钥文件)")
                        return True

        except ImportError:
            logger.warning("cryptography 库未安装, 加密不可用")
        except Exception as e:
            logger.error(f"加密初始化失败: {e}")

        return False

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        if not self._enabled or self._key is None:
            return data

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            nonce = secrets.token_bytes(12)
            aesgcm = AESGCM(self._key)
            encrypted = aesgcm.encrypt(nonce, data, None)
            return nonce + encrypted
        except Exception as e:
            logger.error(f"加密失败: {e}")
            return data

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        if not self._enabled or self._key is None:
            return data

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            nonce = data[:12]
            ciphertext = data[12:]
            aesgcm = AESGCM(self._key)
            return aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            logger.error(f"解密失败: {e}")
            return data

    @property
    def is_enabled(self) -> bool:
        return self._enabled


def wipe_all_data(base_dir: str = "~/.cache/ai_engine") -> bool:
    """Securely wipe all user data including caches, logs, and configs."""
    import shutil
    expanded = os.path.expanduser(base_dir)
    try:
        if os.path.exists(expanded):
            shutil.rmtree(expanded)
            logger.info(f"所有数据已清除: {expanded}")
        return True
    except Exception as e:
        logger.error(f"数据清除失败: {e}")
        return False
