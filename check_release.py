"""检查独立展示目录，并只把已复核的白名单文件打包。

检查不读取父目录或旧 Git 历史；常见秘密扫描不是完整安全审计。
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


PUBLIC_FILES = frozenset({
    ".gitignore", "README.md", "ARCHITECTURE.md", "PUBLICATION.md",
    "VERIFICATION.md", "requirements.txt", "demo.py", "check_release.py",
    "model/__init__.py", "model/components.py", "model/demo_model.py", "model/xlstm.py",
})
MAX_FILE_BYTES = 256 * 1024
SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws-access-id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential-assignment": re.compile(
        r"(?im)^\s*(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
    ),
    "credential-in-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "local-user-path": re.compile(r"(?i)(?:[A-Z]:[/\\]+Users[/\\]+|/home/|/Users/)[A-Za-z0-9_.-]+"),
}


def audit(root):
    """返回待打包的字节快照，拒绝未知文件，避免扫描后再读取不同内容。"""
    errors = []
    payload = {}

    def visit(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                errors.append(relative + ": link/reparse point is not allowed")
                continue
            if relative == ".git":
                if not path.is_dir():
                    errors.append(".git must not be a worktree pointer file")
                continue
            if path.is_dir():
                if relative in {"model", "__pycache__", "model/__pycache__"}:
                    visit(path)
                else:
                    errors.append(relative + ": unexpected directory")
                continue
            if not stat.S_ISREG(info.st_mode):
                errors.append(relative + ": not a regular file")
                continue
            if path.parent.name == "__pycache__" and path.suffix == ".pyc":
                continue
            if relative not in PUBLIC_FILES:
                errors.append(relative + ": not in public allowlist")
                continue
            if info.st_size > MAX_FILE_BYTES:
                errors.append(relative + ": exceeds source file size limit")
                continue
            content = path.read_bytes()
            try:
                source = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                errors.append(relative + ": expected UTF-8 text")
                continue
            if "\x00" in source:
                errors.append(relative + ": binary content is not allowed")
                continue
            for category, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(source):
                    line = source.count("\n", 0, match.start()) + 1
                    errors.append(relative + ":" + str(line) + ": review " + category)
            payload[relative] = content

    visit(root)
    for missing in sorted(PUBLIC_FILES - payload.keys()):
        errors.append(missing + ": missing required public file")
    if errors:
        raise ValueError("Public release check failed:\n" + "\n".join(errors))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, dest="archive", help="Create a new archive outside this directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        payload = audit(root)
        archive_path = None
        if args.archive:
            archive_path = args.archive.expanduser().resolve()
            if root == archive_path.parent or root in archive_path.parents:
                raise ValueError("ZIP output must be outside the public directory")
            if archive_path.suffix.lower() != ".zip":
                raise ValueError("Archive must use the .zip extension")
            # x 模式拒绝覆盖已有包；只使用扫描时的字节快照，不遍历父项目。
            with zipfile.ZipFile(str(archive_path), "x", zipfile.ZIP_DEFLATED) as archive:
                for name, content in sorted(payload.items()):
                    archive.writestr("cs2-price-prediction/" + name, content)
            with zipfile.ZipFile(str(archive_path), "r") as archive:
                expected = {"cs2-price-prediction/" + name for name in payload}
                if set(archive.namelist()) != expected or archive.testzip() is not None:
                    raise ValueError("Archive integrity verification failed")
        print(json.dumps({
            "status": "PASS", "file_count": len(payload),
            "total_bytes": sum(map(len, payload.values())),
            "archive_created": archive_path is not None,
            "files": [
                {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in sorted(payload.items())
            ],
            "scope": "Allowlist and common-pattern scan; not a complete security or provenance audit",
        }, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
