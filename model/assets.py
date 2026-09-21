"""Unpack the small tokenizer files supplied with the models."""
from pathlib import Path
import zipfile


def tokenizers():
    root = Path(__file__).resolve().parents[1]
    target = root / "outputs" / "tokenizers"
    with zipfile.ZipFile(Path(__file__).with_name("tokenizers.zip")) as archive:
        for member in archive.infolist():
            path = target / member.filename
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("Invalid tokenizer archive path")
            if not path.is_file() or path.stat().st_size != member.file_size:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))
    return target
