"""Frozen PhoBERT, Vietnamese segmentation, token windows and original offsets."""

import importlib.metadata
import re
from pathlib import Path

import torch
from pyvi import ViTokenizer
from transformers import AutoModel, AutoTokenizer
from data_pipeline import PIPELINE_VERSION, atoms, digest


def segmented_words(text):
    """Align PyVi words back to the unchanged input; never infer annotation offsets."""
    result = []
    for line in re.finditer(r"[^\r\n]+", text):
        cursor = line.start()
        for word in ViTokenizer.tokenize(line.group()).split():
            literal_start = cursor
            while literal_start < line.end() and text[literal_start].isspace():
                literal_start += 1
            # Preserve literal underscores (e.g. signature separator lines).
            if text.startswith(word, literal_start):
                cursor = literal_start + len(word)
                result.append((word, literal_start, cursor))
                continue
            start = None
            for part in word.split("_"):
                if not part:
                    continue
                match = text.find(part, cursor, line.end())
                if match < 0 or any(
                    not c.isspace() and c != "_" for c in text[cursor:match]
                ):
                    raise ValueError(
                        f"Cannot align segmented word {word!r} at character {cursor}; inspect raw text."
                    )
                start = match if start is None else start
                cursor = match + len(part)
            if start is not None:
                result.append((word, start, cursor))
        if text[cursor : line.end()].strip():
            raise ValueError(f"Segmentation dropped input near character {cursor}")
    return result


def aligned_units(text, words):
    """Split raw lexical units when PyVi splits inside them (e.g. '2,5a')."""
    units = []
    index = 0
    for start, end in atoms(text):
        cursor = start
        while cursor < end:
            while index + 1 < len(words) and words[index][2] <= cursor:
                index += 1
            if not words or not words[index][1] <= cursor < words[index][2]:
                raise ValueError(f"No segmented word covers character {cursor}")
            stop = min(end, words[index][2])
            units.append((cursor, stop))
            cursor = stop
    return units


class ContextEncoder:
    def __init__(
        self,
        model_name="vinai/phobert-base-v2",
        window=256,
        stride=128,
        batch_size=8,
        device=None,
        revision="main",
        local_files_only=False,
    ):
        if not 4 <= window <= 256 or not 1 <= stride <= window - 2:
            raise ValueError("Require window<=256 and 1<=stride<=window-2")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 768:
            raise ValueError("This pipeline expects a 768-dimensional PhoBERT encoder")
        self.window, self.stride, self.batch_size = window, stride, batch_size
        self.config = {
            "version": PIPELINE_VERSION,
            "model_name": model_name,
            "revision": getattr(self.model.config, "_commit_hash", None) or revision,
            "window": window,
            "stride": stride,
            "pyvi_version": importlib.metadata.version("pyvi"),
            "transformers_version": importlib.metadata.version("transformers"),
        }

    @torch.no_grad()
    def encode(self, text, cache_dir=None):
        key = digest(str(sorted(self.config.items())) + text)
        path = Path(cache_dir) / f"{key}.pt" if cache_dir else None
        if path and path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        words = segmented_words(text)
        units = aligned_units(text, words)
        if not words or not units:
            return {"x": torch.empty(0, 772), "offsets": units, "cache_key": key}
        token_ids, owners = [], []
        for i, (word, _, _) in enumerate(words):
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not ids:
                ids = [self.tokenizer.unk_token_id]
            token_ids.extend(ids)
            owners.extend([i] * len(ids))
        capacity = self.window - 2
        starts = []
        for start in range(0, len(token_ids), self.stride):
            starts.append(start)
            if start + capacity >= len(token_ids):
                break
        # Accumulate on CPU: long legal documents do not occupy all GPU memory.
        sums = torch.zeros(len(words), 768)
        counts = torch.zeros(len(words), 1)
        for first in range(0, len(starts), self.batch_size):
            positions = starts[first : first + self.batch_size]
            chunks = [token_ids[s : s + capacity] for s in positions]
            width = max(map(len, chunks)) + 2
            ids = torch.full(
                (len(chunks), width), self.tokenizer.pad_token_id, dtype=torch.long
            )
            mask = torch.zeros_like(ids)
            for i, chunk in enumerate(chunks):
                seq = (
                    [self.tokenizer.bos_token_id]
                    + chunk
                    + [self.tokenizer.eos_token_id]
                )
                ids[i, : len(seq)] = torch.tensor(seq)
                mask[i, : len(seq)] = 1
            out = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            ).last_hidden_state.cpu()
            for i, (start, chunk) in enumerate(zip(positions, chunks)):
                idx = torch.tensor(owners[start : start + len(chunk)])
                sums.index_add_(0, idx, out[i, 1 : 1 + len(chunk)])
                counts.index_add_(0, idx, torch.ones(len(chunk), 1))
        means = sums / counts.clamp_min(1)
        features = torch.empty(len(units), 772)
        word_idx = 0
        for i, (a, b) in enumerate(units):
            while word_idx + 1 < len(words) and words[word_idx][2] <= a:
                word_idx += 1
            _, wa, wb = words[word_idx]
            if not wa <= a < b <= wb:
                raise ValueError(
                    f"Lexical unit {(a, b)} not covered by segmented words"
                )
            features[i, :768] = means[word_idx]
            features[i, 768:] = torch.tensor(
                [
                    (a - wa) / max(1, wb - wa),
                    (b - wa) / max(1, wb - wa),
                    a / max(1, len(text)),
                    min(b - a, 100) / 100,
                ]
            )
        result = {"x": features.half(), "offsets": units, "cache_key": key}
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, path)
        return result
