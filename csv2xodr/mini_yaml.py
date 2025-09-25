"""A tiny YAML loader supporting the subset used by csv2xodr configs."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple


def _strip_comments(line: str) -> str:
    if "#" in line:
        idx = line.index("#")
        return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if value.startswith("0") and value != "0" and not value.startswith("0."):
            raise ValueError
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _preprocess(lines: Iterable[str]) -> List[Tuple[int, str]]:
    processed: List[Tuple[int, str]] = []
    for raw in lines:
        stripped = _strip_comments(raw).rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip())
        content = stripped.lstrip()
        processed.append((indent, content))
    return processed


def _parse_block(lines: Sequence[Tuple[int, str]], index: int, indent: int):
    if index >= len(lines):
        return {}, index
    current_indent, text = lines[index]
    if current_indent < indent:
        return {}, index
    if text.startswith("- "):
        result: List[Any] = []
        while index < len(lines):
            level, line = lines[index]
            if level != indent or not line.startswith("- "):
                break
            item_text = line[2:].strip()
            index += 1
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_block(lines, index, indent + 2)
                if item_text:
                    result.append(_parse_scalar(item_text))
                result.append(child)
            else:
                result.append(_parse_scalar(item_text))
        return result, index

    result_dict: dict = {}
    while index < len(lines):
        level, line = lines[index]
        if level < indent:
            break
        if level > indent:
            raise ValueError(f"Unexpected indentation at line: {line}")
        if ":" not in line:
            raise ValueError(f"Invalid line (expected key: value): {line}")
        key, value_text = line.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        index += 1
        if value_text == "":
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_block(lines, index, indent + 2)
                result_dict[key] = child
            else:
                result_dict[key] = {}
        else:
            result_dict[key] = _parse_scalar(value_text)
    return result_dict, index


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        lines = _preprocess(fh)
    if not lines:
        return {}
    data, _ = _parse_block(lines, 0, lines[0][0])
    return data

