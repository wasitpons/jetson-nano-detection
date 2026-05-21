#!/usr/bin/env python3
"""In-place edit of config.yaml's detector.engine + detector.imgsz lines.

PyYAML's safe_dump destroys comments and reflows lists; we want to preserve
the human edits in config.yaml. So this helper does targeted line surgery:
it scans line-by-line within the `detector:` block and replaces only the
two relevant lines. If the keys are missing it appends them at the end of
the block.

Usage:
    python3 _set_active_engine.py <config.yaml> <engine_path> <imgsz>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ENGINE_RE = re.compile(r"^(\s*)engine\s*:\s*\S.*$")
IMGSZ_RE = re.compile(r"^(\s*)imgsz\s*:\s*\S.*$")
DETECTOR_HEADER_RE = re.compile(r"^detector\s*:\s*$")
# Any non-indented "top-level" key terminates the detector block.
TOPLEVEL_KEY_RE = re.compile(r"^\S")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: _set_active_engine.py <config.yaml> <engine_path> <imgsz>",
              file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    engine = sys.argv[2]
    imgsz = sys.argv[3]
    int(imgsz)  # validate it's numeric

    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        return 2

    lines = config_path.read_text().splitlines()
    out = []
    in_detector = False
    detector_indent = None
    engine_set = False
    imgsz_set = False
    detector_start_idx = None
    detector_end_idx = None

    for idx, line in enumerate(lines):
        if DETECTOR_HEADER_RE.match(line):
            in_detector = True
            detector_indent = None
            detector_start_idx = idx
            out.append(line)
            continue
        if in_detector:
            stripped = line.lstrip()
            # First non-empty non-comment indented line establishes indent.
            if stripped and not stripped.startswith("#") and detector_indent is None:
                detector_indent = " " * (len(line) - len(stripped))
            # End of detector block: a line that's non-empty and not indented.
            if line and TOPLEVEL_KEY_RE.match(line):
                detector_end_idx = idx
                in_detector = False
                # We'll re-process this line below.
                out.append(line)
                continue

            if ENGINE_RE.match(line):
                indent = ENGINE_RE.match(line).group(1)
                out.append(f"{indent}engine: {engine}")
                engine_set = True
                continue
            if IMGSZ_RE.match(line):
                indent = IMGSZ_RE.match(line).group(1)
                out.append(f"{indent}imgsz: {imgsz}")
                imgsz_set = True
                continue

        out.append(line)

    # If we ran off the end while still inside detector:
    if in_detector and detector_end_idx is None:
        detector_end_idx = len(lines)

    # If one or both keys weren't found, append them inside the block.
    if (not engine_set or not imgsz_set) and detector_start_idx is not None:
        indent = detector_indent or "  "
        insertions = []
        if not engine_set:
            insertions.append(f"{indent}engine: {engine}")
        if not imgsz_set:
            insertions.append(f"{indent}imgsz: {imgsz}")
        # Insert just before the line that ended the detector block (or at end).
        insert_at = detector_end_idx if detector_end_idx is not None else len(out)
        out[insert_at:insert_at] = insertions

    config_path.write_text("\n".join(out) + "\n")
    print(f"updated {config_path}: engine={engine} imgsz={imgsz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
