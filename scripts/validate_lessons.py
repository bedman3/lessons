#!/usr/bin/env python3
"""Structural checks for static lesson courses."""

from __future__ import annotations

import argparse
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "link", "script"}:
            return
        keys = {key: value for key, value in attrs}
        target = keys.get("href") if tag in {"a", "link"} else keys.get("src")
        if target:
            self.links.append(target)


def validate_markdown(path: Path) -> list[str]:
    """Return structural Markdown errors for one lesson."""
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not text.startswith("# "):
        errors.append(f"{path}: lesson must start with one level-1 heading")
    if sum(1 for line in text.splitlines() if line.startswith("```")) % 2:
        errors.append(f"{path}: unbalanced fenced code block")
    if sum(1 for line in text.splitlines() if line.strip() == "$$") % 2:
        errors.append(f"{path}: unbalanced display-math delimiter")
    return errors


def _local_target(root: Path, page: Path, raw_link: str) -> Path | None:
    if raw_link.startswith("#"):
        return None
    parsed = urlsplit(raw_link)
    if parsed.scheme or parsed.netloc:
        return None
    clean_path = unquote(parsed.path)
    if not clean_path:
        return None
    if clean_path.startswith("/"):
        return root / clean_path.lstrip("/")
    return page.parent / clean_path


def validate_html_links(root: Path, path: Path) -> list[str]:
    """Return missing local targets referenced by one HTML page."""
    parser = _LinkParser()
    parser.feed(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for link in parser.links:
        target = _local_target(root, path, link)
        if target is not None and not target.resolve().exists():
            errors.append(f"{path}: missing local link: {link}")
    return errors


def validate_course(root: Path, course: Path) -> list[str]:
    """Validate lesson/viewer pairing, Markdown, navigation, and local links."""
    root = root.resolve()
    course = course.resolve()
    errors: list[str] = []
    lessons = sorted(course.glob("*-lesson.md"))
    viewers = sorted(course.glob("*-viewer.html"))

    for lesson in lessons:
        errors.extend(validate_markdown(lesson))
        viewer = lesson.with_name(lesson.name.replace("-lesson.md", "-viewer.html"))
        if not viewer.exists():
            errors.append(f"{course}: missing viewer for {lesson.name}")

    for viewer in viewers:
        lesson = viewer.with_name(viewer.name.replace("-viewer.html", "-lesson.md"))
        if not lesson.exists():
            errors.append(f"{course}: missing lesson for {viewer.name}")
        source = viewer.read_text(encoding="utf-8")
        if "index.html" not in source:
            errors.append(f"{viewer}: missing course navigation")
        if "../index.html" not in source:
            errors.append(f"{viewer}: missing root navigation")

    html_pages = sorted(course.glob("*.html"))
    for page in html_pages:
        errors.extend(validate_html_links(root, page))

    index = course / "index.html"
    if not index.exists():
        errors.append(f"{course}: missing course index.html")
    return errors


def _self_test() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        course = root / "course"
        course.mkdir()
        lesson = course / "ch1-test-lesson.md"
        lesson.write_text("# Test\n\n```python\n", encoding="utf-8")
        (course / "index.html").write_text('<a href="missing.html">Broken</a>', encoding="utf-8")
        errors = validate_course(root, course)
        expected = ("unbalanced fenced code block", "missing viewer", "missing local link")
        if not all(any(fragment in error for error in errors) for fragment in expected):
            print("Self-test failed")
            for error in errors:
                print(error)
            return 1
    print("Self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("course", nargs="?", type=Path)
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if args.markdown_only:
        if args.root is None:
            parser.error("--markdown-only requires a course directory")
        course = args.root
        errors = [error for path in sorted(course.glob("*-lesson.md")) for error in validate_markdown(path)]
        if errors:
            print("\n".join(errors))
            return 1
        print(f"Validated Markdown: {len(list(course.glob('*-lesson.md')))} lessons")
        return 0

    if args.root is None or args.course is None:
        parser.error("provide repository root and course directory")
    errors = validate_course(args.root, args.course)
    if errors:
        print("\n".join(errors))
        return 1
    lessons = len(list(args.course.glob("*-lesson.md")))
    pages = len(list(args.course.glob("*.html")))
    print(f"Validated {args.course.name}: {lessons} lessons, {pages} HTML pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
