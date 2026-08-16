import tempfile
import unittest
from pathlib import Path

from scripts.validate_lessons import validate_course, validate_html_links, validate_markdown


class MarkdownValidationTests(unittest.TestCase):
    def test_unbalanced_fence_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            lesson = Path(directory) / "broken-lesson.md"
            lesson.write_text("# Broken\n\n```python\nprint('open')\n", encoding="utf-8")

            errors = validate_markdown(lesson)

            self.assertIn("unbalanced fenced code block", "\n".join(errors))

    def test_valid_markdown_has_no_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            lesson = Path(directory) / "valid-lesson.md"
            lesson.write_text("# Valid\n\nA formula: $x^2$.\n", encoding="utf-8")

            self.assertEqual(validate_markdown(lesson), [])


class HtmlLinkValidationTests(unittest.TestCase):
    def test_missing_local_link_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "index.html"
            page.write_text('<a href="missing.html">Missing</a>', encoding="utf-8")

            errors = validate_html_links(root, page)

            self.assertIn("missing local link: missing.html", "\n".join(errors))

    def test_external_and_fragment_links_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "index.html"
            page.write_text(
                '<a href="#topic">Topic</a><a href="https://example.com">External</a>',
                encoding="utf-8",
            )

            self.assertEqual(validate_html_links(root, page), [])


class CourseValidationTests(unittest.TestCase):
    def test_lesson_without_matching_viewer_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            course = root / "course"
            course.mkdir()
            (course / "ch1-topic-lesson.md").write_text("# Topic\n", encoding="utf-8")
            (course / "index.html").write_text('<a href="ch1-topic-lesson.md">Markdown</a>', encoding="utf-8")

            errors = validate_course(root, course)

            self.assertIn("missing viewer for ch1-topic-lesson.md", "\n".join(errors))

    def test_complete_course_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            course = root / "course"
            course.mkdir()
            (course / "ch1-topic-lesson.md").write_text("# Topic\n", encoding="utf-8")
            (course / "ch1-topic-viewer.html").write_text(
                '<a href="index.html">Course TOC</a><a href="../index.html">All lessons</a>',
                encoding="utf-8",
            )
            (course / "index.html").write_text(
                '<a href="ch1-topic-viewer.html">Read</a><a href="ch1-topic-lesson.md">Markdown</a>',
                encoding="utf-8",
            )
            (root / "index.html").write_text("Root", encoding="utf-8")

            self.assertEqual(validate_course(root, course), [])


if __name__ == "__main__":
    unittest.main()
