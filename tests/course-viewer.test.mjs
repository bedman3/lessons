import test from 'node:test';
import assert from 'node:assert/strict';

import { renderMarkdown } from '../assets/course-viewer.mjs';


test('renders headings, emphasis, code, and safe links', () => {
  const markdown = [
    '# Lesson',
    '',
    'Read **carefully**, use `x`, and visit [the next chapter](ch2-viewer.html).',
  ].join('\n');

  const html = renderMarkdown(markdown);

  assert.match(html, /<h1>Lesson<\/h1>/);
  assert.match(html, /<strong>carefully<\/strong>/);
  assert.match(html, /<code>x<\/code>/);
  assert.match(html, /<a href="ch2-viewer.html">the next chapter<\/a>/);
});

test('does not turn unsafe markdown links into anchors', () => {
  const html = renderMarkdown('[bad](javascript:alert(1))');

  assert.doesNotMatch(html, /<a /);
  assert.match(html, /\[bad\]/);
});

test('renders tables, lists, fenced code, and display mathematics', () => {
  const markdown = [
    '| A | B |',
    '|---|---|',
    '| 1 | 2 |',
    '',
    '- first',
    '- second',
    '',
    '```python',
    'x = 1 < 2',
    '```',
    '',
    '$$',
    'x^2',
    '$$',
  ].join('\n');

  const html = renderMarkdown(markdown);

  assert.match(html, /<table>/);
  assert.match(html, /<ul><li>first<\/li><li>second<\/li><\/ul>/);
  assert.match(html, /<pre><code class="language-python">x = 1 &lt; 2<\/code><\/pre>/);
  assert.match(html, /<div class="display-math">\$\$\nx\^2\n\$\$<\/div>/);
});
