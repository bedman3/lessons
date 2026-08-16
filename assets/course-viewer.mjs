function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function safeHref(rawHref) {
  const href = rawHref.trim();
  if (/^(https?:|mailto:|#|\.\.?\/)/i.test(href)) return href;
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//')) return null;
  return href;
}

function renderInline(text) {
  let rendered = escapeHtml(text);
  rendered = rendered.replace(/`([^`]+)`/g, '<code>$1</code>');
  rendered = rendered.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, rawHref) => {
    const href = safeHref(rawHref);
    return href === null ? match : `<a href="${escapeHtml(href)}">${label}</a>`;
  });
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  rendered = rendered.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return rendered;
}

function tableCells(line) {
  return line
    .trim()
    .replace(/^\||\|$/g, '')
    .split('|')
    .map((cell) => renderInline(cell.trim()));
}

function isTableDivider(line) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line);
}

export function renderMarkdown(markdown) {
  const lines = markdown.replaceAll('\r\n', '\n').split('\n');
  let html = '';
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (line.startsWith('```')) {
      const language = line.slice(3).trim().replace(/[^a-z0-9_-]/gi, '');
      index += 1;
      const code = [];
      while (index < lines.length && !lines[index].startsWith('```')) code.push(lines[index++]);
      html += `<pre><code${language ? ` class="language-${language}"` : ''}>${escapeHtml(code.join('\n'))}</code></pre>`;
      if (index < lines.length) index += 1;
      continue;
    }

    if (line.trim() === '$$') {
      const equation = ['$$'];
      index += 1;
      while (index < lines.length && lines[index].trim() !== '$$') equation.push(lines[index++]);
      equation.push('$$');
      html += `<div class="display-math">${escapeHtml(equation.join('\n'))}</div>`;
      if (index < lines.length) index += 1;
      continue;
    }

    if (line.startsWith('|') && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const heading = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].startsWith('|')) {
        rows.push(`<tr>${tableCells(lines[index++]).map((cell) => `<td>${cell}</td>`).join('')}</tr>`);
      }
      html += `<table><thead><tr>${heading.map((cell) => `<th>${cell}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      html += `<h${level}>${renderInline(heading[2])}</h${level}>`;
      index += 1;
      continue;
    }

    if (/^---+$/.test(line)) {
      html += '<hr>';
      index += 1;
      continue;
    }

    if (line.startsWith('> ')) {
      const quote = [];
      while (index < lines.length && lines[index].startsWith('> ')) quote.push(lines[index++].slice(2));
      html += `<blockquote>${renderInline(quote.join(' '))}</blockquote>`;
      continue;
    }

    if (/^- /.test(line)) {
      const items = [];
      while (index < lines.length && /^- /.test(lines[index])) items.push(`<li>${renderInline(lines[index++].slice(2))}</li>`);
      html += `<ul>${items.join('')}</ul>`;
      continue;
    }

    if (/^\d+\. /.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+\. /.test(lines[index])) {
        items.push(`<li>${renderInline(lines[index++].replace(/^\d+\. /, ''))}</li>`);
      }
      html += `<ol>${items.join('')}</ol>`;
      continue;
    }

    if (line.trim() === '') {
      index += 1;
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (
      index < lines.length
      && lines[index].trim() !== ''
      && !/^(#{1,4})\s|^\||^> |^- |^\d+\. |^---+$|^```|^\$\$/.test(lines[index])
    ) {
      paragraph.push(lines[index++]);
    }
    html += `<p>${renderInline(paragraph.join(' '))}</p>`;
  }

  return html;
}

function typeset(target, attempt = 0) {
  if (globalThis.MathJax?.typesetPromise) {
    globalThis.MathJax.typesetPromise([target]);
    return;
  }
  if (attempt < 100) globalThis.setTimeout(() => typeset(target, attempt + 1), 50);
}

async function boot() {
  const target = document.querySelector('main');
  const source = document.body.dataset.source;
  if (!target || !source) return;
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Could not load lesson (${response.status})`);
    target.innerHTML = renderMarkdown(await response.text());
    typeset(target);
  } catch (error) {
    target.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

if (typeof document !== 'undefined') {
  addEventListener('DOMContentLoaded', boot);
}
