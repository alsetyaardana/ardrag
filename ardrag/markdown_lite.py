"""Minimal markdown renderer for chat answers — headings, ordered/bullet lists, pipe tables,
fenced/inline code, blockquotes, horizontal rules, links, and bold/italic/strikethrough. Covers
the markdown constructs DeepSeek (and LLMs generally) actually reach for in RAG-style answers —
this isn't a full CommonMark implementation, just enough that answers stop showing raw `**`/`##`/
`1.`/`[text](url)` syntax to the user. Output uses chat-md-* CSS classes (see index.html).
"""
import html
import re

_INLINE_CODE_RE = re.compile(r"`([^`]+?)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)|(?<!_)_([^_\s][^_]*?)_(?!_)")

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_HEADING_LINE_RE = re.compile(r"^#{1,4}\s")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+")
_ORDERED_STRIP_RE = re.compile(r"^\s*\d+[.)]\s+")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:-]+\|[\s:|-]+\s*$")
_BLANK_TABLE_SEP_LINE_RE = re.compile(r"^[\s|:-]+$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")

_BLOCK_STARTS = (_BULLET_RE, _ORDERED_RE, _HEADING_LINE_RE, _TABLE_ROW_RE, _BLOCKQUOTE_RE, _HR_RE, _FENCE_RE)


def _is_block_start(line: str) -> bool:
    return any(p.match(line) for p in _BLOCK_STARTS)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = _INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _LINK_RE.sub(r'<a href="\2" target="_blank" rel="noopener">\1</a>', escaped)
    escaped = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", escaped)
    escaped = _STRIKE_RE.sub(r"<del>\1</del>", escaped)
    escaped = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", escaped)
    return escaped


def _table_cells(line: str) -> list[str]:
    stripped = re.sub(r"^\||\|$", "", line.strip())
    return [c.strip() for c in stripped.split("|")]


def _render_table(lines: list[str]) -> str:
    header = _table_cells(lines[0])
    body_lines = [l for l in lines[1:] if not _BLANK_TABLE_SEP_LINE_RE.match(l)]
    thead = "<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>"
    tbody = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _table_cells(l)) + "</tr>" for l in body_lines
    )
    return f'<table class="chat-md-table"><thead>{thead}</thead><tbody>{tbody}</tbody></table>'


def render_markdown_lite(raw: str) -> str:
    lines = raw.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        fence = _FENCE_RE.match(line.strip())
        if fence:
            lang = fence.group(1)
            code_lines = []
            i += 1
            while i < n and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip the closing fence (or run off the end if the model never closed it)
            code_html = html.escape("\n".join(code_lines))
            lang_class = f' class="language-{lang}"' if lang else ""
            out.append(f'<pre class="chat-md-pre"><code{lang_class}>{code_html}</code></pre>')
            continue

        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            table_lines = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            out.append(_render_table(table_lines))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            out.append(f'<div class="chat-md-h{level}">{_inline(heading.group(2))}</div>')
            i += 1
            continue

        if _HR_RE.match(line):
            out.append('<hr class="chat-md-hr">')
            i += 1
            continue

        if _BLOCKQUOTE_RE.match(line):
            quoted = []
            while i < n and _BLOCKQUOTE_RE.match(lines[i]):
                quoted.append(_inline(_BLOCKQUOTE_RE.match(lines[i]).group(1)))
                i += 1
            out.append(f'<blockquote class="chat-md-quote">{"<br>".join(quoted)}</blockquote>')
            continue

        if _ORDERED_RE.match(line):
            items = []
            while i < n and _ORDERED_RE.match(lines[i]):
                items.append(f"<li>{_inline(_ORDERED_STRIP_RE.sub('', lines[i]))}</li>")
                i += 1
            out.append(f'<ol class="chat-md-list">{"".join(items)}</ol>')
            continue

        if _BULLET_RE.match(line):
            items = []
            while i < n and _BULLET_RE.match(lines[i]):
                items.append(f"<li>{_inline(_BULLET_RE.sub('', lines[i]))}</li>")
                i += 1
            out.append(f'<ul class="chat-md-list">{"".join(items)}</ul>')
            continue

        if line.strip() == "":
            i += 1
            continue

        para = []
        while i < n and lines[i].strip() != "" and not _is_block_start(lines[i]):
            para.append(_inline(lines[i]))
            i += 1
        out.append(f'<p class="chat-md-p">{"<br>".join(para)}</p>')

    return "".join(out)
