"""ch03 结构感知 Markdown 切分器(spec §4)。纯函数、零 I/O、零第三方依赖。

顺序契约:文档内按阅读顺序产 chunk;FAQ 节内先 Q/A 组、后非问答内容(表格等)——
prev/next 链由 indexer 按此顺序回填。FAQ 的 Q/A chunk 原子(一问一答是自然粒度,
不再递归切、不参与重叠)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_KEY_SUFFIX = "（关键条款）"
_SENTENCE_END = "。！？；"
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_Q_PREFIX = re.compile(r"^Q[:：]\s*")
_A_PREFIX = re.compile(r"^A[:：]\s*")


@dataclass
class ChunkDraft:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool

    def vector_text(self) -> str:
        return "\n".join([self.category, self.questions, self.answer])  # §4-6:只在 embed 时拼


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """§4-7: ---\ncontent_type: xxx\n--- 头。缺省按 policy。"""
    m = re.match(r"^---\n(.*?)\n---\n?", text, flags=re.S)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def split_sentences(text: str) -> list[str]:
    """§4-2 句边界:。！？； 与「. + 空白 + 非数字」(2.0 这类版本号不断句)。"""
    sents: list[str] = []
    buf: list[str] = []
    for i, ch in enumerate(text):
        buf.append(ch)
        boundary = ch in _SENTENCE_END or (
            ch == "."
            and i + 2 < len(text)
            and text[i + 1].isspace()
            and not text[i + 2].isdigit()
        )
        if boundary:
            s = "".join(buf).strip()
            if s:
                sents.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return sents


def tail_sentences(sentences: list[str], overlap: int) -> str:
    """§4-3 整句后缀:总长 ≤1.5×overlap、优先 ≥0.5×overlap;凑不出 → ""(宁缺毋滥)。"""
    best = ""
    for n in range(1, len(sentences) + 1):
        cand = "".join(sentences[-n:])
        if len(cand) > overlap * 1.5:
            break
        best = cand
    return best if len(best) >= overlap * 0.5 else ""


def split_body(text: str, chunk_size: int, _level: int = 0) -> list[str]:
    """§4-2 递归降档:段落 → 行 → 句 → 硬切兜底(单句超长极端情形)。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if _level >= 3:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep = {0: "\n\n", 1: "\n"}.get(_level)  # _level==2 → None → 句级
    parts = split_sentences(text) if sep is None else text.split(sep)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return split_body(text, chunk_size, _level + 1)
    joiner = "" if sep is None else sep
    expanded: list[str] = []  # 单 parts 仍超限 → 降一级继续切
    for p in parts:
        if len(p) > chunk_size:
            expanded.extend(split_body(p, chunk_size, _level + 1))
        else:
            expanded.append(p)
    out: list[str] = []  # 贪心装箱
    cur: list[str] = []
    for p in expanded:
        if cur and len(joiner.join(cur + [p])) > chunk_size:
            out.append(joiner.join(cur))
            cur = [p]
        else:
            cur.append(p)
    if cur:
        out.append(joiner.join(cur))
    return out


def split_table(table_lines: list[str], chunk_size: int) -> list[str]:
    """§4-4 表按行分组,每块复制表头行+分隔行;单行超限也整行成块(行是原子)。"""
    if len(table_lines) <= 2:
        return ["\n".join(table_lines)]
    h2 = "\n".join(table_lines[:2])
    out: list[str] = []
    cur: list[str] = []
    for row in table_lines[2:]:
        if cur and len("\n".join([h2, *cur, row])) > chunk_size:
            out.append("\n".join([h2, *cur]))
            cur = [row]
        else:
            cur.append(row)
    if cur:
        out.append("\n".join([h2, *cur]))
    return out


def split_blocks(text: str, chunk_size: int) -> list[str]:
    """连续 | 行 = 表块走 split_table;其余文字走 split_body;保序。"""
    out: list[str] = []
    buf: list[str] = []
    tbl: list[str] = []

    def flush_text() -> None:
        if buf:
            out.extend(split_body("\n".join(buf), chunk_size))
            buf.clear()

    def flush_table() -> None:
        if tbl:
            out.extend(split_table(list(tbl), chunk_size))
            tbl.clear()

    for ln in text.splitlines():
        if ln.strip().startswith("|"):
            flush_text()
            tbl.append(ln)
        else:
            flush_table()
            buf.append(ln)
    flush_text()
    flush_table()
    return out


def iter_sections(body: str):
    """§4-1 标题栈切 section。yield (titles, is_key, section_body);
    标题「（关键条款）」后缀剥离并标记;标记对后代节传导;无正文的容器节(如纯 H1)不出块。
    """
    stack: list[tuple[int, str, bool]] = []
    buf: list[str] = []

    def flush() -> None:
        if any(l.strip() for l in buf):
            yield_ = ([t for _, t, _ in stack], any(k for _, _, k in stack), "\n".join(buf).strip())
            _out.append(yield_)

    # 用列表收集再返回生成器等价物,避免闭包 yield 的复杂度(函数仍是纯的)
    _out: list[tuple[list[str], bool, str]] = []
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            raw = m.group(2).strip()
            key = raw.endswith(_KEY_SUFFIX)
            title = raw[: -len(_KEY_SUFFIX)] if key else raw
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title, key))
            buf = []
        else:
            buf.append(line)
    flush()
    yield from _out


def parse_faq_qa(section_body: str) -> tuple[list[tuple[list[str], str]], str]:
    """§4-8: **Q： 行开组、连续 Q 并入 questions;**A： 行至下一个 Q/标题/表格 = answer。
    返回 ([(questions, answer)], 非问答残留文本)。表格/标题行会先闭合未决组再进残留。
    """
    qa: list[tuple[list[str], str]] = []
    questions: list[str] = []
    answer: list[str] = []
    other: list[str] = []

    def clean(s: str) -> str:
        if s.startswith("**"):
            s = s[2:]
        if s.endswith("**"):
            s = s[:-2]
        return s.strip()

    def close() -> None:
        nonlocal questions, answer
        if questions and answer:
            qa.append((questions, "\n".join(answer).strip()))
        questions, answer = [], []

    for line in section_body.splitlines():
        s = line.strip()
        c = clean(s)
        if s.startswith("**Q"):
            if answer:  # 连续 Q 并入当前组(§4-8);上一组已开答才算结束才闭合
                close()
            questions.append(_Q_PREFIX.sub("", c))
        elif s.startswith("**A"):
            answer.append(_A_PREFIX.sub("", c))
        elif s.startswith("|") or s.startswith("#"):
            close()
            other.append(line)
        elif s and answer:
            answer.append(s)
        elif s:
            other.append(s)
    close()
    return qa, "\n".join(other)


def apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """§4-3: 前块「原始末尾」的整句后缀前置到后块开头(不级联,重叠不会滚雪球)。
    表块行无句末标点 → tail 凑不出 → 天然不加重叠。
    """
    if not pieces:
        return []
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = tail_sentences(split_sentences(prev), overlap)
        out.append(tail + cur if tail else cur)
    return out


def split_markdown(text: str, *, chunk_size: int = 500, chunk_overlap: int = 80) -> list[ChunkDraft]:
    meta, body = parse_front_matter(text)
    ctype = meta.get("content_type", "policy")
    drafts: list[ChunkDraft] = []
    for titles, key, section_body in iter_sections(body):
        section_path = " > ".join(titles)
        leaf = titles[-1]
        category = " > ".join(titles[:-1]) or titles[0]  # §4-5: policy/manual 的 category=上级路径,顶级兜底自身
        leftovers = section_body
        if ctype == "faq":
            groups, leftovers = parse_faq_qa(section_body)
            for questions, answer in groups:
                drafts.append(
                    ChunkDraft(
                        category=leaf,  # §4-5: faq 的 category=H2 节名(商品分类)
                        questions="\n".join(questions),
                        answer=answer,
                        section_path=section_path,
                        content_type="faq",
                        is_key_clause=key,
                    )
                )
        for piece in apply_overlap(split_blocks(leftovers, chunk_size), chunk_overlap):
            drafts.append(
                ChunkDraft(
                    category=category,
                    questions=leaf,  # §4-5: 无天然问法 → questions=所在节标题
                    answer=piece,
                    section_path=section_path,
                    content_type=ctype,
                    is_key_clause=key,
                )
            )
    return drafts
