"""ch04 忠实度评估(spec §6.3):真实链路生成 → LLM-as-judge → fabricated 落 faith_cases。

uv run python evals/run_faith_eval.py [--sample 30] [--d-limit 60] [--seed 42] [--recheck 0]
- 可答桶按桶分层抽 --sample 题,每题走完整 stream_chat_with_tools(query_faq 真检索+双闸;
  persister=None 零消息落库,conversation_id 用本次创建的 eval_faith 会话 → 池写可落);
- REFUSAL_ANSWER 出口的答案(含 round-1 前言)计「显式拒答」不送判;其余全部送判(证据=本轮帧 citations,温度 0 可复现);
- D 桶全量测正确拒答率;未拒答且判 fabricated → 台账 bucket='D_absent'(验收④反向证据);
- --recheck N:台账最近 N 条「已解决」复判,观察处置稳定性(只打印不自动翻状态);
- 报告 evals/reports/ch04_faith_report.md(重跑覆盖)+ 末尾打印低置信池本次增量。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db import crud  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import LowConfidenceQuestion  # noqa: E402
from app.prompts.faith_judge import FAITH_JUDGE_SYSTEM  # noqa: E402
from app.schemas.chat import ChatMessage  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402
from app.services.refusals import REFUSAL_ANSWER  # noqa: E402
from app.services.tool_chat_service import stream_chat_with_tools  # noqa: E402
from evals.teacher_csv import load_questions  # noqa: E402

REPORT_FILE = Path("evals/reports/ch04_faith_report.md")
ANSWER_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")


class FaithVerdict(BaseModel):
    verdict: Literal["faithful", "fabricated"]
    reason: str = ""


def sample_answerable(questions, per_bucket: int, seed: int) -> list:
    by: dict[str, list] = defaultdict(list)
    for q in questions:
        if q.bucket in ANSWER_BUCKETS:
            by[q.bucket].append(q)
    rng = random.Random(seed)
    out = []
    for b in ANSWER_BUCKETS:
        out.extend(rng.sample(by[b], min(per_bucket, len(by[b]))))
    return out


async def run_chain(question: str, st, conversation_id: int) -> tuple[str, list]:
    """真实链:token 聚合成答案,query_faq 帧的 citations 作评审证据。"""
    model = get_model(st)
    parts, citations = [], []
    async for evt, payload in stream_chat_with_tools(
        [ChatMessage(role="user", content=question)], st, model, conversation_id=conversation_id,
    ):
        if evt == "token":
            parts.append(payload)
        elif evt == "tool_result" and payload.get("name") == "query_faq":
            citations = payload.get("citations") or citations
    return "".join(parts), citations


def _evidence_block(citations: list) -> str:
    return "\n".join(
        f"[{c['n']}] {c.get('section_path', '')} | {c.get('question', '')} | {(c.get('answer') or '')[:400]}"
        for c in citations)


async def judge(question: str, answer: str, citations: list, st) -> FaithVerdict:
    key = st.faith_judge_model or st.model_name
    judge_st = st.model_copy(update={"model_name": key, "temperature": 0})
    model = get_model(judge_st).with_structured_output(FaithVerdict)
    user = (f"【问题】{question}\n【证据】\n{_evidence_block(citations) or '(本轮无知识库证据命中)'}"
            f"\n【候选回答】{answer}")
    return await asyncio.wait_for(
        model.ainvoke([("system", FAITH_JUDGE_SYSTEM), ("user", user)]), timeout=60)


# ---- T12 韧性增件(派工纪律:云失败单样本先重试一次,仍败记降级不拦跑;评审失败不毒化整跑) ----

CHAIN_TIMEOUT_SECONDS = 240.0  # 直连云可通但会整连接僵死(T12 冒烟实测挂 15+ 分钟无输出);
                                # 生产 get_model 无 request_timeout 且 app 层不可动 → 评估侧 wait_for 兜底


async def _capped(aw) -> object:
    return await asyncio.wait_for(aw, CHAIN_TIMEOUT_SECONDS)


async def _retry_once(fn: Callable[[], Awaitable]) -> object:
    """第一次瞬态失败(网络/超时/上游抖动)重试一次;仍败上抛,由调用方记降级。"""
    try:
        return await fn()
    except Exception:  # noqa: BLE001
        return await fn()


async def run(args) -> int:
    st = get_settings()
    init_engine(st)
    questions = load_questions()
    started = datetime.now()
    async with get_session_factory()() as session:
        conv = await crud.create_or_get_conversation(session, None, "eval_faith")
        cid = conv.id
        pool_before = (await session.execute(
            select(func.count()).select_from(LowConfidenceQuestion))).scalar()

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fabricated_rows: list[str] = []
    degrade_rows: list[str] = []  # 降级个案如实入报告(§0-9 如实出数,不静默吞)
    d_refused = d_total = 0
    recheck_lines: list[str] = []

    def dump() -> None:  # 每题落盘一次:断流被杀只丢当前步(报告重跑覆盖,末次即全量)
        lines = [
            "# ch04 忠实度评估报告",
            f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 抽样:每桶 {args.sample} + D 桶 {d_total}(seed={args.seed}) · 裁判:{st.faith_judge_model or st.model_name} · 用时:{(datetime.now() - started).seconds // 60} 分",
            "- 链路与生产一致(temperature=生产值);仅裁判温度 0 保可复现。REFUSAL 出口答案计显式拒答不送判(拒答出口恒为末段,谓词 endswith)。",
            "",
            "| 桶 | faithful | fabricated | 显式拒答 | 降级 |", "|---|---|---|---|---|",
        ]
        for b in ANSWER_BUCKETS + ("D_absent",):
            s = stats[b]
            lines.append(f"| {b} | {s['faithful']} | {s['fabricated']} | {s['refused']} | {s['degraded']} |")
        lines += [
            "",
            f"D 桶正确拒答率:{d_refused}/{d_total}(D3 老师标注出入见策略报告,只读如实呈现)",
            f"低置信池本次增量:{_pool_now - pool_before} 行(验收④证据链;演练另见 evals/demo_ch04.py)",
            "",
            "## fabricated 个案(已入 faith_cases 台账)", "",
        ] + (fabricated_rows or ["(无)"])
        if degrade_rows:
            lines += ["", "## 降级个案(重试一次仍败,未计入判决)", ""] + degrade_rows
        if recheck_lines:
            lines += ["", "## 复判抽验(已解决个案,仅观察)", ""] + recheck_lines
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    async def pool_now_get() -> int:  # dump 内同步取用 → 缓存最近一次实测值
        nonlocal _pool_now
        async with get_session_factory()() as session:
            _pool_now = (await session.execute(
                select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        return _pool_now

    _pool_now = pool_before
    try:
        for q in sample_answerable(questions, args.sample, args.seed):
            try:
                answer, citations = await _retry_once(lambda: _capped(run_chain(q.query, st, cid)))
            except Exception as exc:  # noqa: BLE001
                stats[q.bucket]["degraded"] += 1
                degrade_rows.append(f"- `{q.id}` [{q.bucket}] 链路失败(重试 1 次,{type(exc).__name__}):{q.query[:40]}")
                print(f"[faith] {q.id} → 降级(chain)", flush=True)
                await pool_now_get(); dump()
                continue
            if answer.endswith(REFUSAL_ANSWER):  # 固定拒答出口必为末段(L141/148 yield 后即 return)
                stats[q.bucket]["refused"] += 1
                print(f"[faith] {q.id} → 显式拒答", flush=True)
                await pool_now_get(); dump()
                continue
            try:
                v = await _retry_once(lambda: judge(q.query, answer, citations, st))
            except Exception as exc:  # noqa: BLE001
                stats[q.bucket]["degraded"] += 1
                degrade_rows.append(f"- `{q.id}` [{q.bucket}] 裁判失败(重试 1 次,{type(exc).__name__}):{q.query[:40]}")
                print(f"[faith] {q.id} → 降级(judge)", flush=True)
                await pool_now_get(); dump()
                continue
            stats[q.bucket][v.verdict] += 1
            if v.verdict == "fabricated":
                fabricated_rows.append(f"- `{q.id}` [{q.bucket}] {q.query}\n  - 裁判:{v.reason[:80]}\n  - 回答:{answer[:120]}")
                async with get_session_factory()() as session:
                    await crud.upsert_faith_case(
                        session, eval_id=q.id, bucket=q.bucket, query=q.query, answer=answer,
                        reason=v.reason, citations=citations, judge_model=st.faith_judge_model or st.model_name)
            print(f"[faith] {q.id} → {v.verdict}", flush=True)
            await pool_now_get(); dump()

        for q in [q for q in questions if q.bucket == "D_absent"][: args.d_limit]:
            d_total += 1
            try:
                answer, citations = await _retry_once(lambda: _capped(run_chain(q.query, st, cid)))
            except Exception as exc:  # noqa: BLE001
                stats["D_absent"]["degraded"] += 1
                degrade_rows.append(f"- `{q.id}` [D_absent] 链路失败(重试 1 次,{type(exc).__name__}):{q.query[:40]}")
                print(f"[faith] {q.id} D→降级(chain)", flush=True)
                await pool_now_get(); dump()
                continue
            if answer.endswith(REFUSAL_ANSWER):  # 固定拒答出口必为末段(L141/148 yield 后即 return)
                d_refused += 1
                stats["D_absent"]["refused"] += 1
                print(f"[faith] {q.id} D→显式拒答", flush=True)
                await pool_now_get(); dump()
                continue
            try:
                v = await _retry_once(lambda: judge(q.query, answer, citations, st))
            except Exception as exc:  # noqa: BLE001
                stats["D_absent"]["degraded"] += 1
                degrade_rows.append(f"- `{q.id}` [D_absent] 裁判失败(重试 1 次,{type(exc).__name__}):{q.query[:40]}")
                print(f"[faith] {q.id} D→降级(judge)", flush=True)
                await pool_now_get(); dump()
                continue
            stats["D_absent"][v.verdict] += 1
            if v.verdict == "fabricated":
                fabricated_rows.append(f"- `{q.id}` [D_absent 应拒未拒] {q.query}\n  - 裁判:{v.reason[:80]}\n  - 回答:{answer[:120]}")
                async with get_session_factory()() as session:
                    await crud.upsert_faith_case(
                        session, eval_id=q.id, bucket="D_absent", query=q.query, answer=answer,
                        reason="应拒答未拒:" + v.reason, citations=citations,
                        judge_model=st.faith_judge_model or st.model_name)
            print(f"[faith] {q.id} D→{v.verdict}", flush=True)
            await pool_now_get(); dump()

        if args.recheck:
            async with get_session_factory()() as session:
                rows = [r for r in await crud.list_faith_cases(session, status="已解决")][: args.recheck]
            for r in rows:
                try:
                    v = await _retry_once(lambda r=r: judge(r.query, r.answer, r.citations or [], st))
                    recheck_lines.append(f"- 复判 `{r.eval_id}`:{v.verdict} | {v.reason[:60]}")
                except Exception:  # noqa: BLE001
                    recheck_lines.append(f"- 复判 `{r.eval_id}`:裁判失败,记降级(不翻状态)")

        await pool_now_get()
        dump()
        print(f"[faith] 报告 → {REPORT_FILE}")
        return 0
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ch04 忠实度评估")
    parser.add_argument("--sample", type=int, default=30, help="每答桶抽样题数(四臂成本闸)")
    parser.add_argument("--d-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recheck", type=int, default=0, help="复判 N 条已解决个案")
    sys.exit(asyncio.run(run(parser.parse_args())))
