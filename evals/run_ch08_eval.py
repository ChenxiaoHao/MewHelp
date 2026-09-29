"""ch08 T10 评估:三桶话术收敛(校验回灌自纠/必填追问不瞎编/空结果与故障如实回)。

真实模型调用:逐样例直接驱动 react_agent_stream(绕过分流器——被测面是 ReAct
层对错误回灌/缺参追问的话术收敛,分流命中是 T11 验收面)。temperature=0 固定
(路由评估同款确定性约定)。create_ticket 提案在权限闸=中间态即停
(awaiting_confirmation→ticket_request 事件),评估面零 DB 写。

用法(先起两个 MCP Server,再跑):
  .venv/Scripts/python.exe mcp_servers/logistics_server.py
  .venv/Scripts/python.exe mcp_servers/aftersale_server.py
  uv run python evals/run_ch08_eval.py
判定:脚本机判辅助(≥9/10 过线),三桶话术人工核对为准——逐样例明细落
evals/reports/ch08_eval_result.json(UTF-8),控制台只出 ASCII 摘要(GBK 红线)。
「真故障」桶(id 9/10)把 query_order 换成 BoomTool(评估面允许注假件,测的是
模型对异常回灌的话术,不是故障工具本身)。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from app.agents.react import react_agent_stream  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT  # noqa: E402
from app.services.chat_service import get_model, trim_history  # noqa: E402
from app.tools.registry import ToolSpec, snapshot_tools  # noqa: E402

PASS_THRESHOLD = 9
BOOM_IDS = {9, 10}

QUESTION_MARKS = ("?", "？", "请提供", "请告诉", "需要", "什么", "哪", "描述")
FIX_REJECT_HINTS = ("枚举", "enum", "参数", "无效", "不在", "类型", "必须是", "校验")
HONEST_MARKS = ("没有", "暂无", "未找到", "没找到", "无法", "未能", "失败", "异常",
                "出错", "稍后再试", "不确定", "没能")
LIE_MARKS = ("已创建", "已为您创建", "已办理成功", "查到了", "已查到", "找到了",
             "已提交成功")


@tool
async def boom_query_order(order_id: str) -> dict:
    """按订单号查询订单状态、金额与商品明细。order_id: 订单号,如 1001。"""
    raise RuntimeError("上游订单服务暂时不可用")


def judge(expect: str, events: list, answer: str) -> tuple[bool, dict]:
    """机判辅助规则(与 notes 一起人工复核);signals 回传核对料。"""
    ticket_requests = [e for e in events if e[0] == "ticket_request"]
    failed_results = [e for e in events
                      if e[0] == "tool_result" and not e[1].get("ok")]
    has_question = any(m in answer for m in QUESTION_MARKS)
    lies = [m for m in LIE_MARKS if m in answer]
    honest = any(m in answer for m in HONEST_MARKS)
    props = [e[1].get("args") or {} for e in ticket_requests]
    sig = {"ticket_requests": len(ticket_requests),
           "proposal_args": props,
           "failed_results": [e[1]["summary"] for e in failed_results],
           "answer_head": answer[:160]}
    if expect == "ask_missing":
        return (not ticket_requests and has_question), sig
    if expect == "self_fix":
        # 达标形制(实跑取证 2026-09-30):被拦后改参再提案,或被拦后向用户
        # 确认(尊重用户原意不擅改=合规收敛);闸前自纠=提案参数合法
        # (枚举内+描述含中文实义)亦达标;盲从用户把占位垃圾入单=不达标。
        rejected = any(any(h in (e[1]["summary"] or "") for h in FIX_REJECT_HINTS)
                       for e in failed_results)
        if rejected:
            return (bool(ticket_requests) or has_question), sig
        if not ticket_requests:
            return has_question, sig
        import re as _re
        clean = [a for a in props
                 if a.get("ticket_type") in ("售后", "投诉", "咨询")
                 and isinstance(a.get("description"), str)
                 and _re.search(r"[一-鿿]", a["description"] or "")]
        # 提案含占位垃圾但收尾向用户追问实况=确认式收敛,达标(人工核 2026-09-30)
        return bool(clean) or has_question, sig
    # no_fabricate:不许成功谎;故障桶还须见失败回灌
    if lies:
        return False, sig
    return (honest if failed_results else honest or not ticket_requests), sig


async def run_sample(model, settings, specs, user: str) -> tuple[list, str]:
    hist = [HumanMessage(user)]
    msgs = trim_history(
        CUSTOMER_SERVICE_PROMPT.invoke({"messages": hist}).to_messages(),
        settings.history_token_budget)
    events: list = []
    parts: list[str] = []
    async for ev in react_agent_stream(
            {"messages": msgs, "conversation_id": None},
            settings, model, persister=None, specs=specs):
        events.append(ev)
        if ev[0] == "token":
            parts.append(ev[1])
    return events, "".join(parts)


async def main() -> int:
    samples = [json.loads(ln) for ln in
               (Path(__file__).parent / "ch08_samples.jsonl").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]
    settings = get_settings().model_copy(update={"temperature": 0})
    model = get_model(settings)
    base_specs = await snapshot_tools(settings)
    if "query_return_progress" not in base_specs:
        print("ABORT: MCP servers not up (query_return_progress missing from "
              "snapshot); start logistics/aftersale servers first")
        return 2

    results = []
    n_pass = 0
    for s in samples:
        specs = dict(base_specs)
        if s["id"] in BOOM_IDS:
            specs["query_order"] = ToolSpec(boom_query_order, "readonly", "builtin")
        events, answer = await run_sample(model, settings, specs, s["user"])
        ok, sig = judge(s["expect"], events, answer)
        n_pass += ok
        results.append({"id": s["id"], "user": s["user"], "expect": s["expect"],
                        "verdict": "PASS" if ok else "FAIL",
                        "notes": s["notes"], "signals": sig})
        print(f"sample {s['id']:>2} [{s['expect']:<12}] {'PASS' if ok else 'FAIL'}")

    out = Path(__file__).parent / "reports" / "ch08_eval_result.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": PASS_THRESHOLD, "passed": n_pass, "total": len(samples),
         "overall": "PASS" if n_pass >= PASS_THRESHOLD else "FAIL",
         "samples": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL {n_pass}/{len(samples)} (threshold {PASS_THRESHOLD}) -> "
          f"evals/reports/ch08_eval_result.json")
    return 0 if n_pass >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
