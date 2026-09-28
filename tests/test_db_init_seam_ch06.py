"""终审 Important#1 回归:db/init DDL 与 SQLAlchemy 枚举的接缝。

docker compose down -v && up 按 05/06 文件重建库——若 SQL 面 lcq_source ENUM 少值,
闸池写入回到「MySQL 1265 被 refusals 吞成 WARN=静默丢行」,而 Python 侧枚举测试全绿
看不见。本测把这类「套件结构性盲区」bug 转成单元断言:init SQL 里 source 列的
ENUM 值并集 ⊇ models 定义。
"""

import re
from pathlib import Path

from app.db.models import LowConfidenceQuestion

INIT_DIR = Path(__file__).resolve().parent.parent / "db" / "init"


def _ddl_source_enum_values() -> set[str]:
    vals: set[str] = set()
    for sql in sorted(INIT_DIR.glob("*.sql")):
        text = sql.read_text(encoding="utf-8")
        for m in re.finditer(r"source\s+(?:\w+\s+)?ENUM\(([^)]*)\)", text, re.I):
            vals |= set(re.findall(r"'([^']+)'", m.group(1)))
    return vals


def test_lcq_source_ddl_covers_model_enum():
    ddl = _ddl_source_enum_values()
    model = set(LowConfidenceQuestion.source.type.enums)
    assert ddl, "db/init 里找不到 source ENUM 定义(文件被改写?)"
    assert model <= ddl, f"fresh 环境 DDL 缺值 → 池写静默丢行: 缺 {model - ddl}"
