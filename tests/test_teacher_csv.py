"""解析器与指标纯函数。样例行逐字抄自 evals/run_rag.py(附录 B 语法);真文件全量对账。"""

from evals.teacher_csv import eval_question, load_questions, parse_expect_section

SAMPLE = """id,桶(bucket),问题(query),期望章节(expect_section),标准要点(expect_points),应拒答(should_refuse)
A5,A_policy,"满多少钱包邮,不满怎么收运费",运费怎么算,满 99 元包邮 | 10 元运费,否
A23,A_policy,"新疆西藏内蒙的运费是多少,能参与包邮吗",运费与包邮 | 配送范围,偏远地区,否
E2,E_multi,"东西不想要了要退货,寄回去的运费谁承担,钱多久回来",退货政策 | 适用范围 + 退换货运费承担 + 退款时效,7 天无理由 | 无理由退货的退回运费由买家承担 | 原路退回,否
D3,D_absent,能不能开纸质发票邮寄给我,,,是
"""


def test_parse_expect_section_grammar():
    assert parse_expect_section("运费与包邮 | 配送范围") == [["运费与包邮", "配送范围"]]  # 组内 OR
    assert parse_expect_section("A + B") == [["A"], ["B"]]                                # 组间 AND
    assert parse_expect_section("会员 / 积分规则") == [["会员 > 积分规则"]]                 # 路径分隔符归一
    assert parse_expect_section("") == [] and parse_expect_section(None) == []


def test_load_sample_rows(tmp_path):
    f = tmp_path / "q.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    qs = load_questions(f)
    assert len(qs) == 4
    assert qs[0].query == "满多少钱包邮,不满怎么收运费"  # 引号字段内逗号不拆
    assert qs[0].groups == [["运费怎么算"]] and qs[0].should_refuse is False
    assert qs[2].groups == [["退货政策", "适用范围"], ["退换货运费承担"], ["退款时效"]]  # |与+混排
    assert qs[3].should_refuse is True and qs[3].groups == [] and qs[3].bucket == "D_absent"


def test_real_teacher_file_shape():
    qs = load_questions()  # 默认 evals/run_rag.py
    assert len(qs) == 300
    from collections import Counter

    c = Counter(q.bucket for q in qs)
    assert set(c) == {"A_policy", "B_model", "C_colloquial", "D_absent", "E_multi"}
    assert all(v == 60 for v in c.values())
    assert all(q.should_refuse for q in qs if q.bucket == "D_absent")
    assert all(not q.should_refuse for q in qs if q.bucket != "D_absent")


def test_eval_question_metrics_group_semantics():
    groups = [["运费"], ["包邮"]]
    m = eval_question(groups, ["x > 退货", "y > 包邮与运费", "z"], ks=(3, 10))
    assert m["hit@3"] == 1.0 and m["recall@3"] == 1.0
    assert abs(m["mrr@10"] - 0.5) < 1e-9  # 两组都 rank2 → (1/2+1/2)/2


def test_eval_question_window_and_empty():
    m = eval_question([["深层"]], ["a", "b深层c"], ks=(1, 10))
    assert m["hit@1"] == 0.0 and m["recall@1"] == 0.0 and m["hit@10"] == 1.0
    assert eval_question([], ["anything"]) == {}  # D 桶不参与排序指标
    assert eval_question([["x"]], [])["mrr@10"] == 0.0
