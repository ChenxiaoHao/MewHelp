"""ch10 T2:17 类权威术语表钉——名单/顺序/边界说明/近邻三对/同义词容量。

glossary 是标注、造数、增强、id2label 四处共用的唯一词表(CLAUTES 序定死);
此测试漂移即全链红。
"""

import pytest

from finetune.glossary import CLASSES, assert_valid_labels, load_glossary


def test_classes_exact_17_ordered():
    assert CLASSES == ["退换货", "物流", "尺码", "发票", "质量问题", "运费",
                       "优惠活动", "价保", "支付", "订单修改", "库存补货",
                       "商品信息", "保修维修", "账号", "会员积分", "评价", "其他"]
    assert CLASSES[0] in ("退换货",) and CLASSES[-1] == "其他"


def test_every_class_has_boundary_and_synonyms():
    g = load_glossary()
    assert set(g) == set(CLASSES)
    for name, meta in g.items():
        assert meta["boundary"].strip(), name
        assert len(meta["synonyms"]) >= 3, f"{name} 同义词不足 3"


def test_neighbor_boundaries_verbatim():
    g = load_glossary()
    assert "修归保修维修" in g["退换货"]["boundary"] or "修归保修维修" in g["保修维修"]["boundary"]
    assert ("运费管钱" in g["运费"]["boundary"] and "物流管货" in g["物流"]["boundary"]
            ) or "运费管钱、物流管货" in (g["运费"]["boundary"] + g["物流"]["boundary"])
    assert "补差价" in g["价保"]["boundary"]
    assert "券" in g["优惠活动"]["boundary"] and "满减" in g["优惠活动"]["boundary"]


def test_assert_valid_labels():
    assert assert_valid_labels(["发票", "发票"]) == ["发票"]
    assert assert_valid_labels(["发票", "退换货"]) == ["发票", "退换货"]
    with pytest.raises(ValueError):
        assert_valid_labels(["物流", "配送"])  # 越类名即炸,白名单是唯一入口
    with pytest.raises(ValueError):
        assert_valid_labels([])
