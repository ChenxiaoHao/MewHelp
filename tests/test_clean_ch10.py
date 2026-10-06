"""ch10 T3:清洗钉——脱敏(手机/长数字/地址→占位符)+格式规范+幂等。

错别字不在本模块(拍板 2:并入 T5 预标调用)。短数字(尺码/价格)不得误伤。
"""

from finetune.clean import clean, desensitize, normalize


def test_phone_to_placeholder():
    assert clean("我的电话13812345678麻烦回一下") == "我的电话<PHONE>麻烦回一下"


def test_long_number_masked_short_kept():
    assert "<NUM>" in clean("订单号123456789012345到哪了")
    assert clean("裤子腰围90cm") == "裤子腰围90cm", "短数字不误伤"


def test_address_placeholder():
    out = clean("收货地址是浙江省杭州市西湖区文一西路123号,帮我改")
    assert "<ADDR>" in out and "浙江省" not in out


def test_normalize_fullwidth_and_whitespace():
    assert normalize("　偏Ａ　大小") == "偏A 大小"
    assert clean("电话１３８１２３４５６７８　　急") == "电话<PHONE> 急", "双全角空格折叠为一个半角空格"


def test_idempotent_and_empty():
    for s in ("我的电话13812345678", "订单号123456789012345到哪了", "正常一句喵"):
        assert clean(clean(s)) == clean(s)
    assert clean("") == ""


def test_desensitize_normalize_compose():
    assert desensitize(normalize("１３９１２３４５６７８")) == "<PHONE>"
