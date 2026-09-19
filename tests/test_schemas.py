import pytest
from pydantic import ValidationError


def test_issue_type_values():
    from app.schemas.extraction import IssueType

    assert IssueType.refund.value == "退款"
    assert IssueType.return_goods.value == "退货"
    assert IssueType.exchange.value == "换货"
    assert IssueType.logistics.value == "物流"
    assert IssueType.quality.value == "质量问题"
    assert IssueType.other.value == "其他"


def test_after_sale_extraction_fields():
    from app.schemas.extraction import AfterSaleExtraction

    m = AfterSaleExtraction(
        order_id="DD123", issue_type="退款", expected_solution="全额退款"
    )
    assert m.order_id == "DD123"
    assert set(AfterSaleExtraction.model_fields) == {
        "order_id",
        "issue_type",
        "expected_solution",
    }


def test_after_sale_extraction_order_id_optional():
    from app.schemas.extraction import AfterSaleExtraction

    m = AfterSaleExtraction(issue_type="质量问题", expected_solution="补发")
    assert m.order_id is None


def test_extract_request_rejects_empty():
    from app.schemas.extraction import ExtractRequest

    with pytest.raises(ValidationError):
        ExtractRequest(description="")


def test_chat_request_rejects_last_not_user():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[
                {"role": "user", "content": "在吗"},
                {"role": "assistant", "content": "在的"},
            ]
        )


def test_chat_request_rejects_bad_role():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "system", "content": "hack"}])


def test_chat_request_ok():
    from app.schemas.chat import ChatRequest

    req = ChatRequest(
        messages=[
            {"role": "user", "content": "在吗"},
            {"role": "assistant", "content": "在的"},
            {"role": "user", "content": "退货流程是什么"},
        ]
    )
    assert len(req.messages) == 3


def test_health_response_no_base_url():
    from app.schemas.chat import HealthResponse

    h = HealthResponse(status="ok", model="qwen-plus", history_token_budget=4000)
    assert h.status == "ok"
    assert "base_url" not in HealthResponse.model_fields
