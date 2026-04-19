from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.api.deps import get_answer_service
from app.schemas.request import ChatRequest


REQUIRED_CASES = [
    {"question": "阿里巴巴当前股价是多少？", "expected_intent": "asset_price"},
    {"question": "BABA 最近 7 天涨跌情况如何？", "expected_intent": "asset_trend"},
    {"question": "阿里巴巴最近为何1月15日大涨？", "expected_intent": "asset_event_analysis"},
    {"question": "特斯拉近期走势如何？", "expected_intent": "asset_trend"},
    {"question": "什么是市盈率？", "expected_intent": "finance_knowledge"},
    {"question": "收入和净利润的区别是什么？", "expected_intent": "finance_knowledge"},
    {"question": "腾讯最近季度财报摘要是什么？", "expected_intent": "report_summary"},
]

VARIANT_CASES = [
    {"question": "阿里现在多少钱？", "expected_intent": "asset_price"},
    {"question": "TSLA 这周表现怎么样？", "expected_intent": "asset_trend"},
    {"question": "英伟达 2026-01-15 为什么涨？", "expected_intent": "asset_event_analysis"},
    {"question": "PE ratio 是什么？", "expected_intent": "finance_knowledge"},
    {"question": "苹果最近季度业绩摘要是什么？", "expected_intent": "report_summary"},
]


def validate_response(payload: dict) -> list[str]:
    errors: list[str] = []
    if not payload.get("summary"):
        errors.append("missing_summary")
    if "objective_data" not in payload:
        errors.append("missing_objective_data")
    if "analysis" not in payload or not payload["analysis"]:
        errors.append("missing_analysis")
    if payload["question_type"] in {"asset_price", "asset_trend", "asset_event_analysis"} and "symbol" not in payload["objective_data"]:
        errors.append("missing_symbol")
    if payload["question_type"] in {"finance_knowledge", "report_summary"} and "retrieval_enabled" not in payload["objective_data"]:
        errors.append("missing_retrieval_flag")
    return errors


def main() -> None:
    answer_service = get_answer_service()
    cases = {"required": REQUIRED_CASES, "variants": VARIANT_CASES}
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": {},
    }

    for group_name, group_cases in cases.items():
        outputs = []
        for case in group_cases:
            result = answer_service.answer(ChatRequest(message=case["question"]))
            payload = result.model_dump(mode="json")
            checks = validate_response(payload)
            if payload["question_type"] != case["expected_intent"]:
                checks.append(f"unexpected_intent:{payload['question_type']}")

            outputs.append(
                {
                    "question": case["question"],
                    "expected_intent": case["expected_intent"],
                    "actual_intent": payload["question_type"],
                    "passed": not checks,
                    "checks": checks,
                    "summary": payload["summary"],
                    "source_count": len(payload["sources"]),
                    "source_names": [source["name"] for source in payload["sources"][:4]],
                }
            )
        report["cases"][group_name] = outputs

    target_dir = Path(__file__).resolve().parents[1] / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "phase6-eval.json"
    target_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report to {target_path}")


if __name__ == "__main__":
    main()
