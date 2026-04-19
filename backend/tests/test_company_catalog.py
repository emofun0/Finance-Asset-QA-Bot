from app.core.company_catalog import find_company_profile


def test_find_company_profile_supports_alibaba_aliases() -> None:
    assert find_company_profile(company="Alibaba Group Holding Limited").symbol == "BABA"
    assert find_company_profile(symbol="9988.HK").canonical_name == "Alibaba"
    assert find_company_profile(company="阿里巴巴集团").symbol == "BABA"


def test_find_company_profile_supports_tencent_aliases() -> None:
    assert find_company_profile(company="腾讯").symbol == "0700.HK"
    assert find_company_profile(company="Tencent Holdings Limited").canonical_name == "Tencent"
