from app.core.company_catalog import find_company_profile


def test_find_company_profile_matches_canonical_name_or_symbol() -> None:
    assert find_company_profile(company="Alibaba").symbol == "BABA"
    assert find_company_profile(symbol="BABA").canonical_name == "Alibaba"


def test_find_company_profile_matches_aliases() -> None:
    assert find_company_profile(company="腾讯").canonical_name == "Tencent"
    assert find_company_profile(company="腾讯控股").symbol == "0700.HK"
    assert find_company_profile(company="Tencent Holdings Limited").canonical_name == "Tencent"
