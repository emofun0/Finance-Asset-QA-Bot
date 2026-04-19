from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyProfile:
    canonical_name: str
    symbol: str
    country_group: str
    official_domains: tuple[str, ...]
    aliases: tuple[str, ...] = ()


_COMPANY_CATALOG = (
    CompanyProfile(
        canonical_name="Tencent",
        symbol="0700.HK",
        country_group="china",
        official_domains=("tencent.com", "static.www.tencent.com"),
    ),
    CompanyProfile(
        canonical_name="ICBC",
        symbol="1398.HK",
        country_group="china",
        official_domains=("icbc-ltd.com",),
    ),
    CompanyProfile(
        canonical_name="Agricultural Bank of China",
        symbol="601288.SS",
        country_group="china",
        official_domains=("abchina.com",),
    ),
    CompanyProfile(
        canonical_name="China Construction Bank",
        symbol="601939.SS",
        country_group="china",
        official_domains=("ccb.com",),
    ),
    CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        official_domains=("alibabagroup.com",),
    ),
    CompanyProfile(
        canonical_name="PetroChina",
        symbol="0857.HK",
        country_group="china",
        official_domains=("petrochina.com.cn", "petrochina.com"),
    ),
    CompanyProfile(
        canonical_name="CATL",
        symbol="300750.SZ",
        country_group="china",
        official_domains=("catl.com",),
    ),
    CompanyProfile(
        canonical_name="Bank of China",
        symbol="601988.SS",
        country_group="china",
        official_domains=("boc.cn",),
    ),
    CompanyProfile(
        canonical_name="Moutai",
        symbol="600519.SH",
        country_group="china",
        official_domains=("moutaichina.com",),
    ),
    CompanyProfile(
        canonical_name="China Mobile",
        symbol="0941.HK",
        country_group="china",
        official_domains=("chinamobileltd.com",),
    ),
    CompanyProfile(
        canonical_name="NVIDIA",
        symbol="NVDA",
        country_group="usa",
        official_domains=("nvidia.com",),
    ),
    CompanyProfile(
        canonical_name="Alphabet",
        symbol="GOOG",
        country_group="usa",
        official_domains=("abc.xyz", "google.com"),
    ),
    CompanyProfile(
        canonical_name="Apple",
        symbol="AAPL",
        country_group="usa",
        official_domains=("apple.com",),
    ),
    CompanyProfile(
        canonical_name="Intel",
        symbol="INTC",
        country_group="usa",
        official_domains=("intel.com",),
    ),
    CompanyProfile(
        canonical_name="Microsoft",
        symbol="MSFT",
        country_group="usa",
        official_domains=("microsoft.com",),
    ),
    CompanyProfile(
        canonical_name="Amazon",
        symbol="AMZN",
        country_group="usa",
        official_domains=("amazon.com",),
    ),
    CompanyProfile(
        canonical_name="Broadcom",
        symbol="AVGO",
        country_group="usa",
        official_domains=("broadcom.com",),
    ),
    CompanyProfile(
        canonical_name="Meta",
        symbol="META",
        country_group="usa",
        official_domains=("meta.com", "atmeta.com"),
    ),
    CompanyProfile(
        canonical_name="Tesla",
        symbol="TSLA",
        country_group="usa",
        official_domains=("tesla.com", "sec.gov"),
    ),
    CompanyProfile(
        canonical_name="Berkshire Hathaway",
        symbol="BRK-B",
        country_group="usa",
        official_domains=("berkshirehathaway.com",),
    ),
    CompanyProfile(
        canonical_name="Walmart",
        symbol="WMT",
        country_group="usa",
        official_domains=("walmart.com",),
    ),
    CompanyProfile(
        canonical_name="Xiaomi",
        symbol="1810.HK",
        country_group="china",
        official_domains=("mi.com", "hkexnews.hk"),
    ),
    CompanyProfile(
        canonical_name="China Merchants Bank",
        symbol="600036.SH",
        country_group="china",
        official_domains=("cmbchina.com",),
    ),
)


def get_company_catalog() -> tuple[CompanyProfile, ...]:
    return _COMPANY_CATALOG


def find_company_profile(company: str | None = None, symbol: str | None = None) -> CompanyProfile | None:
    normalized_company = company.strip().lower() if company else None
    normalized_symbol = symbol.strip().lower() if symbol else None
    for profile in _COMPANY_CATALOG:
        if normalized_company and profile.canonical_name.lower() == normalized_company:
            return profile
        if normalized_symbol and profile.symbol.lower() == normalized_symbol:
            return profile
    return None
