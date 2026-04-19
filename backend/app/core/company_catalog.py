from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyProfile:
    canonical_name: str
    symbol: str
    country_group: str
    aliases: tuple[str, ...]
    official_domains: tuple[str, ...]


_COMPANY_CATALOG = (
    CompanyProfile(
        canonical_name="Tencent",
        symbol="0700.HK",
        country_group="china",
        aliases=("腾讯", "tencent", "0700.hk", "tcehy"),
        official_domains=("tencent.com", "static.www.tencent.com"),
    ),
    CompanyProfile(
        canonical_name="ICBC",
        symbol="1398.HK",
        country_group="china",
        aliases=("工商银行", "icbc", "1398.hk"),
        official_domains=("icbc-ltd.com",),
    ),
    CompanyProfile(
        canonical_name="Agricultural Bank of China",
        symbol="601288.SS",
        country_group="china",
        aliases=("农业银行", "农行", "agricultural bank of china", "abc", "601288.ss"),
        official_domains=("abchina.com",),
    ),
    CompanyProfile(
        canonical_name="China Construction Bank",
        symbol="601939.SS",
        country_group="china",
        aliases=("建设银行", "建行", "china construction bank", "ccb", "601939.ss"),
        official_domains=("ccb.com",),
    ),
    CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        aliases=("阿里", "阿里巴巴", "alibaba", "baba"),
        official_domains=("alibabagroup.com",),
    ),
    CompanyProfile(
        canonical_name="PetroChina",
        symbol="0857.HK",
        country_group="china",
        aliases=("中国石油", "petrochina", "0857.hk"),
        official_domains=("petrochina.com.cn", "petrochina.com"),
    ),
    CompanyProfile(
        canonical_name="CATL",
        symbol="300750.SZ",
        country_group="china",
        aliases=("宁德时代", "catl", "300750.sz"),
        official_domains=("catl.com",),
    ),
    CompanyProfile(
        canonical_name="Bank of China",
        symbol="601988.SS",
        country_group="china",
        aliases=("中国银行", "中行", "bank of china", "601988.ss"),
        official_domains=("boc.cn",),
    ),
    CompanyProfile(
        canonical_name="Moutai",
        symbol="600519.SH",
        country_group="china",
        aliases=("贵州茅台", "茅台", "kweichow moutai", "moutai", "600519.sh"),
        official_domains=("moutaichina.com",),
    ),
    CompanyProfile(
        canonical_name="China Mobile",
        symbol="0941.HK",
        country_group="china",
        aliases=("中国移动", "china mobile", "0941.hk"),
        official_domains=("chinamobileltd.com",),
    ),
    CompanyProfile(
        canonical_name="NVIDIA",
        symbol="NVDA",
        country_group="usa",
        aliases=("英伟达", "nvidia", "nvda"),
        official_domains=("nvidia.com",),
    ),
    CompanyProfile(
        canonical_name="Alphabet",
        symbol="GOOG",
        country_group="usa",
        aliases=("谷歌", "google", "alphabet", "goog", "googl"),
        official_domains=("abc.xyz", "google.com"),
    ),
    CompanyProfile(
        canonical_name="Apple",
        symbol="AAPL",
        country_group="usa",
        aliases=("苹果", "apple", "aapl"),
        official_domains=("apple.com",),
    ),
    CompanyProfile(
        canonical_name="Intel",
        symbol="INTC",
        country_group="usa",
        aliases=("英特尔", "intel", "intc"),
        official_domains=("intel.com",),
    ),
    CompanyProfile(
        canonical_name="Microsoft",
        symbol="MSFT",
        country_group="usa",
        aliases=("微软", "microsoft", "msft"),
        official_domains=("microsoft.com",),
    ),
    CompanyProfile(
        canonical_name="Amazon",
        symbol="AMZN",
        country_group="usa",
        aliases=("亚马逊", "amazon", "amzn"),
        official_domains=("amazon.com",),
    ),
    CompanyProfile(
        canonical_name="Broadcom",
        symbol="AVGO",
        country_group="usa",
        aliases=("博通", "broadcom", "avgo"),
        official_domains=("broadcom.com",),
    ),
    CompanyProfile(
        canonical_name="Meta",
        symbol="META",
        country_group="usa",
        aliases=("meta", "facebook", "脸书", "meta platforms", "fb", "meta platforms (facebook)"),
        official_domains=("meta.com", "atmeta.com"),
    ),
    CompanyProfile(
        canonical_name="Tesla",
        symbol="TSLA",
        country_group="usa",
        aliases=("特斯拉", "tesla", "tsla"),
        official_domains=("tesla.com", "sec.gov"),
    ),
    CompanyProfile(
        canonical_name="Berkshire Hathaway",
        symbol="BRK-B",
        country_group="usa",
        aliases=("伯克希尔", "berkshire", "berkshire hathaway", "brk-b", "brk.b"),
        official_domains=("berkshirehathaway.com",),
    ),
    CompanyProfile(
        canonical_name="Walmart",
        symbol="WMT",
        country_group="usa",
        aliases=("沃尔玛", "walmart", "wmt"),
        official_domains=("walmart.com",),
    ),
    CompanyProfile(
        canonical_name="Xiaomi",
        symbol="1810.HK",
        country_group="china",
        aliases=("小米", "xiaomi", "1810.hk"),
        official_domains=("mi.com",),
    ),
    CompanyProfile(
        canonical_name="China Merchants Bank",
        symbol="600036.SH",
        country_group="china",
        aliases=("招商银行", "招行", "china merchants bank", "cmb", "600036.sh"),
        official_domains=("cmbchina.com",),
    ),
)


def get_company_catalog() -> tuple[CompanyProfile, ...]:
    return _COMPANY_CATALOG


def build_company_alias_map() -> dict[str, tuple[str, str]]:
    aliases: dict[str, tuple[str, str]] = {}
    for company in _COMPANY_CATALOG:
        for alias in company.aliases:
            aliases[alias.lower()] = (company.canonical_name, company.symbol)
    return aliases


def find_company_profile(company: str | None = None, symbol: str | None = None) -> CompanyProfile | None:
    normalized_company = company.lower() if company else None
    normalized_symbol = symbol.lower() if symbol else None
    for profile in _COMPANY_CATALOG:
        if normalized_company and profile.canonical_name.lower() == normalized_company:
            return profile
        if normalized_symbol and profile.symbol.lower() == normalized_symbol:
            return profile
        if normalized_company and normalized_company in {alias.lower() for alias in profile.aliases}:
            return profile
    return None
