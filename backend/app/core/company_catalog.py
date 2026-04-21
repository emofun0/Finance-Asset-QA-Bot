from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CompanyProfile:
    canonical_name: str
    symbol: str
    country_group: str
    official_domains: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def normalized_names(self) -> tuple[str, ...]:
        values = [self.canonical_name, self.symbol, *self.aliases]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = normalize_company_token(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return tuple(deduped)

    def search_terms(self) -> tuple[str, ...]:
        values = [self.canonical_name, *self.aliases, self.symbol]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = str(value or "").strip()
            if not compact:
                continue
            normalized = normalize_company_token(compact)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(compact)
        return tuple(deduped)

    def matches_any(self, candidates: Iterable[str]) -> bool:
        candidate_set = {normalize_company_token(item) for item in candidates if normalize_company_token(item)}
        return bool(candidate_set & set(self.normalized_names()))


_COMPANY_CATALOG = (
    CompanyProfile(
        canonical_name="Tencent",
        symbol="0700.HK",
        country_group="china",
        official_domains=("tencent.com", "static.www.tencent.com"),
        aliases=("腾讯", "腾讯控股", "Tencent Holdings", "Tencent Holdings Limited"),
    ),
    CompanyProfile(
        canonical_name="ICBC",
        symbol="1398.HK",
        country_group="china",
        official_domains=("icbc-ltd.com",),
        aliases=("工商银行", "中国工商银行", "Industrial and Commercial Bank of China"),
    ),
    CompanyProfile(
        canonical_name="Agricultural Bank of China",
        symbol="601288.SS",
        country_group="china",
        official_domains=("abchina.com",),
        aliases=("农业银行", "中国农业银行", "ABC"),
    ),
    CompanyProfile(
        canonical_name="China Construction Bank",
        symbol="601939.SS",
        country_group="china",
        official_domains=("ccb.com",),
        aliases=("建设银行", "中国建设银行", "CCB"),
    ),
    CompanyProfile(
        canonical_name="Alibaba",
        symbol="BABA",
        country_group="china",
        official_domains=("alibabagroup.com",),
        aliases=("阿里", "阿里巴巴", "Alibaba Group", "阿里巴巴集团"),
    ),
    CompanyProfile(
        canonical_name="PetroChina",
        symbol="0857.HK",
        country_group="china",
        official_domains=("petrochina.com.cn", "petrochina.com"),
        aliases=("中石油", "中国石油", "中国石油天然气股份有限公司"),
    ),
    CompanyProfile(
        canonical_name="CATL",
        symbol="300750.SZ",
        country_group="china",
        official_domains=("catl.com",),
        aliases=("宁德时代", "Contemporary Amperex Technology", "Contemporary Amperex Technology Co., Limited"),
    ),
    CompanyProfile(
        canonical_name="Bank of China",
        symbol="601988.SS",
        country_group="china",
        official_domains=("boc.cn",),
        aliases=("中国银行", "BOC"),
    ),
    CompanyProfile(
        canonical_name="Moutai",
        symbol="600519.SH",
        country_group="china",
        official_domains=("moutaichina.com",),
        aliases=("茅台", "贵州茅台", "Kweichow Moutai"),
    ),
    CompanyProfile(
        canonical_name="China Mobile",
        symbol="0941.HK",
        country_group="china",
        official_domains=("chinamobileltd.com",),
        aliases=("中国移动", "China Mobile Limited"),
    ),
    CompanyProfile(
        canonical_name="NVIDIA",
        symbol="NVDA",
        country_group="usa",
        official_domains=("nvidia.com",),
        aliases=("英伟达",),
    ),
    CompanyProfile(
        canonical_name="Alphabet",
        symbol="GOOG",
        country_group="usa",
        official_domains=("abc.xyz", "google.com"),
        aliases=("谷歌", "Google", "Google LLC"),
    ),
    CompanyProfile(
        canonical_name="Apple",
        symbol="AAPL",
        country_group="usa",
        official_domains=("apple.com",),
        aliases=("苹果", "Apple Inc"),
    ),
    CompanyProfile(
        canonical_name="Intel",
        symbol="INTC",
        country_group="usa",
        official_domains=("intel.com",),
        aliases=("英特尔", "Intel Corporation"),
    ),
    CompanyProfile(
        canonical_name="Microsoft",
        symbol="MSFT",
        country_group="usa",
        official_domains=("microsoft.com",),
        aliases=("微软", "Microsoft Corporation"),
    ),
    CompanyProfile(
        canonical_name="Amazon",
        symbol="AMZN",
        country_group="usa",
        official_domains=("amazon.com",),
        aliases=("亚马逊", "Amazon.com", "Amazon.com, Inc."),
    ),
    CompanyProfile(
        canonical_name="Broadcom",
        symbol="AVGO",
        country_group="usa",
        official_domains=("broadcom.com",),
        aliases=("博通", "Broadcom Inc"),
    ),
    CompanyProfile(
        canonical_name="Meta",
        symbol="META",
        country_group="usa",
        official_domains=("meta.com", "atmeta.com"),
        aliases=("脸书", "Facebook", "Meta Platforms", "Meta Platforms, Inc."),
    ),
    CompanyProfile(
        canonical_name="Tesla",
        symbol="TSLA",
        country_group="usa",
        official_domains=("tesla.com", "sec.gov"),
        aliases=("特斯拉", "Tesla, Inc."),
    ),
    CompanyProfile(
        canonical_name="Berkshire Hathaway",
        symbol="BRK-B",
        country_group="usa",
        official_domains=("berkshirehathaway.com",),
        aliases=("伯克希尔", "伯克希尔哈撒韦"),
    ),
    CompanyProfile(
        canonical_name="Walmart",
        symbol="WMT",
        country_group="usa",
        official_domains=("walmart.com",),
        aliases=("沃尔玛", "Wal-Mart", "Walmart Inc"),
    ),
    CompanyProfile(
        canonical_name="Xiaomi",
        symbol="1810.HK",
        country_group="china",
        official_domains=("mi.com", "hkexnews.hk"),
        aliases=("小米", "小米集团", "Xiaomi Corporation"),
    ),
    CompanyProfile(
        canonical_name="China Merchants Bank",
        symbol="600036.SH",
        country_group="china",
        official_domains=("cmbchina.com",),
        aliases=("招商银行", "CMB", "China Merchants Bank Co., Ltd."),
    ),
)


def get_company_catalog() -> tuple[CompanyProfile, ...]:
    return _COMPANY_CATALOG


def normalize_company_token(value: str | None) -> str:
    return str(value or "").strip().lower()


def find_company_profile(company: str | None = None, symbol: str | None = None) -> CompanyProfile | None:
    candidates = [item for item in [company, symbol] if item and str(item).strip()]
    if not candidates:
        return None
    for profile in _COMPANY_CATALOG:
        if profile.matches_any(candidates):
            return profile
    return None
