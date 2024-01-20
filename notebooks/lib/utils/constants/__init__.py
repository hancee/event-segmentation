# Source: https://www.worldstandards.eu/other/tlds/
# Last updated: 10 April 2023
COUNTRY_DOMAIN_LIST = [
    "ABUDHABI",
    "AF",
    "AX",
    "AL",
    "DZ",
    "AS",
    "AD",
    "AO",
    "AI",
    "AQ",
    "AG",
    "AR",
    "AM",
    "AW",
    "AC",
    "AU",
    "AT",
    "AZ",
    "BS",
    "BH",
    "BD",
    "BB",
    "EUS",
    "BY",
    "BE",
    "BZ",
    "BJ",
    "BM",
    "BT",
    "BO",
    "BQ",
    "AN",
    "NL",
    "BA",
    "BW",
    "BV",
    "BR",
    "IO",
    "VG",
    "BN",
    "BG",
    "BF",
    "MM",
    "BI",
    "KH",
    "CM",
    "CA",
    "CV",
    "CAT",
    "KY",
    "CF",
    "TD",
    "CL",
    "CN",
    "CX",
    "CC",
    "CO",
    "KM",
    "CD",
    "CG",
    "CK",
    "CR",
    "CI",
    "HR",
    "CU",
    "CW",
    "CY",
    "NC",
    "TR",
    "CZ",
    "DK",
    "DJ",
    "DM",
    "DO",
    "AE",
    "TL",
    "TP",
    "EC",
    "EG",
    "SV",
    "UK",
    "GQ",
    "ER",
    "EE",
    "ET",
    "EU",
    "FO",
    "FK",
    "FJ",
    "FI",
    "FR",
    "GF",
    "PF",
    "TF",
    "GA",
    "GAL",
    "GM",
    "PS",
    "GE",
    "DE",
    "GH",
    "GI",
    "UK",
    "GB",
    "GR",
    "GL",
    "GD",
    "GP",
    "GU",
    "GT",
    "GG",
    "GN",
    "GW",
    "GY",
    "HT",
    "HM",
    "NL",
    "HN",
    "HK",
    "HU",
    "IS",
    "IN",
    "ID",
    "IR",
    "IQ",
    "IE",
    "UK",
    "IM",
    "IL",
    "IT",
    "JM",
    "JP",
    "JE",
    "JO",
    "KZ",
    "KE",
    "KI",
    "KP",
    "KR",
    "AL",
    "KW",
    "KG",
    "LA",
    "LV",
    "LB",
    "LS",
    "LR",
    "LY",
    "LI",
    "LT",
    "LU",
    "MO",
    "MK",
    "MG",
    "MW",
    "MY",
    "MV",
    "ML",
    "MT",
    "MH",
    "MQ",
    "MR",
    "MU",
    "YT",
    "MX",
    "FM",
    "MD",
    "MC",
    "MN",
    "ME",
    "MS",
    "MA",
    "MZ",
    "MM",
    "NA",
    "NR",
    "NP",
    "NL",
    "NC",
    "NZ",
    "NI",
    "NE",
    "NG",
    "NU",
    "NF",
    "KP",
    "MK",
    "UK",
    "MP",
    "NO",
    "OM",
    "PK",
    "PW",
    "PS",
    "PA",
    "PG",
    "PY",
    "PE",
    "PH",
    "PN",
    "PL",
    "PT",
    "PR",
    "QA",
    "RE",
    "RO",
    "RU",
    "RW",
    "BQ",
    "BL",
    "SH",
    "KN",
    "LC",
    "MF",
    "VC",
    "PM",
    "WS",
    "SM",
    "ST",
    "SA",
    "UK",
    "SN",
    "RS",
    "SC",
    "SL",
    "SG",
    "BQ",
    "SX",
    "SK",
    "SI",
    "SB",
    "SO",
    "SO",
    "ZA",
    "GS",
    "KR",
    "SS",
    "ES",
    "LK",
    "SD",
    "SR",
    "SJ",
    "SZ",
    "SE",
    "CH",
    "SY",
    "PF",
    "TW",
    "TJ",
    "TZ",
    "TH",
    "TG",
    "TK",
    "TO",
    "TT",
    "TN",
    "TM",
    "TC",
    "TV",
    "UG",
    "UA",
    "AE",
    "UK",
    "US",
    "VI",
    "UY",
    "UZ",
    "VU",
    "VA",
    "VE",
    "VN",
    "UK",
    "WF",
    "PS",
    "EH",
    "YE",
    "ZM",
    "ZW",
]

TOP_LEVEL_DOMAIN_LIST = [
    "COM",
    "NET",
    "ORG",
    "GOV",
    "EDU",
    "EN",
    "ENG",
    # "COM:443",
    # "NET:443",
    "OR",
    "NETWORK",
]
SUBDOMAIN_LIST = [
    "WWW3",
    "NEWS",
    "ENGLISH",
    "OPINION",
    "BUSINESS",
    "LEGACY",
    "ENTERTAINMENT",
    "OPINION",
    "LIFESTYLE",
    "POP",
    "NEWSINFO",
    "GLOBALNATION",
    "CEBUDAILYNEWS",
    "RADYO",
    "USA",
    "INTERAKSYON",
    "GLOBAL",
    "ENG",
    "INTERNATIONAL",
    "FINANCE",
    "ASIA",
    "TECHNOLOGY",
    "HOME",
    "EUROPE",
    "GO",
    "ECONOMICTIMES",
    "TIMESOFINDIA",
    "BLOGSPOT",
    "GATE",
    "FRENCH",
    "EDITION",
    "NOTICES",
    "ARTICLE",
    "BETA",
]

from lib.utils.constants.mappers import (
    country_code_map,
    event_code_map,
    geo_type_code_map,
    gov_arm_code_map,
    role_code_map,
)

country_role_code_map = {
    f"{c}{r}": f"{country_code_map[c].upper()} {role_code_map[r]}"
    for c in country_code_map
    for r in role_code_map
}

country_gov_arm_code_map = {
    f"{k}{ga}": f"{country_code_map[k].upper()} {gov_arm_code_map[ga]}"
    for k in country_code_map.keys()
    for ga in gov_arm_code_map.keys()
}
