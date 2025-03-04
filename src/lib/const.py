YOKOU_METHODS = [
    "accuracy",
    "diversity:topic:category",
    "novelty:ip",
    "serendipity:topic:category",
    "information-gain",
]
ALL_METHODS = YOKOU_METHODS + [
    "diversity:embedding",
    "diversity:topic:category-u",
    "novelty:du:topic:category",
    "novelty:du:topic:category-u",
    "novelty:du:embedding",
    "serendipity:topic:category-u",
]
LANGUANGE_MAP_JP = {
    "accuracy": "再現率",
    "diversity": "多様性",
    "novelty": "新規性",
    "serendipity": "意外性",
    "information-gain": "提案指標",
    "topic": "トピック",
    "embedding": "埋め込み",
    "ip": "IP",  # Item Popularity
    "du": "DU",  # Distance from User
    "category": "小カテゴリ",
    "category-u": "大カテゴリ",
    "round": "ラウンド",
    "method": "アルゴリズム",
    "artificial": "人工データ",
    "dunnhumby": "Dunnhumby",
    "tafeng": "Tafeng",
    "entropy": r"$H(P(k_u|\mathcal{D}))$",
    "test-precision": r"$\mathcal{I}^{test}$に対する正解率",
    "test-recall": r"$\mathcal{I}^{test}$に対する再現率",
    "hit": "正解数",
}


def translate(s: str, lan_map: dict[str, str]) -> str:
    e = s.split(":")[0]
    ret = lan_map[e] if e in lan_map else e
    if len(s.split(":")) > 1:
        ret += (
            "（"
            + "，".join(
                map(lambda e: lan_map[e] if e in lan_map else e, s.split(":")[1:])
            )
            + "）"
        )
    ret = (
        ret.replace("トピック，", "")
        .replace("（", "")
        .replace("，", "")
        .replace("）", "")
        .replace("DU", "")
        .replace("IP", r"$I$")
        .replace("小カテゴリ", r"$c$")
        .replace("大カテゴリ", r"$C$")
        .replace("埋め込み", r"$e$")
    )
    return ret
