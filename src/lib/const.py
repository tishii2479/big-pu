PREPRINT_METHODS = [
    "accuracy",
    "diversity:topic:category",
    "novelty:ip",
    "serendipity:topic:category",
    "information-gain",
]
ALL_METHODS = PREPRINT_METHODS + [
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
    "coverage": "網羅性",
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
LANGUANGE_MAP_EN = {
    "accuracy": "Recall",
    "diversity": "Diversity",
    "novelty": "Novelty",
    "serendipity": "Serendipity",
    "coverage": "Coverage",
    "information-gain": "Proposed",
    "topic": "Topic",
    "embedding": "Embedding",
    "ip": "IP",  # Item Popularity
    "du": "DU",  # Distance from User
    "category": "Subcategory",
    "category-u": "Category",
    "round": "Round",
    "method": "Method",
    "artificial": "Artificial Data",
    "dunnhumby": "Dunnhumby",
    "tafeng": "Tafeng",
    "entropy": r"$H(P(k_u|\mathcal{D}))$",
    "test-precision": r"Accuracy against $\mathcal{I}^{test}$",
    "test-recall": r"Recall against $\mathcal{I}^{test}$",
    "hit": "Hit",
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
