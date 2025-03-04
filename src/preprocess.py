import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_dunnhumby(output_path: str, item_topic_file: str) -> None:
    def data_path(file: str) -> str:
        return f"./data/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/{file}"

    transaction_df = pd.read_csv(data_path("transaction_data.csv"))
    causal_df = pd.read_csv(data_path("causal_data.csv"))
    product_df = pd.read_csv(data_path("product.csv"))

    # exclude weeks without recommendation
    valid_weeks = set(causal_df.WEEK_NO.unique())
    transaction_df = transaction_df[transaction_df.WEEK_NO.isin(valid_weeks)]
    causal_df = causal_df[causal_df.WEEK_NO.isin(valid_weeks)]

    causal_df = causal_df[causal_df.display != "0"]
    # shops that have at least one visitor for each week
    a = transaction_df.groupby("STORE_ID")["WEEK_NO"].nunique()
    valid_stores = set(a[a == transaction_df["WEEK_NO"].nunique()].index)
    transaction_df = transaction_df[transaction_df["STORE_ID"].isin(valid_stores)]
    causal_df = causal_df[causal_df["STORE_ID"].isin(valid_stores)]

    # items recommended for at least one week on average among the shops
    a = causal_df["PRODUCT_ID"].value_counts()
    valid_items1 = set(a[a >= len(valid_stores)].index)

    # items that existed for at least half the period (47 weeks)
    a = transaction_df.groupby("PRODUCT_ID")["WEEK_NO"].nunique()
    valid_items2 = set(a[a > transaction_df["WEEK_NO"].nunique() / 2].index)

    # users visiting more than one store in at least five weeks
    a = transaction_df.groupby(["household_key", "WEEK_NO"])["STORE_ID"].nunique()
    a = a[a > 1].groupby("household_key").count()
    valid_users = set(a[a >= 5].index)

    valid_items = valid_items1 & valid_items2
    print(
        "user_n:",
        len(valid_users),
        "item_n:",
        len(valid_items),
        "store_n:",
        len(valid_stores),
    )

    transaction_df = transaction_df[
        transaction_df.PRODUCT_ID.isin(valid_items)
        & transaction_df.household_key.isin(valid_users)
    ].reset_index(drop=True)
    causal_df = causal_df[causal_df.PRODUCT_ID.isin(valid_items)].reset_index(drop=True)

    transaction_df = transaction_df[
        ["household_key", "WEEK_NO", "STORE_ID", "PRODUCT_ID"]
    ]
    causal_df = (
        causal_df.groupby(["WEEK_NO", "STORE_ID"])["PRODUCT_ID"]
        .unique()
        .rename("PRODUCT_IDS")
        .to_frame()
        .reset_index()
    )

    causal_df["PRODUCT_IDS"] = causal_df["PRODUCT_IDS"].apply(
        lambda s: " ".join(map(str, s))
    )
    transaction_df = transaction_df.rename(
        columns={
            "household_key": "user_id",
            "WEEK_NO": "t",
            "STORE_ID": "store_id",
            "PRODUCT_ID": "item_id",
        }
    )
    causal_df = causal_df.rename(
        columns={"WEEK_NO": "t", "STORE_ID": "store_id", "PRODUCT_IDS": "item_ids"}
    )

    user_le = LabelEncoder().fit(transaction_df.user_id)
    item_le = LabelEncoder().fit(transaction_df.item_id)
    item_ids = set(item_le.classes_)

    transaction_df.user_id = user_le.transform(transaction_df.user_id)
    transaction_df.item_id = item_le.transform(transaction_df.item_id)

    causal_df.item_ids = (
        causal_df.item_ids.apply(lambda s: list(map(int, s.split())))
        .apply(lambda s: list(filter(lambda p: p in item_ids, s)))
        .apply(lambda s: item_le.transform(s).tolist())
    )

    item_df = product_df.rename(columns={"PRODUCT_ID": "item_id"})
    item_df = item_df[item_df.item_id.isin(item_ids)]
    item_df.item_id = item_le.transform(item_df.item_id)
    item_df.to_csv(item_topic_file, index=False)

    recommendation_df = causal_df.set_index(["t", "store_id"])
    recommendation_df.item_ids = recommendation_df.item_ids.apply(set)
    rec_dict = recommendation_df.item_ids.to_dict()

    td = transaction_df.t.max()
    te = 8
    data: dict[str, dict] = {"valid": {}, "test": {}}

    time_split = {
        ("valid", "train"): (1, td - 2 * te),
        ("valid", "eval"): (td - 2 * te + 1, td - te),
        ("test", "train"): (te + 1, td - te),
        ("test", "eval"): (td - te + 1, td),
    }

    p_u: dict = {}
    for (mode, split_type), (tl, tr) in time_split.items():
        p_u[mode, split_type] = (
            transaction_df[(tl <= transaction_df.t) & (transaction_df.t < tr)]
            .groupby("user_id")["item_id"]
            .agg(set)
            .apply(list)
        ).to_dict()

    for (mode, split_type), _ in time_split.items():
        data[mode][f"{split_type}_purchased_items"] = []
        data[mode][f"{split_type}_recommended_items"] = []

    for user_id, user_df in transaction_df.drop_duplicates(
        subset=["user_id", "t", "store_id"]
    ).groupby("user_id")[["t", "store_id"]]:
        for (mode, split_type), (tl, tr) in time_split.items():
            data[mode][f"{split_type}_purchased_items"].append(
                p_u[mode, split_type][user_id]
                if user_id in p_u[mode, split_type]
                else []
            )

        for (mode, split_type), (tl, tr) in time_split.items():
            col_name = f"{split_type}_recommended_items"
            data[mode][col_name].append(set())

        for _, row in user_df.iterrows():
            for (mode, split_type), (tl, tr) in time_split.items():
                col_name = f"{split_type}_recommended_items"
                if tl <= row.t <= tr and (row.t, row.store_id) in rec_dict:
                    data[mode][col_name][-1] |= rec_dict[row.t, row.store_id]

        for (mode, split_type), _ in time_split.items():
            col_name = f"{split_type}_recommended_items"
            data[mode][col_name][-1] = list(data[mode][col_name][-1])

    topics = {
        "category-u": LabelEncoder()
        .fit_transform(product_df["COMMODITY_DESC"])
        .tolist(),
        "category": LabelEncoder()
        .fit_transform(product_df["SUB_COMMODITY_DESC"])
        .tolist(),
    }

    user_n, item_n = (
        transaction_df["user_id"].nunique(),
        transaction_df["item_id"].nunique(),
    )
    params = {"user_n": user_n, "item_n": item_n}
    json_data = {
        "params": params,
        "data": {
            "valid": json.loads(pd.DataFrame(data["valid"]).to_json(orient="index")),
            "test": json.loads(pd.DataFrame(data["test"]).to_json(orient="index")),
        },
        "item_topics": topics,
    }

    with open(output_path, "w") as f:
        json.dump(json_data, f)


def preprocess_tafeng(output_path: str, item_topic_file: str) -> None:
    df = pd.read_csv("./data/tafeng/ta_feng_all_months_merged.csv")
    df["TRANSACTION_DT"] = pd.to_datetime(df["TRANSACTION_DT"])

    a = (
        df.groupby(["PRODUCT_ID", "TRANSACTION_DT"])["SALES_PRICE"]
        .median()
        .rename("SALES_PRICE_DAY_MEDIAN")
    )
    b = (
        df.groupby("PRODUCT_ID")["SALES_PRICE"]
        .median()
        .rename("SALES_PRICE_TOTAL_MEDIAN")
    )
    df = pd.merge(df, a, how="left", on=["PRODUCT_ID", "TRANSACTION_DT"])
    df = pd.merge(df, b, how="left", on="PRODUCT_ID")

    df["DISCOUNT_RATIO"] = (
        1 - df["SALES_PRICE_DAY_MEDIAN"] / df["SALES_PRICE_TOTAL_MEDIAN"]
    )
    df["IS_RECOMMENDED"] = df["DISCOUNT_RATIO"] > 0.1

    # items purchased at least 100 times
    a = df.groupby("PRODUCT_ID")["TRANSACTION_DT"].count()
    valid_items = set(a[a >= 30].index)

    # users visiting the shop at least 100 times
    a = df.groupby("CUSTOMER_ID")["TRANSACTION_DT"].count()
    valid_users = set(a[a >= 100].index)

    item_df = df.groupby("PRODUCT_ID")["PRODUCT_SUBCLASS"].first()
    item_df = item_df.reset_index().rename(columns={"PRODUCT_ID": "item_id"})
    df = df[df["CUSTOMER_ID"].isin(valid_users) & df["PRODUCT_ID"].isin(valid_items)]
    print(
        "user_n:",
        len(valid_users),
        "item_n:",
        len(valid_items),
    )
    df = df[["CUSTOMER_ID", "PRODUCT_ID", "TRANSACTION_DT", "IS_RECOMMENDED"]].rename(
        columns={
            "CUSTOMER_ID": "user_id",
            "PRODUCT_ID": "item_id",
            "TRANSACTION_DT": "date",
            "IS_RECOMMENDED": "is_recommended",
        }
    )

    user_le = LabelEncoder().fit(df.user_id)
    item_le = LabelEncoder().fit(df.item_id)
    item_ids = set(item_le.classes_)

    item_df = item_df[item_df.item_id.isin(item_ids)]
    item_df.item_id = item_le.transform(item_df.item_id)
    item_df = item_df.sort_values("item_id").reset_index(drop=True)

    df.user_id = user_le.transform(df.user_id)
    df.item_id = item_le.transform(df.item_id)

    a = df.groupby(["date", "item_id"]).is_recommended.mean()
    assert len(a.unique()) == 2  # 0 or 1

    rec_dict = (
        a[a == 1]
        .reset_index()
        .set_index("date")
        .drop("is_recommended", axis=1)
        .groupby("date")
        .agg(set)
        .item_id.to_dict()
    )

    data: dict[str, dict] = {"valid": {}, "test": {}}
    time_split = {
        ("valid", "train"): (pd.to_datetime("2000/11/1"), pd.to_datetime("2001/1/1")),
        ("valid", "eval"): (pd.to_datetime("2001/1/1"), pd.to_datetime("2001/2/1")),
        ("test", "train"): (pd.to_datetime("2000/12/1"), pd.to_datetime("2001/2/1")),
        ("test", "eval"): (pd.to_datetime("2001/2/1"), pd.to_datetime("2001/3/1")),
    }

    p_u: dict = {}
    for (mode, split_type), (tl, tr) in time_split.items():
        p_u[mode, split_type] = (
            df[(tl <= df.date) & (df.date < tr)]
            .groupby("user_id")["item_id"]
            .agg(set)
            .apply(list)
        ).to_dict()

    for (mode, split_type), _ in time_split.items():
        data[mode][f"{split_type}_purchased_items"] = []
        data[mode][f"{split_type}_recommended_items"] = []

    for user_id, user_df in df.drop_duplicates(subset=["user_id", "date"]).groupby(
        "user_id"
    )[["date"]]:
        for (mode, split_type), (tl, tr) in time_split.items():
            data[mode][f"{split_type}_purchased_items"].append(
                p_u[mode, split_type][user_id]
                if user_id in p_u[mode, split_type]
                else []
            )

        for (mode, split_type), _ in time_split.items():
            data[mode][f"{split_type}_recommended_items"].append(set())

        for _, row in user_df.iterrows():
            for (mode, split_type), (tl, tr) in time_split.items():
                if tl <= row.date <= tr and row.date in rec_dict:
                    data[mode][f"{split_type}_recommended_items"][-1] |= rec_dict[
                        row.date
                    ]

        for (mode, split_type), _ in time_split.items():
            data[mode][f"{split_type}_recommended_items"][-1] = list(
                data[mode][f"{split_type}_recommended_items"][-1]
            )

    user_n, item_n = len(user_le.classes_), len(item_le.classes_)
    params = {"user_n": user_n, "item_n": item_n}
    topics = {
        "category-u": LabelEncoder()
        .fit_transform(item_df["PRODUCT_SUBCLASS"] // 100)
        .tolist(),  # PRODUCT_SUBCLASSの最初の4文字が大分類として使える（っぽい）
        "category": LabelEncoder().fit_transform(item_df["PRODUCT_SUBCLASS"]).tolist(),
    }
    json_data = {
        "params": params,
        "data": {
            "valid": json.loads(pd.DataFrame(data["valid"]).to_json(orient="index")),
            "test": json.loads(pd.DataFrame(data["test"]).to_json(orient="index")),
        },
        "item_topics": topics,
    }

    item_df.to_csv(item_topic_file, index=False)

    with open(output_path, "w") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    preprocess_dunnhumby(
        output_path="./data/preprocessed/dunnhumby/data.json",
        item_topic_file="./data/preprocessed/dunnhumby/item_topic.csv",
    )

    preprocess_tafeng(
        output_path="./data/preprocessed/tafeng/data.json",
        item_topic_file="./data/preprocessed/tafeng/item_topic.csv",
    )
