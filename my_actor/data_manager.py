import pandas as pd
import numpy as np


COLUMNS_CHART_DATA = ["date", "open", "high", "low", "close", "volume"]

COLUMNS_TRAINING_DATA_V1 = [
    "open_lastclose_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "close_lastclose_ratio",
    "volume_lastvolume_ratio",
    "close_ma5_ratio",
    "volume_ma5_ratio",
    "close_ma10_ratio",
    "volume_ma10_ratio",
    "close_ma20_ratio",
    "volume_ma20_ratio",
    "close_ma60_ratio",
    "volume_ma60_ratio",
    "close_ma120_ratio",
    "volume_ma120_ratio",
]

COLUMNS_TRAINING_DATA_V1_RICH = [
    "open_lastclose_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "close_lastclose_ratio",
    "volume_lastvolume_ratio",
    "close_ma5_ratio",
    "volume_ma5_ratio",
    "close_ma10_ratio",
    "volume_ma10_ratio",
    "close_ma20_ratio",
    "volume_ma20_ratio",
    "close_ma60_ratio",
    "volume_ma60_ratio",
    "close_ma120_ratio",
    "volume_ma120_ratio",
    "inst_lastinst_ratio",
    "frgn_lastfrgn_ratio",
    "inst_ma5_ratio",
    "frgn_ma5_ratio",
    "inst_ma10_ratio",
    "frgn_ma10_ratio",
    "inst_ma20_ratio",
    "frgn_ma20_ratio",
    "inst_ma60_ratio",
    "frgn_ma60_ratio",
    "inst_ma120_ratio",
    "frgn_ma120_ratio",
]

COLUMNS_TRAINING_DATA_V2 = [
    "per",
    "pbr",
    "roe",
    "open_lastclose_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "close_lastclose_ratio",
    "volume_lastvolume_ratio",
    "close_ma5_ratio",
    "volume_ma5_ratio",
    "close_ma10_ratio",
    "volume_ma10_ratio",
    "close_ma20_ratio",
    "volume_ma20_ratio",
    "close_ma60_ratio",
    "volume_ma60_ratio",
    "close_ma120_ratio",
    "volume_ma120_ratio",
    "market_kospi_ma5_ratio",
    "market_kospi_ma20_ratio",
    "market_kospi_ma60_ratio",
    "market_kospi_ma120_ratio",
    "bond_k3y_ma5_ratio",
    "bond_k3y_ma20_ratio",
    "bond_k3y_ma60_ratio",
    "bond_k3y_ma120_ratio",
]

COLUMNS_TRAINING_DATA_V3 = [
    "per",
    "pbr",
    "open_lastclose_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "close_lastclose_ratio",
    "volume_lastvolume_ratio",
    "close_ma5_ratio",
    "volume_ma5_ratio",
    "close_ma10_ratio",
    "volume_ma10_ratio",
    "close_ma20_ratio",
    "volume_ma20_ratio",
    "close_ma60_ratio",
    "volume_ma60_ratio",
    "close_ma120_ratio",
    "volume_ma120_ratio",
    "market_sma5",
    "market_sma20",
    "market_sma60",
    "market_sma120",
    "bond_u3y_sma5",
    "bond_u3y_sma20",
    "bond_u3y_sma60",
    "bond_u3y_sma120",
    "wti_sma5",
    "wti_sma20",
    "wti_sma60",
    "wti_sma120",
    "aroon_5",
    "adx_5",
    "elder_ray_bull_5",
    "elder_ray_bear_5",
    "vortex_pos_5",
    "vortex_neg_5",
    "donchian_5",
    "fcb_upper_5",
    "fcb_lower_5",
    "gator_upper",
    "gator_lower",
    "alligator_jaw",
    "alligator_teeth",
    "alligator_lips",
    "ichimoku_9_26_52",
    "macd_12_26_9",
    "super_trend_14_3",
    "bollinger_bands_upper_20_2",
    "bollinger_bands_lower_20_2",
    "std_dev_channels_20_2",
    "nlp_pos",
    "nlp_neg",
    "nlp_neu",
    "nlp_compound",
]


def preprocess(data, ver="v3"):
    windows = [5, 10, 20, 60, 120]
    for window in windows:  # 이동 평균 전처리
        data["close_ma{}".format(window)] = data["close"].rolling(window).mean()
        data["volume_ma{}".format(window)] = data["volume"].rolling(window).mean()
        data["close_ma%d_ratio" % window] = (
            data["close"] - data["close_ma%d" % window]
        ) / data["close_ma%d" % window]
        data["volume_ma%d_ratio" % window] = (
            data["volume"] - data["volume_ma%d" % window]
        ) / data["volume_ma%d" % window]

        if ver == "v1.rich":
            data["inst_ma{}".format(window)] = data["close"].rolling(window).mean()
            data["frgn_ma{}".format(window)] = data["volume"].rolling(window).mean()
            data["inst_ma%d_ratio" % window] = (
                data["close"] - data["inst_ma%d" % window]
            ) / data["inst_ma%d" % window]
            data["frgn_ma%d_ratio" % window] = (
                data["volume"] - data["frgn_ma%d" % window]
            ) / data["frgn_ma%d" % window]

    data["open_lastclose_ratio"] = np.zeros(len(data))  # 시가/전일 종가 비율 구하기

    data.loc[1:, "open_lastclose_ratio"] = (
        data["open"][1:].values - data["close"][:-1].values
    ) / data["close"][:-1].values
    data["high_close_ratio"] = (data["high"].values - data["close"].values) / data[
        "close"
    ].values
    data["low_close_ratio"] = (data["low"].values - data["close"].values) / data[
        "close"
    ].values
    data["close_lastclose_ratio"] = np.zeros(len(data))
    data.loc[1:, "close_lastclose_ratio"] = (
        data["close"][1:].values - data["close"][:-1].values
    ) / data["close"][:-1].values
    data["volume_lastvolume_ratio"] = np.zeros(len(data))
    data.loc[1:, "volume_lastvolume_ratio"] = (
        data["volume"][1:].values - data["volume"][:-1].values
    ) / data["volume"][:-1].replace(to_replace=0, method="ffill").replace(
        to_replace=0, method="bfill"
    ).values

    if ver == "v1.rich":
        data["inst_lastinst_ratio"] = np.zeros(len(data))
        data.loc[1:, "inst_lastinst_ratio"] = (
            data["inst"][1:].values - data["inst"][:-1].values
        ) / data["inst"][:-1].replace(to_replace=0, method="ffill").replace(
            to_replace=0, method="bfill"
        ).values
        data["frgn_lastfrgn_ratio"] = np.zeros(len(data))
        data.loc[1:, "frgn_lastfrgn_ratio"] = (
            data["frgn"][1:].values - data["frgn"][:-1].values
        ) / data["frgn"][:-1].replace(to_replace=0, method="ffill").replace(
            to_replace=0, method="bfill"
        ).values

    return data


def load_data(fpath, date_from, date_to, ver="v3"):  # csv file 경로, 시작 날짜, 끝 날짜
    header = None if ver == "v1" else 0
    data = pd.read_csv(
        fpath, thousands=",", header=header, converters={"date": lambda x: str(x)}
    )

    if ver == "v1":
        data.columns = ["date", "open", "high", "low", "close", "volume"]

    # 날짜 오름차순 정렬
    data = data.sort_values(by="date").reset_index()

    # 데이터 전처리
    data = preprocess(data)

    # 기간 필터링
    data["date"] = data["date"].str.replace("-", "")
    data = data[(data["date"] >= date_from) & (data["date"] <= date_to)]
    data = data.dropna()

    # for datas in data:
    #   print(datas)

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]
    # print("chart_data = ", chart_data)

    # 학습 데이터 분리
    training_data = None
    if ver == "v1":
        training_data = data[COLUMNS_TRAINING_DATA_V1]
    elif ver == "v1.rich":
        training_data = data[COLUMNS_TRAINING_DATA_V1_RICH]
    elif ver == "v2":
        data.loc[:, ["per", "pbr", "roe"]] = data[["per", "pbr", "roe"]].apply(
            lambda x: x / 100
        )
        training_data = data[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    elif ver == "v3":
        data.loc[:, ["per", "pbr"]] = data[["per", "pbr"]].apply(lambda x: x / 100)
        training_data = data[COLUMNS_TRAINING_DATA_V3]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception("Invalid version.")

    return chart_data, training_data