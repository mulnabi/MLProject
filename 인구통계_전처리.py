import pandas as pd
import glob

def fill_missing_with_ema(df, group_col, time_col, target_cols, span_value=3):
    df = df.sort_values(by=[group_col, time_col]).reset_index(drop=True)

    for col in target_cols:
        def fill_with_rate(x, col=col):
            rate = x.pct_change()
            ema_rate = rate.ewm(span=span_value, min_periods=1, adjust=False).mean()
            filled = x.copy()
            for i in range(1, len(filled)):
                if pd.isna(filled.iloc[i]):
                    if not pd.isna(filled.iloc[i - 1]):
                        filled.iloc[i] = filled.iloc[i - 1] * (1 + ema_rate.iloc[i])
                    else:
                        filled.iloc[i] = pd.NA
            return filled

        def fill_backward(x, col=col):
            rate = x[::-1].pct_change()
            ema_rate = rate.ewm(span=span_value, min_periods=1, adjust=False).mean()
            filled = x[::-1].copy()
            for i in range(1, len(filled)):
                if pd.isna(filled.iloc[i]):
                    if not pd.isna(filled.iloc[i - 1]):
                        filled.iloc[i] = filled.iloc[i - 1] * (1 + ema_rate.iloc[i])
            return filled[::-1]

        df[col] = df.groupby(group_col)[col].transform(fill_with_rate)

        still_missing = df[col].isna()
        if still_missing.any():
            df.loc[still_missing, col] = df.groupby(group_col)[col].transform(fill_backward)[still_missing]

    df[target_cols] = df[target_cols].round().astype('Int64')
    return df


def add_neighbor_rate(df, group_col, time_col, col, n=2):
    df = df.sort_values(by=[group_col, time_col]).reset_index(drop=True)

    def calc_rate(group):
        values = group.tolist()
        rates = []

        for i in range(len(values)):
            prev_idx = i - n if i - n >= 0 else None
            next_idx = i + n if i + n < len(values) else None

            prev_val = values[prev_idx] if prev_idx is not None and not pd.isna(values[prev_idx]) else None
            next_val = values[next_idx] if next_idx is not None and not pd.isna(values[next_idx]) else None

            if prev_val is not None and next_val is not None and prev_val != 0:
                rate = (next_val - prev_val) / (prev_val * 2 * n)
            elif next_val is not None and not pd.isna(values[i]) and values[i] != 0:
                rate = (next_val - values[i]) / (values[i] * n)
            elif prev_val is not None and prev_val != 0:
                rate = (values[i] - prev_val) / (prev_val * n)
            else:
                rate = None

            rates.append(rate)

        return pd.Series(rates, index=group.index)

    df[f'인구변화율'] = df.groupby(group_col)[col].transform(calc_rate)
    return df


path = '연령통계/*.csv'
file_list = glob.glob(path)
result = pd.DataFrame()
ages = ['0~9', '10~19', '20~29', '30~39', '40~49', '50~59']

for file in file_list:
    data = pd.read_csv(file, encoding='cp949')
    year = int(data.columns[1][:4])

    for y in range(0, 5):
        current_year = year + y
        buff = pd.DataFrame(columns=['시군구', '연도', '총인구수'] + ages)

        buff['시군구'] = data['행정구역'].str.split().str[0:3].str.join(" ")
        buff['연도'] = current_year
        buff['총인구수'] = pd.to_numeric(
            data[f'{current_year}년_계_총인구수'].str.replace(',', ''), errors='coerce'
        )
        for age in ages:
            buff[age] = pd.to_numeric(
                data[f'{current_year}년_계_{age}세'].str.replace(',', ''), errors='coerce'
            )

        buff = buff.groupby(['시군구', '연도'])[['총인구수'] + ages].max().reset_index()
        result = pd.concat([result, buff], axis=0, ignore_index=True)
        
        
        buff = pd.DataFrame(columns=['시군구', '연도', '총인구수'] + ages)

        buff['시군구'] = data['행정구역'].str.split().str[0:2].str.join(" ")
        buff['연도'] = current_year
        buff['총인구수'] = pd.to_numeric(
            data[f'{current_year}년_계_총인구수'].str.replace(',', ''), errors='coerce'
        )
        for age in ages:
            buff[age] = pd.to_numeric(
                data[f'{current_year}년_계_{age}세'].str.replace(',', ''), errors='coerce'
            )

        buff = buff.groupby(['시군구', '연도'])[['총인구수'] + ages].max().reset_index()
        result = pd.concat([result, buff], axis=0, ignore_index=True)

result = fill_missing_with_ema(result, '시군구', '연도', ['총인구수'] + ages, 2)

sum_age = result[ages].sum(axis=1)
avg_age = pd.Series(0.0, index=result.index)
for j, age in enumerate(ages):
    avg_age += (j * 10 + 5) * result[age] / sum_age.replace(0, pd.NA)
result['평균연령'] = avg_age

result = add_neighbor_rate(result, '시군구', '연도', '총인구수', n=2)
result = result.drop(ages, axis=1)

result.to_csv('연령통계.csv', index=False, encoding='utf-8-sig')