import datetime as date
import numpy as np
import pandas as pd
import os


def _clean_data(df):
    initial_rows = len(df)
    df.dropna(how='all', inplace=True)
    numeric_columns = ['起拍价格（单位：元）', '评估价（单位：元）', '成交价',
                       '建筑面积(单位:平方米)', '债权本金（单位：元）',
                       '未尝利息（单位：元）', '债权总额（单位：元）']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if '成交价' in df.columns:
        before_clean = len(df)
        df = df[df['成交价'].notna()]
        df = df[df['成交价'] > 0]
        removed = before_clean - len(df)
        if removed > 0:
            print(f"移除无效成交记录：{removed} 条")
    else:
        print("错误：'成交价' 列不存在，无法继续分析")
        return None

    # 折扣率 = 成交价/评估价*100%
    if '评估价（单位：元）' in df.columns and '成交价' in df.columns:
        df['折扣率'] = np.where(
            df['评估价（单位：元）'] > 0,
            df['成交价'] / df['评估价（单位：元）']*100,
            0
        )
    final_rows = len(df)
    print(f"移除无效数据：{initial_rows - final_rows} 条")

    # 溢价率 = (成交价-起拍价)/起拍价*100%
    if '起拍价格（单位：元）' in df.columns and '成交价' in df.columns:
        df['溢价率(%)'] = np.where(
            df['起拍价格（单位：元）'] > 0,
            (df['成交价'] - df['起拍价格（单位：元）']) / df['起拍价格（单位：元）']*100,
            0
        )
    final_rows = len(df)
    print(f"移除无效数据：{initial_rows - final_rows} 条")


    return df

def analyze_financial_data(df):
    if df is None or len(df) == 0:
        print("待分析数据为空")
        return None
    print("\n" + "=" * 70)
    print("金融交易数据统计分析报告")
    print("=" * 70)
    current_time = date.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"生成时间：{current_time}")
    print("=" * 70)

    stats = {}
    print("\n【1】基础统计分析")
    print("-" * 70)
    total_transactions = len(df)
    total_amount = df['成交价'].sum()
    avg_amount = df['成交价'].mean()
    median_amount = df['成交价'].median()
    max_amount = df['成交价'].max()
    min_amount = df['成交价'].min()
    std_amount = df['成交价'].std()

    stats['总交易笔数'] = total_transactions
    stats['总成交金额(元)'] = total_amount
    stats['平均成交价(元)'] = round(avg_amount, 2)
    stats['中位数成交价(元)'] = round(median_amount, 2)
    stats['最高成交价(元)'] = round(max_amount, 2)
    stats['最低成交价(元)'] = round(min_amount, 2)
    stats['标准差(元)'] = round(std_amount, 2)

    print(f"总交易笔数：{total_transactions:,} 笔")
    print(f"总成交金额：{total_amount:,.2f} 元 ({total_amount:,.2f} 元)")
    print(f"平均成交价：{avg_amount:,.2f} 元")
    print(f"中位数成交价：{median_amount:,.2f} 元")
    print(f"最高成交价：{max_amount:,.2f} 元")
    print(f"最低成交价：{min_amount:,.2f} 元")
    print(f"价格标准差：{std_amount:,.2f} 元")

    if '折扣率' in df.columns:
        print("\n【2】折扣率分析")
        print("-" * 70)
        discount_rate = df['折扣率'].mean()
        print(f"平均折扣率：{discount_rate:,.2f}%")
        stats['平均折扣率(%)'] = round(discount_rate, 2)


    if '溢价率(%)' in df.columns:
        print("\n【3】溢价率分析")
        print("-" * 70)
        avg_premium = df['溢价率(%)'].mean()
        max_premium = df['溢价率(%)'].max()
        min_premium = df['溢价率(%)'].min()
        print(f"平均溢价率：{avg_premium:,.2f}%")
        print(f"最高溢价率：{max_premium:,.2f}%")
        print(f"最低溢价率：{min_premium:,.2f}%")
        stats['平均溢价率(%)'] = round(avg_premium, 2)
        stats['最高溢价率(%)'] = round(max_premium, 2)
        stats['最低溢价率(%)'] = round(min_premium, 2)

    if '类型' in df.columns:
        print("\n【4】类型分析")
        print("-" * 70)
        type_stats = df.groupby('类型').agg({'成交价': ['count', 'sum', 'mean']}).round(1)
        type_stats.columns = ['交易笔数', '总金额(元)', '平均价格(元)']
        display_cols = ['交易笔数', '总金额(元)', '平均价格(元)']
        if '溢价率(%)' in df.columns:
            type_stats['平均溢价率(%)'] = df.groupby('类型')['溢价率(%)'].mean().round(2)
            display_cols.append('平均溢价率(%)')

        print(type_stats[display_cols].to_string())

        type_analysis = {}
        for type_name, row in type_stats.iterrows():
            type_analysis[str(type_name)] = {
                '交易笔数': int(row['交易笔数']),
                '总金额(元)': round(row['总金额(元)'], 2),
                '平均价格(元)': round(row['平均价格(元)'], 2)
            }
            if '平均溢价率(%)' in display_cols:
                type_analysis[str(type_name)]['平均溢价率(%)'] = round(row['平均溢价率(%)'], 2)

        stats['类型分析'] = type_analysis

    return  stats

def read_csv_file(file_path, skip_rows=0):
    if not os.path.exists(file_path):
        print("文件路径不存在")
        return None
    if not (file_path.endswith(".csv")):
        print("文件格式错误")
        return None
    try:
        df = pd.read_csv(file_path, encoding='utf-8', skiprows=skip_rows)
        print(f"成功读取文件：{file_path}")
        print(f"文件共有{df.shape[0] }行数据,{df.shape[1]}列数据")
        print(f"列名: {list(df.columns)}")
        print(df.columns)
        df = _clean_data(df)
        return  df
    except Exception as e:
        print(f"文件读取错误{e}")
        return None



if __name__ == '__main__':
    file_path = r"D:\work\workspace\AI-PythonProject\拍卖数据-成交房产汇总.csv"
    skip_rows = 2
    try:
        df = read_csv_file(file_path, skip_rows)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.width', None)
        stats = analyze_financial_data(df)
        if stats:
            print("程序执行成功！")
        else:
            print("分析失败")
        print(df)
    except Exception as e:
        print(f"错误：{e}")



