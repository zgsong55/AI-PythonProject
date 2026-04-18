from transformers import pipeline, AutoTokenizer
import torch

device = 0 if torch.cuda.is_available() else -1

from read_csv_file import read_csv_file, analyze_financial_data

def financial_sentiment_analysis(stats):
    print("=" * 70)
    print("金融文本情感分析")
    print("=" * 70)
    print(f"GPU 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    else:
        print("显卡: 无可用GPU，使用CPU运行")
    print(f"待分析内容：{stats}")

    classifier = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device
    )

    model_name = "uer/gpt2-chinese-cluecorpussmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device
    )

    type_analysis = stats['类型分析']

    for type_name, type_data in type_analysis.items():
        text_parts = [f"类型：{type_name}"]
        text_parts.append(f"交易笔数：{type_data['交易笔数']}笔")
        text_parts.append(f"总金额(元)：{type_data['总金额(元)']:.2f}元")
        text_parts.append(f"平均价格(元)：{type_data['平均价格(元)']:.2f}元")
        text_parts.append(f"平均溢价率(%)：{type_data['平均溢价率(%)']:.2f}%")
        text = ",".join(text_parts)
        print("=" * 70)
        print("预测结果")
        print("=" * 70)

        res = classifier(text)[0]
        label = res["label"]
        score = res["score"]

        if label in ["4 stars", "5 stars"]:
            sentiment = "正面情感"
        elif label in ["1 star", "2 stars"]:
            sentiment = "负面情感"
        else:
            sentiment = "中性情感"

        print(f"【{type_name}】{text}")
        print(f"    {sentiment} | 置信度：{score:.2f}")

        result = generator(
            text,
            max_length=512,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        print(f"    生成分析：{result[0]['generated_text']}")
        print("=" * 70)


if __name__ == "__main__":
    file_path = r"D:\work\workspace\AI-PythonProject\拍卖数据-成交房产汇总.csv"
    skip_rows = 2
    df = read_csv_file(file_path, skip_rows)
    if df is None:
        print("数据为空")
        exit()
    stats = analyze_financial_data(df)
    if stats is None:
        print("分析失败")
        exit()
    try:
        financial_sentiment_analysis(stats)
    except Exception as e:
        print(f"错误：{e}")