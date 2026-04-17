from transformers import pipeline, AutoTokenizer

# 1. 加载Tokenizer（与模型匹配，避免警告）
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# 2. 初始化文本生成流水线（指定轻量模型，适合新手）
generator = pipeline("text-generation", model="distilgpt2", device=0)

# 生成语法（控制参数，避坑必写）
result = generator(
    "PyTorch is a powerful framework for",  # 提示词
    max_length=50,  # 生成文本最大长度
    num_return_sequences=1,  # 生成数量
    pad_token_id=tokenizer.eos_token_id  # 避免警告，必加
)
print(result[0]['generated_text'])  # 输出生成的完整文本