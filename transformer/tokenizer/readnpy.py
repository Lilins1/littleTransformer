import numpy as np

# 加载词汇表（注意allow_pickle=True）
vocab = np.load(
    r"C:\Users\Ruizhe\Desktop\Study\Code\AI\transformer\tokenizer\parallel_vocab.npy",
    allow_pickle=True
)
print(vocab)
print(type(vocab))  # 查看类型
print(vocab.dtype)  # 查看前10个元素
print(vocab.shape)  # 查看形状（如果是 ndarray）
print(vocab[0:304])  # 查看前10个元素

# 提取所有word到一个列表（或NumPy数组）
all_words = [item['word'] for item in vocab]  # 推荐用列表，简单直观

# 要查询的token
token_to_find = 'a'

# 查找索引
indices = [i for i, word in enumerate(all_words) if word == token_to_find]

if not indices:
    print(f"'{token_to_find}' 不存在")
else:
    print(f"'{token_to_find}' 存在，位置索引：{indices}")
    # 可以进一步获取对应的index值
    for i in indices:
        print(f"  子词 '{vocab[i]['word']}' 的ID是：{vocab[i]['index']}")
