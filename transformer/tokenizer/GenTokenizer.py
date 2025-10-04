import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # 用于显示进度条

TRAIN_File_PATH = r"C:\Users\Ruizhe\Desktop\Study\ID2221\Project\Data\pubmedqa\test_set.json"

class Tokenizer:
    vocab_dtype = np.dtype(
        [('word', 'U20'), 
         ('index', np.int32)
        ])  # 词汇表数据类型
    num_special_tokens = 4  # 特殊符号数量（<PAD>、<UNK>、<SOS>、<EOS>）
    num_normal_tokens = 10000  # 目标正常子词总数
    vocab_size = num_special_tokens + num_normal_tokens  # 总词汇表大小
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  # 特殊符号
    vocab = np.array([], dtype=vocab_dtype)  # 词汇表数组
    words = []  # 存储处理后的单词（子词序列）
    subwords = set()  # 跟踪所有正常子词（用于控制数量）


    def __init__(self, num_normal_tokens=10000):
        self.num_normal_tokens = num_normal_tokens
        # 初始化特殊符号到词汇表
        special_entries = [(token, idx) for idx, token in enumerate(self.special_tokens)]
        self.vocab = np.array(special_entries, dtype=self.vocab_dtype)


    def preprocess(self, text: str):
        """预处理文本：拆分为单词，每个单词拆分为字符+</w>后缀"""
        words = text.strip().split()
        # 每个单词拆分为字符（用空格分隔），并添加</w>标记（表示词尾）
        self.words = [' '.join(list(word)) + ' </w>' for word in words]
        # 提取初始子词（字符级）
        for word in self.words:
            for subword in word.split():
                self.subwords.add(subword)


    def get_pairs(self):
        """提取所有相邻子词对及出现频率"""
        pairs = defaultdict(int)
        for word in self.words:
            symbols = word.split()  # 按空格拆分出子词
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += 1
        return pairs


    def merge(self, best_pair):
        """合并最佳子词对，返回新生成的子词"""
        new_subword = best_pair[0] + best_pair[1]  # 合并两个子词
        merged_words = []
        for word in self.words:
            symbols = word.split()
            i = 0
            merged = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                    merged.append(new_subword)  # 用新子词替换原对子
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            merged_words.append(' '.join(merged))
        self.words = merged_words
        return new_subword


    def train(self, verbose_interval=100):
        """训练BPE，合并子词对直到正常子词总数达到num_normal_tokens"""
        initial_count = len(self.subwords)
        print(f"初始子词数量（字符级）：{initial_count}")
        
        merge_steps = self.num_normal_tokens - initial_count
        if merge_steps <= 0:
            print("初始子词已超过目标数量，无需合并")
            return
        
        print(f"需要合并{merge_steps}次以达到{self.num_normal_tokens}个子词...")
        
        # 使用tqdm创建进度条
        for step in tqdm(range(merge_steps), desc="BPE训练进度"):
            # 提取当前所有子词对
            pairs = self.get_pairs()
            if not pairs:  # 没有可合并的对子时提前停止
                break
            
            # 找到最频繁的子词对
            best_pair = max(pairs, key=pairs.get)
            # 合并并获取新子词
            # print("best_pair: ", best_pair)
            new_subword = self.merge(best_pair)
            # 将新子词加入集合
            self.subwords.add(new_subword)
            
            # 定期显示合并信息（默认每100步）
            if (step + 1) % verbose_interval == 0:
                tqdm.write(f"第{step+1}步：合并对子 {best_pair} → {new_subword}，当前子词总数：{len(self.subwords)}")
        
        # 训练完成后，将正常子词加入词汇表（索引从特殊符号之后开始）
        normal_entries = [
            (subword, self.num_special_tokens + idx) 
            for idx, subword in enumerate(self.subwords)
        ]
        # 合并特殊符号和正常子词，更新词汇表
        self.vocab = np.concatenate([
            self.vocab, 
            np.array(normal_entries, dtype=self.vocab_dtype)
        ])
        print(f"\n训练完成，最终正常子词数量：{len(self.subwords)}，总词汇表大小：{len(self.vocab)}")


if __name__ == '__main__':
    # 读取JSON训练数据
    text = ""
    with open(TRAIN_File_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 拼接文本数据
    for key, value in data.items():
        text += f"QUESTION: {value['QUESTION']}\n"
        subtext = "CONTEXT: " + " ".join(value["CONTEXTS"])
        text += f"{subtext}\n"
        text += f"LONG_ANSWER: {value['LONG_ANSWER']}\n"

    print(f"加载并拼接完成，文本长度：{len(text)}字符")
    
    # 初始化并训练分词器
    tokenizer = Tokenizer(num_normal_tokens=300)
    tokenizer.preprocess(text)
    tokenizer.train(verbose_interval=500)  # 每500步显示一次合并信息
    
    # 保存词汇表
    np.save("vocab.npy", tokenizer.vocab)
    print("词汇表已保存到vocab.npy")
