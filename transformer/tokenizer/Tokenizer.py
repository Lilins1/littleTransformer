import numpy as np
import torch
import re

class BPETokenizer:
    def __init__(self, vocab_path, verbose=False):
        # 宏变量：控制是否输出匹配过程信息
        self.verbose = verbose
        
        # 加载词汇表
        vocab_np = np.load(vocab_path, allow_pickle=True)
        self.subword_to_id = {entry['word']: entry['index'] for entry in vocab_np}
        self.id_to_subword = {v: k for k, v in self.subword_to_id.items()}
        self.vocab_size = len(self.subword_to_id)

        # 特殊符号
        self.PAD_ID = self.subword_to_id['<PAD>']
        self.UNK_ID = self.subword_to_id['<UNK>']
        self.SOS_ID = self.subword_to_id['<SOS>']
        self.EOS_ID = self.subword_to_id['<EOS>']

        # 预生成子词长度索引
        self._build_subword_length_index()
        
        if self.verbose:
            print(f"[初始化] 词汇表大小: {self.vocab_size}")
            print(f"[初始化] 特殊符号ID: PAD={self.PAD_ID}, UNK={self.UNK_ID}, SOS={self.SOS_ID}, EOS={self.EOS_ID}")

    def _build_subword_length_index(self):
        """按子词长度分组，长子词优先匹配"""
        self.subword_lengths = {}
        for subword in self.subword_to_id:
            if subword in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                continue
            length = len(subword)
            self.subword_lengths.setdefault(length, []).append(subword)
        self.sorted_lengths = sorted(self.subword_lengths.keys(), reverse=True)
        
        if self.verbose:
            print(f"[子词索引] 可用子词长度（从长到短）: {self.sorted_lengths[:5]}...")

    def _split_word_to_subwords(self, word):
        """把单词拆分为 BPE 子词，带匹配过程显示"""
        if self.verbose:
            print(f"\n[拆分单词] 开始拆分: '{word}'")
            
        word += '</w>'  # 添加词尾标记
        subwords = []
        start = 0
        n = len(word)
        
        if self.verbose:
            print(f"[拆分单词] 带词尾标记: '{word}' (长度: {n})")
        
        while start < n:
            matched = False
            # 筛选可用的子词长度
            valid_lengths = [l for l in self.sorted_lengths if l <= n - start]
            max_len = max(valid_lengths) if valid_lengths else 1
            
            if self.verbose:
                print(f"[匹配过程] 位置 {start}，剩余字符: '{word[start:]}'，尝试最大长度: {max_len}")
            
            # 尝试从最长到最短匹配
            for length in [max_len] + [l for l in self.sorted_lengths if l < max_len and l <= n - start]:
                candidate = word[start:start+length]
                if self.verbose:
                    print(f"[匹配过程] 尝试子词: '{candidate}' (长度: {length})")
                
                if candidate in self.subword_to_id:
                    subwords.append(candidate)
                    start += length
                    matched = True
                    if self.verbose:
                        print(f"[匹配成功] 找到子词: '{candidate}'，移动到位置 {start}")
                    break
            
            if not matched:
                # 匹配失败，按单个字符处理
                candidate = word[start]
                fallback = candidate if candidate in self.subword_to_id else '<UNK>'
                subwords.append(fallback)
                start += 1
                if self.verbose:
                    status = "找到字符" if fallback != '<UNK>' else "未找到，使用UNK"
                    print(f"[匹配失败] {status}: '{candidate}'，移动到位置 {start}")
        
        if self.verbose:
            print(f"[拆分结果] 单词 '{word[:-4]}' 拆分为: {subwords}")
        
        return subwords

    def decode(self, token_ids):
        """Token ID -> 文本，带解码过程显示"""
        if self.verbose:
            print("\n[解码开始] 输入Token ID数量: {}".format(len(token_ids)))
        
        subwords = []
        # 转换为整数列表处理
        for idx in token_ids.tolist(): 
            if idx == self.EOS_ID:
                if self.verbose:
                    print(f"[解码过程] 遇到EOS ID ({self.EOS_ID})，停止解码")
                break
            if idx in [self.PAD_ID, self.SOS_ID]:
                if self.verbose:
                    print(f"[解码过程] 跳过特殊ID: {idx}")
                continue
            
            subword = self.id_to_subword.get(idx, '<UNK>')
            subwords.append(subword)
            
            if self.verbose:
                print(f"[解码过程] ID {idx} -> 子词 '{subword}'")
        
        # 合并子词
        text = ''.join(subwords).replace('</w>', ' ')
        # 合并多个空格
        result = ' '.join(text.split())
        
        if self.verbose:
            print(f"[解码结果] 合并后文本: '{result}'")
        
        return result
    
    def encode(self, text, max_len=512):
        """文本 -> Token ID，带编码过程显示"""
        if self.verbose:
            print(f"\n[编码开始] 原始文本: '{text}'")
        
        # 转为小写
        text = text.lower()
        if self.verbose:
            print(f"[编码过程] 转为小写: '{text}'")
        
        # 分离单词和标点
        words = re.findall(r"\w+|[^\w\s]", text)
        if self.verbose:
            print(f"[编码过程] 拆分单词和标点: {words}")
        
        token_ids = [self.SOS_ID]
        if self.verbose:
            print(f"[编码过程] 初始Token: [SOS={self.SOS_ID}]")
        
        for word in words:
            subwords = self._split_word_to_subwords(word)
            # 转换为ID
            word_ids = [self.subword_to_id.get(sw, self.UNK_ID) for sw in subwords]
            token_ids.extend(word_ids)
            
            if self.verbose:
                print(f"[编码过程] 单词 '{word}' 对应的ID: {word_ids}")
        
        # 添加EOS
        token_ids.append(self.EOS_ID)
        if self.verbose:
            print(f"[编码过程] 添加EOS: {self.EOS_ID}，当前总长度: {len(token_ids)}")
        
        # 截断或填充
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            if self.verbose:
                print(f"[编码过程] 长度超过{max_len}，截断为: {len(token_ids)}")
        else:
            pad_length = max_len - len(token_ids)
            token_ids += [self.PAD_ID] * pad_length
            if self.verbose:
                print(f"[编码过程] 填充{pad_length}个PAD，总长度: {max_len}")
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    # ==================== 新增方法在这里 ====================
    def encode_corpus(self, text):
        """
        专门用于编码长文本语料库的方法。
        - 不添加 <SOS> 或 <EOS>
        - 不进行填充或截断
        - 返回一个纯净的 token ID 列表
        """
        if self.verbose:
            print(f"\n[语料库编码开始] 文本长度: {len(text)} 字符")
        
        # 1. 预处理文本 (小写、分离单词和标点)
        text = text.lower()
        words = re.findall(r"\w+|[^\w\s]", text)
        
        if self.verbose:
            print(f"[语料库编码] 预处理后得到 {len(words)} 个单词/标点")
        
        # 2. 遍历所有单词并转换为 token ID
        all_token_ids = []
        for word in words:
            subwords = self._split_word_to_subwords(word)
            word_ids = [self.subword_to_id.get(sw, self.UNK_ID) for sw in subwords]
            all_token_ids.extend(word_ids)
            
        if self.verbose:
            print(f"[语料库编码完成] 共生成 {len(all_token_ids)} 个 Token ID")
            
        return all_token_ids
    # =======================================================


if __name__ == "__main__":
    # 控制是否输出匹配过程（True=显示，False=不显示）
    VERBOSE_MODE = True
    
    tokenizer = BPETokenizer("vocab_test_set.npy", verbose=VERBOSE_MODE)
    
    text = "QUESTION: What causes fever?"
    encoded = tokenizer.encode(text, max_len=30)
    decoded = tokenizer.decode(encoded)

    print("\n===== 最终结果 =====")
    print("原始文本:", text)
    print("编码后ID:", encoded.tolist())
    print("解码后文本:", decoded)
