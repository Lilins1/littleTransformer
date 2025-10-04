import os
import json
import re
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
from numba import jit
import pickle

# 配置路径
TRAIN_File_PATH = r"C:\Users\Ruizhe\Desktop\Study\ID2221\Project\Data\pubmedqa\ori_pqau.json"
MAX_LENGTH = 20_000  # 最大文本长度
NUM_PROCESSES = max(1, 2 * cpu_count() - 1)  # 进程数


# ------------------------------
# 核心编译函数（Numba加速）
# ------------------------------
@jit(nopython=True)
def merge_ids(seq, a_id, b_id, new_id):
    """Numba编译的ID序列合并函数"""
    new_seq = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a_id and seq[i+1] == b_id:
            new_seq.append(new_id)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return np.array(new_seq, dtype=np.int32)


# ------------------------------
# 进程间任务函数
# ------------------------------
def process_chunk(args):
    """子进程任务：处理一部分序列的合并和计数更新"""
    chunk, a_id, b_id, new_id = args
    merged_chunk = []
    local_pairs = defaultdict(int)
    
    # 合并当前块的序列
    for seq in chunk:
        new_seq = merge_ids(seq, a_id, b_id, new_id)
        merged_chunk.append(new_seq)
        
        # 统计新序列中的子词对
        for i in range(len(new_seq) - 1):
            pair = (new_seq[i], new_seq[i+1])
            local_pairs[pair] += 1
    
    return merged_chunk, local_pairs


# ------------------------------
# 分词器类
# ------------------------------
class ParallelUltraFastTokenizer:
    num_special_tokens = 4
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    
    def __init__(self, num_normal_tokens=10000):
        self.num_normal_tokens = num_normal_tokens
        self.subword_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self.id_to_subword = {i: t for i, t in enumerate(self.special_tokens)}
        self.next_id = self.num_special_tokens
        self.id_sequences = []  # 子词ID序列列表（元素为np.array）
        self.pair_counts = defaultdict(int)
        self.pool = Pool(NUM_PROCESSES)


    def preprocess(self, text: str):
        """预处理：先转小写 → 标点单独分词 → 生成字符级ID序列"""
        # 1. 转换为小写
        text = text.lower()
        print("文本已转换为小写，示例：", text[:200])
        
        # 2. 标点单独分词
        tokens = re.findall(r"\w+|[^\w\s]", text)
        print(f"标点拆分示例（小写后）：{tokens[:20]}")
        
        for token in tokens:
            # 3. 生成字符序列
            if re.fullmatch(r"\w+", token):
                chars = list(token) + ['</w>']
            else:
                chars = list(token)
            
            # 4. 生成ID序列
            ids = []
            for c in chars:
                if c not in self.subword_to_id:
                    self.subword_to_id[c] = self.next_id
                    self.id_to_subword[self.next_id] = c
                    self.next_id += 1
                ids.append(self.subword_to_id[c])
            self.id_sequences.append(np.array(ids, dtype=np.int32))
        
        # 初始化子词对计数
        self._parallel_count_pairs()


    def _parallel_count_pairs(self):
        """并行统计所有子词对频率"""
        total = len(self.id_sequences)
        chunk_size = (total + NUM_PROCESSES - 1) // NUM_PROCESSES
        chunks = [
            self.id_sequences[i:i+chunk_size] 
            for i in range(0, total, chunk_size)
        ]
        
        results = self.pool.map(_count_chunk_pairs, chunks)
        
        for chunk_pairs in results:
            for pair, cnt in chunk_pairs.items():
                self.pair_counts[pair] += cnt


    def train(self, verbose_interval=500):
        initial_count = self.next_id - self.num_special_tokens
        print(f"初始子词数量：{initial_count}，使用{NUM_PROCESSES}个进程并行训练")
        
        merge_steps = self.num_normal_tokens - initial_count
        if merge_steps <= 0:
            print("初始子词已满足目标，无需合并")
            return
        
        print(f"需要合并{merge_steps}次以达到{self.num_normal_tokens}个子词...")
        
        for step in tqdm(range(merge_steps), desc="并行BPE训练"):
            if not self.pair_counts:
                break
            
            best_pair = max(self.pair_counts, key=self.pair_counts.get)
            a_id, b_id = best_pair
            a_sub = self.id_to_subword[a_id]
            b_sub = self.id_to_subword[b_id]
            new_sub = a_sub + b_sub
            new_id = self.next_id
            self.next_id += 1
            self.subword_to_id[new_sub] = new_id
            self.id_to_subword[new_id] = new_sub
            
            del self.pair_counts[best_pair]
            
            total = len(self.id_sequences)
            chunk_size = (total + NUM_PROCESSES - 1) // NUM_PROCESSES
            chunks = [
                self.id_sequences[i:i+chunk_size] 
                for i in range(0, total, chunk_size)
            ]
            args = [(chunk, a_id, b_id, new_id) for chunk in chunks]
            results = self.pool.map(process_chunk, args)
            
            self.id_sequences = []
            for merged_chunk, local_pairs in results:
                self.id_sequences.extend(merged_chunk)
                for pair, cnt in local_pairs.items():
                    self.pair_counts[pair] += cnt
            
            if (step + 1) % verbose_interval == 0:
                tqdm.write(f"第{step+1}步：合并 {a_sub}+{b_sub}→{new_sub}，当前子词数：{self.next_id - self.num_special_tokens}")
        
        self.pool.close()
        self.pool.join()
        
        final_count = self.next_id - self.num_special_tokens
        print(f"\n训练完成，最终子词数：{final_count}，总词汇表大小：{self.next_id}")


    def save_vocab(self, path="parallel_vocab", format="npy"):
        """
        保存词汇表（支持npy和json两种格式）
        :param path: 保存路径（不含后缀）
        :param format: 格式，可选"npy"或"json"
        """
        if format == "npy":
            # 保存为npy格式（二进制，高效）
            save_path = f"{path}.npy"
            # 转换为列表形式（npy更适合存储数组，这里用列表保存字典条目）
            vocab_list = [{"word": k, "index": v} for k, v in self.subword_to_id.items()]
            np.save(save_path, vocab_list)
            print(f"词汇表已保存为npy格式：{save_path}")
        
        elif format == "json":
            # 保存为json格式（可读，跨平台）
            save_path = f"{path}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.subword_to_id, f, ensure_ascii=False, indent=2)
            print(f"词汇表已保存为json格式：{save_path}")
        
        else:
            raise ValueError("格式必须为'npy'或'json'")


# 辅助函数：用于进程池中的子词对计数
def _count_chunk_pairs(chunk):
        chunk_pairs = defaultdict(int)
        for seq in chunk:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                chunk_pairs[pair] += 1
        return chunk_pairs


def load_and_concatenate_text(file_path=TRAIN_File_PATH, max_length=MAX_LENGTH):
    """加载JSON文件并拼接成一个大字符串，限制总长度为MAX_LENGTH"""
    text_list = []
    current_length = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        q = f"QUESTION: {value['QUESTION']}"
        c = f"CONTEXT: {' '.join(value['CONTEXTS'])}"
        la = f"LONG_ANSWER: {value['LONG_ANSWER']}"

        record_length = len(q) + len(c) + len(la)

        if current_length + record_length > max_length:
            remaining = max_length - current_length
            partial_text = (q + " " + c + " " + la)[:remaining]
            text_list.append(partial_text)
            break

        text_list.extend([q, c, la])
        current_length += record_length

    text = " ".join(text_list)
    print(f"加载并拼接完成，文本长度：{len(text)}字符")
    return text


if __name__ == '__main__':
    text = load_and_concatenate_text()
    tokenizer = ParallelUltraFastTokenizer(num_normal_tokens=10000)
    tokenizer.preprocess(text)
    tokenizer.train(verbose_interval=500)
    
    # 保存为npy（默认）
    tokenizer.save_vocab(format="npy")
    # 如需保存为json，取消下面一行注释
    # tokenizer.save_vocab(format="json")
