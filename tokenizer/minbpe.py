import regex as re
import unicodedata

"""
计算连续整数对的出现次数，并存储在字典中。
例如：输入 [1, 2, 3, 1, 2]，返回 {(1, 2): 2, (2, 3): 1, (3, 1): 1}
可以选择更新一个已有的字典 counts。
"""
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


"""
在整数列表 `ids` 中，将所有连续出现的 `pair` 替换为新的整数 `idx`。
例如：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
"""
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids



"""
处理字符串 `s`，将控制字符转换为 Unicode 转义序列，例如 `\n` -> `\u000A`。
这样可以避免打印控制字符时导致格式混乱。
"""
def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

"""
以可读格式打印 token，转义控制字符。
"""
def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

'''
Tokenizer 基类
'''
class Tokenizer:
    def __init__(self):
        # 要合并的token,有原token下标对和合并后新下标组成的map，(int, int) -> int
        self.merges = {}
        self.pattern = ""
        # 特殊 token 映射，例如 {'<|endoftext|>': 100257}
        self.special_tokens = {}
        # 构建词汇表
        self.vocab = self._build_vocab()

    """
    训练 Tokenizer，生成大小为 vocab_size 的词汇表
    """
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    """
    将文本编码为整数列表
    """
    def encode(self, text):
        raise NotImplementedError

    """
    将整数列表解码为文本
    """
    def decode(self, ids):
        raise NotImplementedError

    """
    根据 merges 构建词汇表
    """
    def _build_vocab(self):
        # 基本词汇
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 合并词汇
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 特殊词汇
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    """
    将模型保存到两个文件：
    - `{file_prefix}.model`：包含合并规则等关键数据
    - `{file_prefix}.vocab`：仅供人工查看的词汇表
    """
    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # 特殊词汇行数
            f.write(f"{len(self.special_tokens)}\n")
            # 特殊词汇及其下标
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 要合并词汇的下标对
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                # 如果是合并的词汇
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"

            self.pattern = f.readline().strip()

            # 读取特殊词汇及在词汇表中的下标
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取要合并的下标
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
re.compile(GPT4_SPLIT_PATTERN)


class BPETokenizer(Tokenizer):

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        # 将special_tokens中的key、value反转
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 对文本进行分词
        text_chunks = re.findall(self.compiled_pattern, text)

        # 将分词结果转为UTF-8编码的字节列表
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 要合并词汇列表
        merges = {}
        # 初始化基础词汇表（0-255 对应的单字节）
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 合并token至要求的词汇表数量
        for i in range(num_merges):
            stats = {}
            # 选取分词中出现最多的组合
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            # 词汇表下标
            idx = 256 + i
            # 和并出现最多的组合
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 保存合并结果
            merges[pair] = idx
            # 将合并的词汇加到词汇表中
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存要合并的词汇
        self.merges = merges
        # 保存词汇表
        self.vocab = vocab

    '''
    将special_tokens中的key、value反转
    '''
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    """
    将 token ID 序列解码回文本
    """
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    """
    对文本字节序列进行BPE编码，返回token ID列表
    """
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    """
    对普通文本进行编码，不考虑特殊 token
    """
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    """
    对文本进行编码，支持特殊 token 处理

    :param text: 输入文本
    :param allowed_special: 特殊 token 处理策略：
        - "all"：允许所有特殊 token
        - "none"：不允许特殊 token
        - "none_raise"：遇到特殊 token 时抛出异常（默认）
        - 自定义 set：允许指定特殊 token
        :return: token ID 列表
    """
    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

