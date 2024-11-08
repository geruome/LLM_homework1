from base import Tokenizer, get_stats, merge
from queue import PriorityQueue

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255
        # print(text_bytes[:30])
        # print(ids[:30])
        merges = {} # (id0,id1) -> new_id
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # 统计pair的出现次数
            stats = get_stats(ids)
            # print(list(stats.keys())[:10])
            # pair = max(stats, key=stats.get)
            pair = max(stats.items(), key=lambda item: (item[1], -item[0][0], -item[0][1])) [0] # 取出现次数最大的pair, 若次数相同选pair标号最小的
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab 

    def decode(self, ids):
        # token ids -> string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # string -> token ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2: # 1个元素时无法合并,终止
            # 统计pair的出现次数
            stats = get_stats(ids)
            # 找到merge后idx最小的pair (出现频率相对更高)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: # 没有高频的pair可供合并
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

class LinkedListText(): # 双向链表维护文本中的pair合并
    def init(self, text):
        assert type(text)==list and type(text[0])==int # 0-255 interger list

        n = len(text)
        self.text = text
        self.pre = [0 for _ in range(n)] # 前驱
        self.nxt = [0 for _ in range(n)] # 后继
        self.poss = {} # (int,int) -> set(int), pair所在的所有位置
        self.changes = {}
        stats = {}
        self.start = 0
        for i in range(n):
            self.pre[i] = i-1 if i>0 else None
            self.nxt[i] = i+1 if i<n-1 else None
            if i<n-1:
                pair = (self.text[i], self.text[i+1])
                self.ins(i, i+1)
                stats[pair] = stats.get(pair, 0) + 1
        return stats

    def ins(self, x, y):
        if x==None or y==None:
            return
        pair = (self.text[x], self.text[y])
        if pair not in self.poss:
            self.poss[pair] = set()
        self.poss[pair].add(x)
        self.changes[pair] = self.changes.get(pair, 0) + 1

    def era(self, x, y):
        if x==None or y==None:
            return
        pair = (self.text[x], self.text[y])
        self.changes[pair] = self.changes.get(pair, 0) - 1
        if pair not in self.poss:
            return
        self.poss[pair].remove(x) # discard

    def merge_one(self, x, newid):
        # 合并单处
        y = self.nxt[x]
        pre = self.pre[x]
        nxt = self.nxt[y]
        now = len(self.text)
        # print(pre, x, y, nxt, newid, now, '-------')
        self.text.append(newid); self.pre.append(pre); self.nxt.append(nxt)
        self.era(x, y); self.era(pre, x); self.era(y, nxt)
        self.ins(pre, now); self.ins(now, nxt)
        if pre!=None:
            self.nxt[pre] = now
        if nxt!=None:
            self.pre[nxt] = now        
        if x == self.start:
            self.start = now

    def merge(self, pair, newid):
        '''
        合并链表中所有相邻的pair为newid
        不需要考虑合并newid的影响, newid不可能参与本轮合并 
        '''
        self.changes = {}
        lst = self.poss.pop(pair)
        for x in lst:
            self.merge_one(x, newid)
        return self.changes
    
    def show_text(self): 
        # 顺序输出文本
        res = []
        now = self.start
        while now!=None:
            res.append(self.text[now])
            now = self.nxt[now]
        print(res)

# a = LinkedListText()
# a.init(text = [0,1,2,0,1,2])
# a.merge((0,1), 3)
# a.merge((3,2), 4)
# a.show_text()


class MyTokenizer(Tokenizer):
    '''
    改进 BPE 复杂度
    现有 train和encode复杂度都是 vocab_size * n (n为文本长度)
    将文本视为链表, 合并token的过程中, 链表相邻元素merge, 动态维护stats并找max, 每个pair的所有出现位置. O(nlogn)
    '''
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255
        text = LinkedListText()
        stats = text.init(ids)
        Q = PriorityQueue()
        for pair,occ in stats.items():
            Q.put((-occ, pair))
        # print(Q.queue[:10]); exit(0)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            occ, pair = None, None
            while not Q.empty():
                occ,pair = Q.get()
                occ = -occ
                # print(occ, pair, '?????????????')
                if (pair not in stats) or (stats[pair]!=occ):
                    continue
                break
            if occ==None: 
                print("The whole text has been merged together!")
                return
            idx = 256 + i
            # print(occ, pair, '?????????????'); exit(0)
            changes = text.merge(pair, idx)
            # print(pair, idx, changes); exit(0)
            for k,v in changes.items():
                stats[k] = stats.get(k, 0) + v
                Q.put((-stats[k], k))
            
            assert stats[pair]==0
            stats.pop(pair)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {occ} occurrences")

        self.merges = merges
        self.vocab = vocab 

    def decode(self, ids):
        # token ids -> string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # string -> token ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2: # 1个元素时无法合并,终止
            # 统计pair的出现次数
            stats = get_stats(ids)
            # 找到merge后idx最小的pair (出现频率相对更高)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: # 没有高频的pair可供合并
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
