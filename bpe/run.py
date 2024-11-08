from basic import BasicTokenizer, MyTokenizer
import time
from transformers import GPT2Tokenizer
import tiktoken

# input_context = 'Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university.'
input_context = '博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。'
# print(len(input_context.encode("utf-8"))); exit(0)

def work_BPEtokenizer():

    fl = open('manual.txt','r')
    context = fl.read()
    # print(context[:20])
    # print("len of context: ",len(context)) # 1.74e5

    tokenizer = MyTokenizer()
    # tmp = time.time()
    tokenizer.train(text=context, vocab_size=1024, verbose=False)
    # print(time.time()-tmp)

    # encode_res = tokenizer.encode(context)
    # decode_res = tokenizer.decode(encode_res)
    # print(context[:30], '\n', decode_res[:30])
    # assert context == decode_res 

    encode_res = tokenizer.encode(input_context)
    print("Encode_res of BPETokenizer:\n", len(encode_res))


def work_GPT2tokenizer():

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encode_res = tokenizer(input_context) # return_tensors='pt')
    encode_res = encode_res['input_ids']
    print("Encode_res of GPT2Tokenizer:\n", len(encode_res))

work_BPEtokenizer()
work_GPT2tokenizer()