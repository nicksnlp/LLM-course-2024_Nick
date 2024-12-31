**Nikolay Vorontsov** 
--- 
**LLMs and GenAI for NLP, 2024**  
<mark>Report on the Exercises in Labs 1 – 6</mark>  
GitHub repository: **[nicksnlp](https://github.com/nicksnlp/LLM-course-2024_Nick)**   

This is what I have done:  

## Week1

### What are tokenisers?

Tokenisers are essential for the implementation of neural networks, each model exist with a tokeniser that was used when training that network.

To tokenise the text means to break it into pieces, whether into words or to smaller parts such as suffixes and stems, characters. There are different tokenisers available, each is best suited to a particular task and/or language.


### Why are they important for language modelling and LLMs?

They are crucial for the further processing of texts. Tokenisation simplifies the text, breaks it into logical units, reduces the vocabulary size, as well as enable models to handle new and rare words. This decreases the training time for the models, and improves their generalisation capabilities. 


## What different tokenisation algorithms there are and which ones are the most popular ones and why?

Text can be tokenised in different ways: into sentences (e.g. in NLTK `sent_piece`), words (e.g.`split()` in python) or on some smaller elements: morphemes (rule-based tokenisation), or into parts of the words, selected on other principles, into characters or bytes.

Some of the well known tokenisers include BPE (Byte Pair Encoding) and Sentence-Piece:

**BPE**  
BPE initially splits the texts of the given sample into characters , but then learning from the co-occurencemerges some of the characters into larger units, this is done recursively. The result of this tokenisation is usually a vocabulary of some `subword-units`, not necesserily morphemes. It is good for treating rare or unknown words, among other things. In a simple algorithm for this is provided in [2]:

```
import re
import collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(vocab)

```

**Sentence-Piece**
Sentence-Piece is another language-independent subword tokeniser, based on unigram model in combination with BPE. It works well with non-phonemic symbols, such as Japanese and Chinese, and is used in such models as T5, XML-R or mBERT.

The traditional BERT model uses [WordPiece](https://paperswithcode.com/method/wordpiece). GPT-2 and GPT-3 models uses BPE on a byte-level, which is effective for treating special characters, for example.

The combination of strategies for tokenisation may be useful to fit training for a particular domain.

An important feature of every tokeniser is also the ability to decode back the tokens into the readable texts, without losses! For example in neural machine translation, a translated text is evaluated on the decoded examples in comparison to the validation counterparts.

*But why not to tokenise everything into bytes, or at least characters?* 
Well, it will lead to very long sequences that the neural network have to process, the vectors will become too long to compute efficiently, and some semantic information that comes from co-occurrence may be lost, or at least require much more computational power to be captured by transformers or other models during training.

### References:

1. [“Why are tokenisers important for language modelling and LLMs?”, and further discussion. ChatGPT, OpenAI, 31 Oct. 2024](https://chatgpt.com/share/67238012-14e0-800b-b251-3907b59bf0f6.)

2. [“tokenisers”, and further discussion. ChatGPT, OpenAI, 31 Oct. 2024](https://chatgpt.com/c/67237c4a-2be0-800b-9998-b4e67753828e.)

3. Neural Machine Translation of Rare Words with Subword Units: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
   
4. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing:  [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)

5. [Notes on BERT tokenizer and model](https://medium.com/@anmolkohli/my-notes-on-bert-tokenizer-and-model-98dc22d0b64#:~:text=BERT%20tokenizer%20uses%20something%20known,algorithm%20to%20generate%20the%20vocabulary.) 

---