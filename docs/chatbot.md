# 聊天机器人教程

## 准备工作

1. 数据下载: [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## 加载和预处理数据

### 创建格式化数据文件

* 10,292 对电影角色的220,579 次对话
* 617部电影中的9,035电影角色
* 总共304,713中语调

> movie_lines.txt

```python
L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!
L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.
L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?
L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.
```

生成为:

```python
line = {
    'L183198': {
        'lineID': 'L183198', 
        'characterID': 'u5022', 
        'movieID': 'm333', 
        'character': 'FRANKIE', 
        'text': "Well we'd sure like to help you.\n"
    }, {...}
}
```

> movie_conversations.txt

```python
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L200', 'L201', 'L202', 'L203']
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L205', 'L206']
u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L207', 'L208']
```

与 movie_lines.txt 合并生成为:

```python
[{
    'character1ID': 'u0',
    'character2ID': 'u2',
    'movieID': 'm0',
    'utteranceIDs': "['L194', 'L195', 'L196', 'L197']\n",
    'lines': [{
        'lineID': 'L194',
        'characterID': 'u0',
        'movieID': 'm0',
        'character': 'BIANCA',
        'text': 'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'
    }, {
        'lineID': 'L195',
        'characterID': 'u2',
        'movieID': 'm0',
        'character': 'CAMERON',
        'text': "Well, I thought we'd start with pronunciation, if that's okay with you.\n"
    }, {
        'lineID': 'L196',
        'characterID': 'u0',
        'movieID': 'm0',
        'character': 'BIANCA',
        'text': 'Not the hacking and gagging and spitting part.  Please.\n'
    }, {
        'lineID': 'L197',
        'characterID': 'u2',
        'movieID': 'm0',
        'character': 'CAMERON',
        'text': "Okay... then how 'bout we try out some French cuisine.  Saturday?  Night?\n"
    }]
}, {...}]
```

> 从对话中提取一对句子

将 lines 中的数据分隔开： 一个对话 存一行，并用 \t 分隔开

例如：['L194', 'L195', 'L196', 'L197'], 会拆分为：

```
'L194', 'L195'
'L195', 'L196',
'L196', 'L197'
```

### 加载和清洗数据

1. 对句子进行处理，去除标点之外的所有非字母字符
2. 然后对句子长度进行处理，超过某个长度就过滤掉（目的：加快训练收敛）
3. 然后把句子存在到 Voc 对象下
4. 打印输出 pairs
5. voc.trim, 过滤掉词频数据（有利于让训练更快收敛的策略是去除词汇表中很少使用的单词。减少特征空间也会降低模型学习目标函数的难度）

```
['there .', 'where ?']
['you have my word . as a gentleman', 'you re sweet .']
['hi .', 'looks like things worked out tonight huh ?']
['you know chastity ?', 'i believe we share an art instructor']
['have fun tonight ?', 'tons']
['well no . . .', 'then that s all you had to say .']
['then that s all you had to say .', 'but']
['but', 'you always been this selfish ?']
['do you listen to this crap ?', 'what crap ?']
['what good stuff ?', 'the real you .']
```

### 为模型格式化数据

1. 加速训练，利用GPU并行计算能力，则需要使用小批量 `mini-batches`
2. 为了保证数据长短一致，设置 `(max_length，batch_size)`, 短于 max_length 的句子在 EOS_token 之后进行零填充 `(zero padded)`
3. 矩阵转置（以便跨第一维的索引返回批处理中所有句子的时间步长）

![](https://pytorch.apachecn.org/docs/1.0/img/b2f1969c698070d055c23fc81ab07b1b.jpg)

## 定义模型

Seq2seq模型的目标是将可变长度序列作为输入，并使用固定大小的模型将可变长度序列作为输出返回。

* Seq2Seq模型: 
    1. 编码器，其将可变长度输入序列编码为固定长度上下文向量。 
    2. 解码器，它接收输入文字和上下文矢量，并返回序列中下一句文字的概率和在下一次迭代中使用的隐藏状态。

![](https://pytorch.apachecn.org/docs/1.0/img/32a87cf8d0353ceb0037776f833b92a7.jpg)


* 编码器:

如果将填充的一批序列传递给RNN模块，我们必须分别使用torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence在RNN传递时分别进行填充和反填充。

```py
def forward(self, input_seq, input_lengths, hidden=None):
    # Convert word indexes to embeddings
    embedded = self.embedding(input_seq)
    # Pack padded batch of sequences for RNN module
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    # Forward pass through GRU
    outputs, hidden = self.gru(packed, hidden)
    # Unpack padding
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    # Sum bidirectional GRU outputs
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
    # Return output and final hidden state
    return outputs, hidden
```

![](https://pytorch.apachecn.org/docs/1.0/img/c653271eb5fb762482bceb5e2464e680.jpg)

* 解码器: