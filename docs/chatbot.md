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

