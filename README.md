# 一种主动学习方法

## 项目简介
本项目是一种新的主动学习方法。

## 环境依赖
- Python >= 3.9
- pytorch >= 2.0.1
- cuda >= 11.7
- BLEURT >= 0.0.2

## 硬件需求
- GPU（推荐支持CUDA的NVIDIA GPU，显存>=24GB）

## 安装
1、克隆仓库：
```
git https://github.com/introverted-programmer/GTAL/tree/master/GTAL.git
cd GTAL
```
2、创建虚拟环境并激活

3、安装依赖

## 数据准备
1、下载平行语料数据集，并将其组织为以下结构：
```
data/
├── train/
│   ├── source.txt  # 源语言文本
│   └── target.txt  # 目标语言文本
├── bpe/
│   ├── source.bpe 
│   └── target.bpe
├── val/
│   ├── source.txt
│   └── target.txt
└── test/
    ├── source.txt
    └── target.txt
```

2、数据预处理：
清洗、bpe分词等

## 模型训练
1、配置训练参数：

  编辑```bin/trainer.py```, 设置模型参数，例如学习率、批量大小、最大训练步数等。

2、开始训练：
```
python bin/trainer.py
```

## 模型使用
1、修改参数：

  编辑```bin/translate.py```, 设置翻译文件路径等。
  
2、开始翻译：
```
python bin/translate.py
```

## 模型测试

使用测试集进行评价分数，可以使用BLEU、TER、BLEURT、chrF指标进行评价，这些评价指标可以在github其他项目上找到。



