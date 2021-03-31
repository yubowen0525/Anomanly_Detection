# 项目结构
目录 | 描述
---|---
config | 本地的目录配置
dataset | 数据集
model | 算法模型、模型测试
preprocess | 预处理模块
utlis | 基础工具
algorithm | 算法实现：SAE、VAE、GAN

# Quick Start
1. 配置config/setting.py项目目录
2. 预处理阶段构建**流特征向量数据集`dataset.npz`**：
    1. 将CICIDS2017数据集的csv文件放于 `dataset/MachineLearningCVE` 目录下
    2. 启动preprocess.py中的 `classification("Classification")`函数, 将数据集按Labels进行分类存于 `dataset/Classification` 以便后续分析
    3. 启动preprocess.py中的 `constructDataSet(quantify="minMax", srcPath="Classification", supervised=False)`函数，将
    csv文件按`x_train, y_train, x_cross, y_cross, x_test, y_test`打包进npz文件内，存于`dataset/DeepLearningDateSet` 目录
3. 训练阶段
    - `vae = VAE("dataset-minMax-x-80-unsupervised")` 初始化配置信息
    - `vae.fit()` 填充数据训练模型
    - `vae.save()` 保存模型结构
    - `vae.anomalyScore()` 对模型进行多方位评估

# utils
1. metrics.py
    - `roc_auc` **计算roc, auc 画图**
    - `MultiClassROCAUC` **对每个恶意分类单独计算 roc, auc 画图**
    - `pre_rec_curve` 计算平均precision 画图
2. sklearn.py
    - `describe_loss_and_accuracy` 训练后可对 loss, accuracy 画图
    - `describe_evaluation` **画混合矩阵，记录precision, recall, f1-score, support**
    - `distribution` 画分布图
    - `violinplot` 画小提琴图
    - `boxplot`  **画箱线图**
