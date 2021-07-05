Hi!这是  <b>基于Bloom认知理论的知识熟练度计算方法研究</b>  的代码实现，根据毕设的内容，本项目的代码分为两个模块，分别是知识熟练度计算和高阶知识组发现。

数据集使用了两个，分别是[ASSIST](https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD/tree/master/data)（公开）和HDU（爬取）

在知识熟练度计算中，相关的文件主要是单独出来的py文件。

在高阶知识组发现中，相关的文件存放在Cluster文件夹下。

* 主要代码的结构
> 
> data_analysis.py :数据集相关信息统计代码
> 
> data_process.py:数据预处理代码，完成知识组矩阵的计算和高阶知识组矩阵的计算，并产生对应的数据对，为模型的训练提供数据。
> 
> BloomCDM.py：Bloom模型的定义代码
> 
> predict.py:模型的预测代码
> 
> train.py：模型的训练代码
>
>constract_test:所有的对比试验代码
* 运行命令行
> 可以下载数据集后通过命令行运行
> 
> 数据处理命令行 python data_process.py
> 
> 模型 python train.py
> 
> (参数在程序中有默认值，也可以自己设置，添加在命令行中即可)
