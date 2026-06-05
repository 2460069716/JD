# JD
格式转换及测试onnx文件效果运行环境：使用服务器上的openstl_lsh

conda activate openstl_lsh

格式转换

--将对应模型源码加入models目录

--在ckpt2onnx.py文件中配置模型、路径、文件名、输入参数等

python ckpt2onnx.py

测试

--在test.py文件中配置路径、文件名、需进行绘图的变量等

python test.py
