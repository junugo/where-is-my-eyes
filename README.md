<div align=center>
<h1>where is my eyes</h1>
<h3>基于OpenCV的Python眼动跟踪</h3>
</div>

## 介绍
本项目基于OpenCV的Python实现，通过摄像头捕捉人脸，并使用OpenCV的Haar级联分类器进行人脸检测，然后使用OpenCV的Haar级联分类器进行眼睛检测，最后通过OpenCV的Haar级联分类器进行瞳孔检测，从而实现眼动跟踪。
具体实现与效果请参考 Python 代码。

仅作学习使用，在正式场景下精度可能不足。

误差大的场景：光线不足、佩戴眼镜、眼镜未完全睁开

## 源码运行
- 克隆此仓库`https://github.com/junugo/where-is-my-eyes.git`
- 一键安装依赖`pip install -r requirements.txt`
- 运行`python OpenCV.py`
