## 依赖

opencv3

## 环境

Windows 7/8  
Visual Studio 2012  


## 配置

1. 官网下载opencv 3.0，并解压到 C:\
2. 配置环境变量
	OPENCV_DIR  
	C:\opencv\build\x86\vc12

工程配置

1. 项目属性，C/C++，包含头文件
	$(OPENCV_DIR)\..\..\include

2. 项目属性，链接器，常规，库文件
	$(OPENCV_DIR)\lib

3. 项目属性，链接器，输入，附加依赖项
	opencv_world300.lib;opencv_ts300.lib;

详细配置过程可以参考[这篇文章](http://blog.martinperis.com/2014/11/how-to-install-opencv-3-in-windows-8.html)
