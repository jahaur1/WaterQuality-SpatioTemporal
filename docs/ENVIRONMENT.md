# 已使用的环境

本次上传没有更改原环境，也没有重新训练。

主数值/深度实验环境：Python 3.11，Windows；numpy 1.26.4、pandas 2.1.4、scipy 1.14.1、scikit-learn 1.5.2、torch 2.11.0+cu128、transformers 4.57.6、requests 2.34.2。Chronos运行时2.3.2，PEFT 0.17.1曾隔离安装在Chronos LoRA实验的deps目录。GPU为RTX 5060 Ti 16GB。

各历史实验的环境核验/配置JSON保留在相应目录；并非所有实验运行时都来自同一个全局site-packages。安装依赖不能替代提供原始数据、缓存、模型权重和上游代码。

地理实验使用局部deps：shapely 2.0.7、pyproj 3.7.1、pyshp 2.3.1、eviltransform 0.1.1；原百度坐标转换还使用bd09convertor 0.0.3。地理结果含粗定位与自动河段匹配，不应当作核实后的站点坐标真值。
