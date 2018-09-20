# histogram_equalization
Some histogram equalization methods to enhance image contrast, including AHE and CLAHE.

* 知乎链接
* 代码架构
* 方法
  * ImageOps HE
  * HE  
  * AHE
  * CLAHE  
  * Local Region Stretch HE
* 结果展示

## 知乎(zhihu)链接
  * [直方图均衡化](https://zhuanlan.zhihu.com/p/44918476)

## 代码架构
 * contrast.py  各种直方图均衡化实现的脚本，ImageContraster类
 * main.py  测试，使用方法参考main.py
 * python3.6

## 方法
  * 代码实现了五种直方图均衡化的方法，分别是：1.利用PIL.ImageOps实现的直方图均衡化；2.自己实现的直方图均衡化HE；3.自适应直方图均衡化AHE；4.限制对比度自适应直方图均衡化CLAHE;5.自适应局部区域伸展直方图均衡化Local Region Stretch HE。其原理详细介绍见知乎链接。

## 结果展示
  下面给出一些结果图片：
  <div> 
    <table>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ops_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/he_car.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ahe_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/clahe_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/lrs_car.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ops_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/he_cap.png"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ahe_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/clahe_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/lrs_cap.png"></td>
     </tr>
    </table>
  </div>
