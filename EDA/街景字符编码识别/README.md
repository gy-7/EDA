## 数据集来源：

[街景字符编码识别赛题与数据-天池大赛-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/531795/information) 



## 分析：

进行一个简单的分析，分析数据集中图片尺寸，宽高比，bbox尺寸，宽高比，以及每张图片中bbox数量的分布情况。

训练集共有三万张图片。

instances_train2017.json，是我们通过将数据集json文件转换后符合coco数据集标准的json文件。




可视化结果展示：

<table>
    <tr>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064340_images_width_distribution.png" >  图片的宽度分布情况 </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064335_images_height_distribution.png" >  图片的高度分布情况  </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064329_images_aspect_ratio.png"  > 图片的宽高比分布情况 </center></td>
    </tr>
    <tr>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064319_bboxes_width_distribution.png" >bbox的宽度分布情况    </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064308_bboxes_height_distribution.png"> bbox的高度分布情况  </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064259_bboxes_aspect_ratio%20.png" > bbox的宽高比分布情况  </center></td>
    </tr>
    <tr>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064345_images_width_height_distribution.png"> 图片宽度和高度的分布情况 </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064324_bboxes_width_height_distribution.png">  bbox宽度和高度的分布情况 </center></td>
        <td ><center><img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_211230064313_bboxes_per_image_distribution.png">  每张图片中bbox数量的分布情况  </center></td>
    </tr>
</table>



:star: 做完所有的EDA后，可以得到以下分析：

- 图片的宽度大部分处于0\~200，小部分处于200\~400之间，极少数>400。
- 图片的高度大部分处于0\~100，小部分处于100\~200之间，极少数>200。
- 图片的宽高比大部分处于1.7~3之间。
- bbox的宽度大部分处于0~50。
- bbox的高度大部分处于0\~50，小部分处于50\~100。
- bbox的宽高比大部分处于0.25\~0.75。
- 每张图片中bbox数量大部分是1，2，3，小部分有4个bbox，极少数有5，6个bbox。

