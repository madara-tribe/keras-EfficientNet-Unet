# EfficientNet-Unet structure

![eff_unet-2 2](https://user-images.githubusercontent.com/48679574/75096355-2b8f4180-55e2-11ea-87ce-a8bb96f95918.JPG)


# Result prediction

【GroundTruth】

![20200223_000344](https://user-images.githubusercontent.com/48679574/75096281-85433c00-55e1-11ea-9d6e-92dc84636013.GIF)


【My Prediction】

![20200223_000725](https://user-images.githubusercontent.com/48679574/75096284-8e340d80-55e1-11ea-8562-232d409bb45d.GIF)


Its logic and details are written my blog.

https://trafalbad.hatenadiary.jp/entry/2020/02/23/094850



# requirements for FPA and hypercolumn

<b>FPA</b>

・image size are more than 256 

・Amount of calculation increase


<b>HyperColumn</b>

If you use multi input,  using convolution layers output which are contrast position in model often get better practice.


hypercolumn about better practice should be refered to [Convolutional hypercolumns in Python](http://blog.christianperone.com/2016/01/convolutional-hypercolumns-in-python/)
