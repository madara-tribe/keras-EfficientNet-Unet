# Parameters

・pixel value range 0~1

・size (256, 256,3) or (128,128,3)

・sigmoid

・loss=MAE

・optimizer=SGD




# not better tecs

・huber loss

・hypercolumn

・dice loss

・Feature Pyramid Attention



# ・Important  parameter
mean_absolute_error loss (it focus on how similar images are)




# ・NN architecture

<img src="https://user-images.githubusercontent.com/48679574/93949013-d6373080-fd7a-11ea-983b-8c760660ad46.png" width="600px">



# datasets
<b>Input data / mask data</b>

<img src="https://user-images.githubusercontent.com/48679574/93950315-282d8580-fd7e-11ea-90f9-a903d8ecedb8.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/93950459-835f7800-fd7e-11ea-8637-8c4da1d704df.png" width="400px">





# Model Performance

## loss curve
train-Loss, validation-loss curve

<img src="https://user-images.githubusercontent.com/48679574/93953140-9295f400-fd85-11ea-8fc0-c5b5c55698c4.png" width="400px">


## Prediction

<b>Ground Truth /model prediction</b>

<img src="https://user-images.githubusercontent.com/48679574/93964301-40ab9900-fd9a-11ea-820c-243f312cc260.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/93964301-40ab9900-fd9a-11ea-820c-243f312cc260.png" width="400px">



## reffered sites

- [NVIDIA SPADE](https://qiita.com/Phoeboooo/items/ad6c0461ab052aae8e89)
- [Kaggle奮闘記 〜塩コンペ編〜](http://phalanks.hatenablog.jp/entry/2018/12/23/195354)
- [AutoencoderでDenoising, Coloring, そして拡大カラー画像生成](https://qiita.com/MuAuan/items/e5f3e67ee24a776380aa)
