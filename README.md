# PGHE-Net

#### **PGHE-Net: Physically Guided Dehazing Network with High-Frequency Enhancement for Remote Sensing Images**

**Abstract:** Haze in the atmosphere diminishes the visibility of optical remote sensing images (RSIs), making optical RSI dehazing an essential pre-processing step for geographic monitoring and advanced computer vision tasks. While current CNN-based dehazing technologies have shown impressive results with natural images, they often struggle with detail blurring and color distortion when applied to RSIs. In this paper, we propose a physically guided dehazing network with high-frequency enhancement (PGHE-Net), which includes a channel attention-based high-frequency enhancement (CAHE) and a physical guided-based refiner (PGR). The CAHE is crafted to compensate for the loss of high-frequency information of features, which utilizes multiscale feature fusion and gated attention mechanisms to mine haze features of different distributions and concentrations. To further enhance the restoration of features in hazy regions, the PGR is developed to guide spatial attention to interact with channel attention utilizing a physical model to refine hierarchically and progressively reconstructed images at different scales. Experiments on three public benchmark RSIs demonstrate the superiority of the proposed method in RSI dehazing.

![](C:\Users\pt\Desktop\MyPapers\PGHE_4月9日提交版本\elsarticle\Figures\network.jpg)

#### **Operational environment requirements**

```
python                    3.9.18
torch                     2.1.2
torchvision               0.16.2
numpy                     1.26.4
matplotlib                3.9.2
scikit-image              0.22.0
einops                    0.8.0
```

#### **Train**

For those who wish to perform rapid training on their custom datasets, we provide a straightforward training code in the train.py file, enabling training of our MABDT. Please refer to the example usage within the file to train the model on your datasets.

```python
python main.py --data_dir ./datasets_name/ --mode train --batch_size 8 --learning_rate 1e-4 --numepoch 1000 --num_worker 1 --valid_freq 10
```

#### Test

To evaluate our EMPF-Net on your own datasets or publicly available datasets, you can utilize the following example usage to conduct experiments.

```python
python main.py --data_dir ./datasets_name/ --mode test --test_model ./results/PGHE/ots/Best.pkl --save_image True
```

#### Contact us

If I have any inquiries or questions regarding our work, please feel free to contact us at yinjie13@stu.cdut.edu.cn.
