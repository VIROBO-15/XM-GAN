# MICCAI-2023

<h1 align="center">Cross-modulated Few-shot Image Generation for
Colorectal Tissue Classification [MICCAI 2023]</h1>
<!-- <p align="center">for 3D-Aware Image Generation</p> -->

<p align="center">
  <img width="90%" src="Fig/Demo-MICCAI.gif"/>
</p>

**Cross-modulated Few-shot Image Generation for Colorectal Tissue Classification**<br>
[Amandeep Kumar](https://virobo-15.github.io/), [Ankan Kumar Bhunia](https://ankanbhunia.github.io/), [Sanath Narayan](https://sites.google.com/view/sanath-narayan/home), [Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), [Rao Muhammad Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en), [Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en),[Fahad Shahbaz Khan](https://scholar.google.com/citations?user=zvaeYnUAAAAJ&hl=en)

###  [Paper](https://arxiv.org/pdf/2304.01992.pdf)

Abstract: *In this work, we propose a few-shot colorectal tissue image generation method for addressing the scarcity of histopathological training data for rare cancer tissues.  Our  few-shot generation method, named XM-GAN, takes one base and a pair of reference tissue images as input and  generates high-quality yet diverse images. Within our XM-GAN, a novel controllable fusion block densely aggregates local regions of reference images based on their similarity to those in the base image, resulting in locally consistent features. To the best of our knowledge, we are the first to investigate few-shot generation in colorectal tissue images. We evaluate our few-shot colorectral tissue image generation by performing extensive qualitative, quantitative and subject specialist (pathologist) based evaluations. Specifically, in specialist-based evaluation, pathologists could differentiate between our XM-GAN generated tissue images and real images only  55\% time.  Moreover, we utilize these generated images as data augmentation to address the few-shot tissue image classification task, achieving a gain of 4.4\% in terms of mean accuracy over the vanilla few-shot classifier..*

## :rocket: News
* Code will released soon.

### Setup

- **Get code**
```shell 
git clone https://github.com/VIROBO-15/XM-GAN.git
```

- **Build environment**
```shell
cd XM-GAN
# use anaconda to build environment 
conda create -n xmgan python=3.6
conda activate xmgan
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

```

### Dataset
**Download the data**:
You can download the colorectral cancer images [here](https://zenodo.org/record/53169). Unzip and rename the folder as data.


**Run the following code to save the image in the .npy format**:

```shell
python create_npy.py --dir data
```

## Training
```shell
python train.py --conf configs/cancer.yaml --output_dir results/cancer --gpu 0
```

* You may also customize the parameters in `configs`.


## Testing
```shell
python test.py --name result/cancer --gpu 0 --conf configs/cancer.yaml 
```

The generated images will be saved in `results/cancer/test`.

## Citation

If you find our work helpful, please **star🌟** this repo and **cite📑** our paper. Thanks for your support!

```
@article{kumar2023cross,
  title={Cross-modulated Few-shot Image Generation for Colorectal Tissue Classification},
  author={Kumar, Amandeep and Narayan, Sanath and Cholakkal, Hisham and Anwer, Rao Muhammad and Laaksonen, Jorma and Khan, Fahad Shahbaz and others},
  journal={arXiv preprint arXiv:2304.01992},
  year={2023}
}
```

## Acknowledgement
Our code is designed based on [LoFGAN](https://github.com/edward3862/LoFGAN-pytorch).





