# Solution for Airbus Ship Detection Challenge

Solution plan was developed based on solutions published by 4th, 6th, 9th, 11th and 12th
places of this challenge on Kaggle + own experience.

Python version used: 3.10.13

The first step is to convert labels from run-length format to images, 
considering what each label could be described by several rows in csv.
Optionally split folder with images to smaller sub-folders, for glory of Ubuntu's file manager.
`0_Service.ipynb` notebook does this.

There are a lot of empty images in dataset (>75%).
Images in the train dataset have intersections. To split it to the train, val and test sets 
we need to ensure there will be no intersection between images in the train and the val sets. 
We group intersecting images and take only 1 image from each group. 
With this approach we loose a lot of data, but according to the top solutions on the kaggle it is enough.
Then, the EDA analysis of selected 8635 images is performed at the notebook `1_EDA.ipynb`.
Key findings:
1. Most images contain 1 ship, 95% images contains up to 5 ships
2. Ships have sizes from 2 to 25904 pixels. The biggest number of ships have size 25-125px
3. If there is a big ship on the image, it is unlikely to have small ship on the same image
4. There are images with ships on the edge, 
so segmentation model will not see entire ship on the image all the time

There are a lot of very small ships on the images. Segmentation models are known to have
problems with predicting small object (information gets lost to the end on the nn).
To couple with this problem, images are splitted to the parts of size (128, 128) 
(only pieces with ships are selected, 24718 items),
each part is up-scaled on the input to the NN to the size of (1024, 1024).
Images are splitted to the train, val and test sets stratified by ship size.
I got 20369 train, 271 val and 4078 test image parts.
Meta files in format required by different libs are generated.
This is performed at `2_Data_preparation.ipynb`.

Next step is to train segmentation model using: BCE loss, dice score as metric during train, 
F2 score as test metric, and apply cyclic LR during training.

I have managed to train segmentation models pp-mobileseg, pp-lite-seg using PaddlePaddle framework,
but they achieved only Mean Intersection Over Union ~0.8 and Dice ~0.9 (`3_Segmentation_paddleseg_colab.ipynb`).

I tried to train U-net with ResNet 34 backbone using fastai library, but I haven't succeed,
the lib have difficult documentation... (`3_Segmentation_fastai_colab.ipynb`)
Also I have found the lib `segmentation_models` with U-net model with replaceable backbone based on tensorflow, but 
I haven't tried to train it. (`3_Segmentation_tf.ipynb`).

The future steps:
1. Train U-net with ResNet34 backbone (+try different backbones)
2. Evaluate model on negative image parts, add negative samples to the dataset 
(to have 10-25% of them. This should a little bit reduce FP, but not too much)
3. Apply algorithm to select distinct ships on the segmentation output (likely using errosion + waterched algorithm)
4. Train classifier to separate TP from FP. (use pre-trained backbone, like EfficientNet).
5. Test time augmentation using horizontal and vertical flips
6. Try to “rectanglize” unet output masks by finding minimal bounding box and shrink it to the same size (of cluster)
7. Different pixel selection threshold for different sizes of ships
8. Add ship borders or split line to the labels for ships separation
9. Cut corners of the label masks
