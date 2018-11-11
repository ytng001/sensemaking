To run the model, launch train.py to start training.
To evaluate the model, run evaluate.py.

In boht train.py and evaluate.py, the models can be toogle to (RadialNetBasicFC, RadialNetBasic, RadialNet, RadialNetInceptionAndTransform)
line 19 and line 20 respectively.

RadialNetBasic - RBF Layer + Inception MLP
RadialNetBasicFC - RBF Layer + MLP
RadialNetInceptionAndTransform - RBF Layer + Affine Transform + Inception MLP
RadialNet - RBF Layer + Affine Transform + Inception MLP + Feature Transform

Special thanks to Charles Qi for sharing the code for PointNet.
https://github.com/charlesq34/pointnet


