# Deep Learning Approach for Tabular Data: Data Transformation 
These notebooks include an implementation of a data transformation method, dynamic weighted tabular method, used to transform tabular data into images with the aim of experimenting with the use of convolutional neural networks.

We apply DWTM through using an open source implementation c on GitHub. We use the DataProcessingCategorical class to allow the model to calculate the required feature weights using the Chi-square measure since our data includes categorical features. DWTM uses ImageDatasetCreation class to create the image canvas from the input data point. Using this class through the package directly did not succeed as we encountered errors stating accessing out of bound elements, accordingly, we edit the class through adding boundary conditions to mitigate the errors we earlier encountered, and we use this class without accessing it from the python package. We set the size of the output images as (224,224), which is the minimum accepted size by the deep convolutional neural network architecture we are using (Inception V1). We train the model on GPU using AUB’s HPC (Octopus).

We train the model using F1 loss function as a criterion to train Inception V1 model. During the transformation of the non-ICU delirium AUBMC extracted dataset, we observed that DWTM is unable to apply its transformation to columns with constant values.

### References
[1] Iqbal, M. I., Mukta, M., Hossain, S., & Hasan, A. R. (2022). A Dynamic Weighted Tabular Method for Convolutional Neural Networks. arXiv preprint arXiv:2205.10386.
