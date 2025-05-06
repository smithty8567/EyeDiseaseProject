# EyeDiseaseProject
We are classifying eye images using a Convolutional Neural Network to determine if the eye has one of three diseases (Cataracts, Glaucoma, or Diabetic Retinopathy) or no disease.

## Data Background
Glaucoma: 
Eye pressure due to the inability to drain fluid damages the optic nerve.

Cataracts:
Aging or injury damages eye tissue, leading to less clear/cloudy vision.

Diabetic Retinopathy:
Blood vessels leak fluid and bleed, which leads to weakened and damaged blood vessels in the eyes from diabetes.

## Transformations of the data
Normalizing our images to the size of 256x256 with the std 1 and mean 0. We also added a canny layer to each image which if the change in color is over the threshhold then the change is classified as an edge.

## Evaluation
Testing score of ~0.81.

![image](https://github.com/user-attachments/assets/fe8a8b9b-b5ad-4c90-8b2f-356e85790bb1)

