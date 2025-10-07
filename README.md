# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY


## Neural Network Model
Include the neural network model diagram.


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transform

### STEP 2: 
Load training and testing datasets using ImageFolder



### STEP 3: 
Visualize sample images from the dataset.



### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.



### STEP 5: 

Define loss function (CrossEntropyLoss) and optimizer (Adam). Train the model and plot the loss curve.





### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.





## PROGRAM

### Name:MARELLA HASINI

### Register Number:212223240083

```python

import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchsummary import summary

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

!unzip -qq ./chip_data.zip -d data
dataset_path='./data/dataset'
train_dataset=datasets.ImageFolder(root=f"{dataset_path}/train",transform=transform)
test_dataset=datasets.ImageFolder(root=f"{dataset_path}/test",transform=transform)

def show_sample_images(dataset,num_images=5):
  fig,axes=plt.subplots(1,num_images,figsize=(5,5))
  for i in range(num_images):
    image,label=dataset[i]
    image=image.permute(1,2,0)
    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label])
    axes[i].axis("off")
  plt.show()
print("Number of training samples:",len(train_dataset))
first_image,label=train_dataset[0]
print("Image shape:",first_image.shape)

print("Number of testing samples:",len(test_dataset))
first_image1,label=test_dataset[0]
print("Image shape:",first_image1.shape)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)
device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
summary(model,input_size=(3,224,224))

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
summary(model,input_size=(3,224,224))

for param in model.features.parameters():
  param.requires_grad=False

criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

def train_model(model,train_loader,test_loader,num_epochs=100):
  train_losses=[]
  val_losses=[]
  for epoch in range(num_epochs):
    running_loss=0.0
    for images,labels in train_loader:
      images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels.unsqueeze(1).float())
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    train_losses.append(running_loss/len(train_loader))

    model.eval()
    val_loss=0.0
    with t.no_grad():
      for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        val_loss+=loss.item()
    val_losses.append(val_loss/len(test_loader))
    model.train()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")
  plt.figure(figsize=(8,6))
  plt.plot(range(1,num_epochs+1),train_losses,label="Train Loss",marker="o")
  plt.plot(range(1,num_epochs+1),val_losses,label="Validation Loss",marker="s")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Training and validation Loss")
  plt.legend()
  plt.show()

device=t.device("cuda" if t.cuda.is_available() else "cpu")
model=model.to(device)
train_model(model,train_loader,test_loader)

def test_model(model,test_loader):
  model.eval()
  correct=0
  total=0
  all_preds=[]
  all_labels=[]

  with t.no_grad():
    for images,labels in test_loader:
      images=images.to(device)
      labels=labels.float().unsqueeze(1).to(device)

      outputs=model(images)
      probs=t.sigmoid(outputs)
      predicted=(probs > 0.5).int()
      total+=labels.size(0)
      correct+=(predicted==labels.int()).sum().item()

      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy().astype(int))
  accuracy=correct/total
  print(f"Test Accuracy: {accuracy:.4f}")

  class_names=['Negative','Positive']
  cm=confusion_matrix(all_labels,all_preds)
  plt.figure(figsize=(6,5))
  sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()


  print("Classification Report :")
  print(classification_report(all_labels,all_preds,target_names=class_names))
test_model(model,test_loader)

def predict_image(model,image_index,dataset):
  model.eval()
  image,label=dataset[image_index]
  with t.no_grad():
    image_tensor = image.unsqueeze(0).to(device)
    output=model(image_tensor)
    _,predicted=t.max(output,1)
  class_names=dataset.classes

  image_to_display=transforms.ToPILImage()(image)

  plt.figure(figsize=(4,4))
  plt.imshow(image_to_display)
  plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
  plt.axis('off')
  plt.show()
predict_image(model,image_index=55,dataset=test_dataset)
predict_image(model,image_index=25,dataset=test_dataset)




```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="718" height="540" alt="image" src="https://github.com/user-attachments/assets/8a59bf42-8e82-4e76-aee6-7392e88ef989" />


## Confusion Matrix


<img width="654" height="621" alt="image" src="https://github.com/user-attachments/assets/5e9d1671-17f3-4b54-8322-8024af4fed2f" />


## Classification Report

<img width="564" height="248" alt="image" src="https://github.com/user-attachments/assets/251b0e1a-aaee-4a81-bdce-5282d29ce44e" />


### New Sample Data Prediction

<img width="456" height="465" alt="image" src="https://github.com/user-attachments/assets/0848a1fd-0574-4a4f-8365-4b7bd322af60" />

<img width="435" height="468" alt="image" src="https://github.com/user-attachments/assets/775f9d6e-ac43-4c86-905a-62c4c434dddb" />

## RESULT
Hence a Neural Network transfer model is developed using transfer learning
