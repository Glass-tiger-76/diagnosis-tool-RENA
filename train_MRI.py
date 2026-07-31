

import torch
import torch.nn as nn
from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

device="cuda" if torch.cuda.is_available() else 'cpu'
print(device)


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data=datasets.ImageFolder("data/Training",transform=transform)
test_data=datasets.ImageFolder("data/Testing",transform=transform)

train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(test_data,batch_size=32)


model=models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
  param.requires_grad=False

for param in model.layer4.parameters():
  param.requires_grad=True




model.fc=nn.Linear(model.fc.in_features,4)
model=model.to(device)
weights = torch.tensor([2.0, 1.0, 1.0, 1.0]).to(device) #glioma is index 0
loss_fn=nn.CrossEntropyLoss(weight=weights)
optimizer=torch.optim.Adam(([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3}

]))


for epoch in range(10):
  model.train()
  for images,labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    pred=model(images)
    loss=loss_fn(pred,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
all_preds, all_labels = [], []
correct = total = 0
class_correct = [0]*4
class_total   = [0]*4

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for lbl, prd in zip(labels, preds):
            class_total[lbl] += 1
            if lbl == prd:
                class_correct[lbl] += 1

print("classes:", train_data.classes)
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

print("\noverall accuracy:", round(correct/total, 4))
for i, name in enumerate(train_data.classes):
    print(f"  {name}: {class_correct[i]}/{class_total[i]} = {class_correct[i]/class_total[i]:.3f}")

