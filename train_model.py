from torch.utils.data import random_split, DataLoader,Subset
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import QuickDraw

data = np.load(r"E:\QuickDraw\data\full_combined_data.npy")
images = data[:,:-1]
labels = data[:,-1]

images = images/255
images[images < 0.6] = 0
images[images >= 0.6] = 1

images = images.reshape(86971,28,28)

images_tensor = torch.Tensor(images)
labels_tensor = torch.Tensor(labels).to(dtype= int)

dataset = TensorDataset(images_tensor, labels_tensor)
total = len(dataset)

train_size = int(0.65*total)
test_size = int(0.2*total)
vali_size = total - train_size - test_size

train_data , test_data, vali_data = random_split(dataset,[train_size,test_size,vali_size])
train_loader = DataLoader(train_data,batch_size= 64)
test_loader = DataLoader(test_data,batch_size= 64)
vali_loader = DataLoader(vali_data,batch_size= 64)



epochs = 20

model = QuickDraw(num_class= 29)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss sẽ tính toán softmax và loss

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()  # Đảm bảo mô hình ở chế độ huấn luyện
    
    for data, label in train_loader:
    
        optimizer.zero_grad()  # Xoá gradient cũ
        
        output = model(data)  # Tiến hành dự đoán
        
        loss = loss_fn(output, label)  # Tính toán loss
        running_loss += loss.item()  # Cộng dồn loss
        loss.backward()
        optimizer.step()
        
        total += label.size(0)
        
        _,predicted = torch.max(output,1)
        correct += (predicted == label).sum().item()
    accuracy_train = 100*correct/total
        
    
    running_loss /= len(train_loader)
    
    model.eval()
    validation_loss = 0.0
    correct_val = 0
    total_val = 0
    
    for data, label in vali_loader:
        
        output = model(data)
        loss_val = loss_fn(output,label)
        validation_loss += loss_val.item()  
        
        total_val += label.size(0)
        _,predicted = torch.max(output,1)
        correct_val += (predicted == label).sum().item()
          
    validation_loss /= len(vali_loader)
    accuracy_val = correct_val*100/total_val
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch: {epoch + 1}, Training loss: {running_loss:.4f}, Accuaracy: {accuracy_train: .2f}, Validation loss: {validation_loss:.4f}, Accuaracy: {accuracy_val:.2f}")

torch.save(model.state_dict(), "QuickDraw.pth")
