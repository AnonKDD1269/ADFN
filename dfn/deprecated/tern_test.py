import torch
import torch.nn as nn
from binlinear import BitLinear
import torchvision
import matplotlib.pyplot as plt
from ternconv import TernConv2d
# simple test script for mnist classification
# using a ternized linear layer
import random 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed to reproduce the results
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ternarized Neural Network

class TernNet(nn.Module):
    def __init__(self):
        super(TernNet, self).__init__()
        self.fc1 = BitLinear(784, 300)
        self.fc2 = BitLinear(300, 100)
        self.fc3 = BitLinear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1.forward_tern(x)
        x = self.fc2.forward_tern(x)
        x = self.fc3.forward_tern(x)
        return x
    

# Load the model
model = TernNet().cuda()

# MNIST Data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64,num_workers = 4, shuffle=False)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=10)

from tqdm import tqdm 
# Train the model
EPOCH = 20
pbar = tqdm(range(EPOCH))
viz_loss = []
correct = 0
acc = 0
train_acc = 0

for epoch in pbar:
    avg_losses = []
    train_correct = 0
    train_total = 0
    model.train()
    for (images, labels) in tqdm(train_loader):
        # Forward pass
        images = images.cuda()
        labels = labels.cuda()
        ori_labels = labels
        # labels to one hot
        one_hot = torch.zeros(labels.size(0), 10).cuda()
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        labels = torch.div(one_hot, 10)

        
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_losses.append(loss.item())
        
        # train accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == ori_labels).sum().item()
        
    
        
    train_acc = 100 * train_correct / train_total
    avg_loss = sum(avg_losses) / len(avg_losses)
    viz_loss.append(avg_loss)

    if epoch % 10== 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
    pbar.set_postfix({'loss': avg_loss, 'val_accuracy': 100* correct / total, 'train_accuracy': train_acc})
print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {} %, Test Accuracy: {} %'.format(epoch+1, EPOCH, avg_loss, train_acc, acc))
        
        # evaluate


plt.plot(viz_loss)
plt.savefig('tern_loss.png')
    
# Save the model checkpoint
torch.save(model.state_dict(), 'tern_model.pt')

# save weight and bias as txt
f = open("tern_weight.txt", "w")
f.write(str(model.fc1.weight))
f.close()

