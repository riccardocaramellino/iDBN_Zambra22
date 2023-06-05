
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if not(num_classes==40):
          self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if not(self.num_classes==40):
          out = self.softmax(out)

        return out


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if model.num_classes==40:
          loss = criterion(output, target.float())
          predicted = torch.sigmoid(output) >= 0.5
        else:
          loss = criterion(output, target.long())
          predicted = torch.argmax(output, axis = 1)
        # Aggiungi regolarizzazione L2 ai pesi della rete
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += torch.numel(target)
        correct += (predicted == target).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = correct / total
    return train_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if model.num_classes==40:
              test_loss = criterion(output, target.float()).item()
              predicted = torch.sigmoid(output) >= 0.5
            else:
              test_loss = criterion(output, target.long()).item()
              predicted = torch.argmax(output, axis = 1)
            total += torch.numel(target)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    return test_loss, accuracy


def main(train_loader, test_loader, num_classes=40):
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 64
    epochs = 10

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model initialization
    model = ResNet(num_classes=num_classes).to(device)

    # Utilizza la funzione di perdita BCEWithLogitsLoss
    if num_classes==40:
      criterion = nn.BCEWithLogitsLoss()
    else:
      criterion =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Utilizza un learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

    # Training and evaluation
    for epoch in range(1, epochs+1):

        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model,device, test_loader, criterion)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        scheduler.step(test_loss)

    return model