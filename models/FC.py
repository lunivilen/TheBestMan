import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        # Полносвязные слои
        self.fc1 = nn.Linear(74, 128, device=device)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64, device=device)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1, device=device)
        self.sig = nn.Sigmoid()

        # Функция активации
        self.relu = nn.PReLU(num_parameters=1, init=0.25).to(device)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Прямой проход через полносвязные слои
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.sig(self.fc3(x))
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs, max_norm=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def top_k_accuracy(model, dataloader):
    model.eval()
    deviation = []
    mse = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, target in dataloader:
            predicted = model(inputs)

            # собираем MSE
            loss = criterion(predicted, target)
            mse.append(loss.item())

            # собираем MAPE
            info = torch.mean(abs((target - predicted) / target))
            deviation.append(info)

    # рассчитываем среднее отклонение
    deviation = torch.tensor(deviation)
    deviation = torch.mean(deviation)

    # рассчитываем среднее mse
    mse = sum(mse) / len(mse)
    return deviation, mse


def get_prediction(model, data_loader):
    model.eval()
    result = []
    target_list = []
    with torch.no_grad():
        for inputs, target in data_loader:
            result.extend(model(inputs).tolist())
            target_list.extend(target.tolist())
    result = [item for sublist in result for item in sublist]
    target_list = [item for sublist in target_list for item in sublist]
    return result, target_list
