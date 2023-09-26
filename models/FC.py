import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        # Сверточные слои
        # self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1, device=device)

        # Полносвязные слои
        self.fc1 = nn.Linear(13, 32, device=device)
        self.fc2 = nn.Linear(32, 16, device=device)
        self.fc3 = nn.Linear(16, 1, device=device)

        # Функция активации
        self.relu = nn.PReLU(num_parameters=1, init=0.25).to(device)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Прямой проход через полносвязные слои
        x = self.relu(self.fc1(x))
        self.dropout(x)
        x = self.relu(self.fc2(x))
        self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
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
