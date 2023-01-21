import torch
import os


class QNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size).cuda()
        self.linear2 = torch.nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model_name.pth'):
        model_folder_path = ''
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QLearning:
    def __init__(self, model, learning_rate=0.1, discount_factor=0.99):
        # Linear NN defined above.
        self.model = model
        # Setting learning rate and discount factor
        self.alpha = learning_rate
        self.gamma = discount_factor

        # optimizer for weight and biases updation
        self.optimer = torch.optim.Adam(model.parameters(), lr=self.alpha)
        # Mean Squared error loss function
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.long).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()

        # if only one parameter to train , then convert to tuple of shape (1, x)
        if (len(state.shape) == 1):
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Predicted Q value with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()  # backward propagation of loss

        self.optimer.step()