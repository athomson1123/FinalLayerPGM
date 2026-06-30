import torch
import numpy as np
import copy
import torch.nn as nn
from scipy.stats import wasserstein_distance
import sys

def uncertain(samples):
    '''A synthetic binary classification problem with uncertain labels'''
    input = torch.Tensor(samples, 5)
    output = torch.Tensor(samples, 1)
    for i in range(samples):
        x1 = np.random.binomial(n = 1, p = 0.5)
        x2 = np.random.binomial(n = 1, p = 0.5)
        x3 = np.random.binomial(n = 1, p = 0.5)

        y = (x1 and x2) != (x2 and x3)

        x4 = 0
        x5 = 0

        if i < int(3*samples/4):
            x4 = 0
            x5 = 0
            y = (x1 and x2) != (x2 and x3)
        elif i < int(7*samples/8):
            x4 = 0
            x5 = 1
            y = (x1 and x2) != (x2 and x3)
            if np.random.binomial(n = 1, size=1, p=0.25):
                y = 1-y
        else: # very uncertain
            x4 = 1
            x5 = 0
            y = (x1 and x2) != (x2 and x3)
            if np.random.binomial(n = 1, size=1, p=0.5):
                y = 1-y
        X = [x1, x2, x3, x4, x5]
        input[i, :] = torch.Tensor(X)
        output[i] = torch.Tensor([y])

    return input, output

def uncertain_ground_truth(input_combinations):
    ground_truth = {}
    for combination in input_combinations:
        x1 = int(combination[0])
        x2 = int(combination[1])
        x3 = int(combination[2])
        x4 = int(combination[3])
        x5 = int(combination[4])
        if x4 == 0 and x5 == 0:
            ground_truth[tuple([x1, x2, x3, x4, x5])] = torch.tensor(float((x1 and x2) != (x2 and x3)))
        elif x4 == 0 and x5 == 1:
            ground_truth[tuple([x1, x2, x3, x4, x5])] = torch.tensor(float(0.5*((x1 and x2) != (x2 and x3))+0.25))
        elif x4 == 1 and x5 == 0:
            ground_truth[tuple([x1, x2, x3, x4, x5])] = torch.tensor(float(0.5))

    return ground_truth

def get_predictions(model, X, y):
    predictions = torch.sigmoid(model(X))
    predictions_rounded = torch.round(predictions)

    print('MSE: ', (torch.square(y.flatten() - predictions.flatten())).mean().item(), flush=True)
    print('Accuracy: ', 1-(torch.abs(y.flatten() - predictions_rounded.flatten())).mean().item(), flush=True)

class Model(torch.nn.Module):
    def __init__(self, l1_size=16, l2_size=16, l3_size=16):
        super().__init__()
        self.fc1 = nn.Linear(5, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 1)

        #self.norm = nn.LayerNorm(l2_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu', mode='fan_out')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu', mode='fan_out')
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.sigmoid(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def sample(self, x, num_samples=1000):
        self.train()  # Enable dropout during sampling
        '''Sample from the final layer of the model'''
        with torch.no_grad():
            samples = []
            for _ in range(num_samples):
                sample = self.forward(x)
                samples.append(sample.cpu().numpy())
            samples = np.array(samples).flatten()
        return samples

def evaluate(model, input_combinations, prediction_history, dataset_size, ground_truth, epoch, metric=wasserstein_distance, num_samples=1000):
    with torch.no_grad():
        window = [dataset_size, int(len(prediction_history[0])/2), len(prediction_history[0])]
        alignment_sum_all = 0
        alignment_sum_obs = 0
        nll_sum = 0
        for i in range(len(prediction_history)):
            input_combination = input_combinations[i]
            samples = torch.Tensor(model.sample(x=input_combination.to(device), num_samples=num_samples))

            print('Means for ', tuple(input_combination.int().tolist()), ': ', torch.mean(samples).item(), 
                  ' , ', torch.sigmoid(model(input_combination.to(device))).item())

            print('Alignment: ')
            for j in range(len(window)):
                metric_value = metric(samples, prediction_history[i][-window[j]:])
                print(tuple(input_combinations[i].int().tolist()), ': ', metric_value, ', Window: ', window[j])
                if j == 1:
                    alignment_sum_all += metric_value
                    if not (input_combination[3] == 1 and input_combination[4] == 1):
                        alignment_sum_obs += metric_value

            input_combination = tuple([int(x) for x in input_combination])

            if input_combination in ground_truth:
                print('Predictive Log-Likelihood (Gaussian Approximation): ')

                true_prob = ground_truth[input_combination]

                variance = torch.var(samples)
                
                log_likelihood = (torch.logsumexp(-0.5 * (1/variance) * (samples - true_prob) ** 2, dim=0) 
                                - torch.log(torch.tensor(len(samples))) - 0.5*torch.log(2 * torch.tensor(torch.pi)) - 0.5 * torch.log(variance))
                
                print(input_combination, ' negative log-likelihood: ', -log_likelihood.item())
                nll_sum += -log_likelihood.item()
        print('Average Alignment (All): ', alignment_sum_all/len(prediction_history), ', Epoch (', epoch+1, ')')
        print('Average Alignment (Observed): ', alignment_sum_obs/len(prediction_history), ', Epoch (', epoch+1, ')')
        print('Average Negative Log-Likelihood: ', nll_sum/len(prediction_history), ', Epoch (', epoch+1, ')')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
      print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    if len(sys.argv) > 1:
        learning_rate = float(sys.argv[1])
        saved_decay = float(sys.argv[2])
        seed = int(sys.argv[3])
    else:
        learning_rate = 1e-3
        saved_decay = 0.9999
        seed = 123

    print('Learning Rate: ', learning_rate)
    print('Saved Decay: ', saved_decay)
    print('Seed: ', seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(256,256,256).to(device)

    initial_state = copy.deepcopy(model.state_dict())

    base_params = []
    final_layer_params = []

    for name, param in model.named_parameters():
        if 'mrf1' in name and 'b' not in name:  
            final_layer_params.append(param)
        else:
            base_params.append(param)
    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0)
    loss_function = torch.nn.BCEWithLogitsLoss()

    print(model)

    X, y = uncertain(10000) #
    X_test, y_test = uncertain(500)
    X_test, y_test = X_test.to(device), y_test.to(device)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    print('y counts:', torch.sum(y).item())

    epochs = 100
    report = 10
    num_samples = 10000
    num_checkpoints = len(dataset)

    X_len = len(X)
    checkpoint_interval = X_len // num_checkpoints

    saved_states = 0
    model_sum = copy.deepcopy(model)
    model_avg = copy.deepcopy(model_sum)
    for name, param in model_sum.named_parameters():
        if param.requires_grad:
            param.data.zero_()

    prediction_history = []

    base = torch.tensor([0, 1])
    input_combinations = torch.cartesian_prod(base, base, base, base, base).float()

    for input_combination in input_combinations:
        prediction_history.append([])

    ground_truth = uncertain_ground_truth(input_combinations)

    # Begin Training:

    for epoch in range(epochs):

        model.train()
        if epoch % report == 0:
            total_norm = 0
            batch_count = 0

        cur_step = 0
        for batch, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.to(device), targets.to(device)

            if cur_step % checkpoint_interval == 0:
                saved_states = 1 + saved_decay*saved_states
                for name, param in model_sum.named_parameters():
                    if param.requires_grad:
                        current_weights = model.state_dict()[name].data
                        param.data.copy_(current_weights + saved_decay * param.data)
            cur_step += 1

            model.eval()
            with torch.no_grad():
                for i in range(len(input_combinations)):
                    prediction_history[i].append(torch.sigmoid(model(input_combinations[i])).item())
            model.train()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs.t(), targets.t())
            loss.backward()

            optimizer.step()


        # Evaluation:

        if (epoch+1) % report == 0 or epoch == 0:
            print(f"Epoch {epoch+1}")
            model.eval()

            print('Learning Rate: ', optimizer.param_groups[0]['lr'])

            print('Predictions: ')
            get_predictions(model, X_test, y_test)
            print('Averaged Predictions: ')
            get_predictions(model_avg, X_test, y_test)

            # Update the averaged model (EMA)
            if saved_states > 0:
                model_avg = copy.deepcopy(model_sum)
                for name, param in model_avg.named_parameters():
                    if param.requires_grad: 
                        param.data = model_sum.state_dict()[name].data / saved_states

            model_avg.eval()

            print('Recent Model: ')
            
            evaluate(model, input_combinations, prediction_history, X_len, ground_truth, epoch, metric=wasserstein_distance, num_samples=num_samples)

            print('\n')

            print('Averaged Model: ')

            evaluate(model_avg, input_combinations, prediction_history, X_len, ground_truth, epoch,metric=wasserstein_distance, num_samples=num_samples)

            print('\n\n\n\n')

            model.train()