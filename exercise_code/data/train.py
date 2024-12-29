
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    num_epochs: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    # Initializing the list for storing the loss and accuracy

    train_loss_history = []  # loss
    train_acc_history = []  # accuracy

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0.0
        total = 0

        # Iterating through the minibatches of the data
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.argmax(outputs.data, 1) # indices that model will predict.
            correct += (predicted == labels).sum().item()


                    
            
            
            ########################################################################
            # TODO:                                                                #
            # Complete the training code. This includes                            #
            # 1. moving the batch data to the correct device
            
            # 2. feeding the input through the model                               #
            # 3. calculating the loss                                              #
            # 4. running an optimizer step NOTE: do not forget to call             #
            #    optimizer.zero_grad() at the appropriate time                     #
            # To keep track of the training it would be helpful, if you update     #
            # the variables that are reset after 1000 minibatches of an epoch      #
            # running_loss = ...    keeps track of the loss since the last reset   #
            # correct = ...         keeps track of the correctly pred. samples     #
            # total = ...           keeps track of the total number of samples     #
            ########################################################################


            

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

            # Print statistics to console
            if i % 1000 == 999:  # print every 1000 mini-batches
                running_loss /= 1000
                correct /= total
                print(
                    "[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%"
                    % (epoch + 1, i + 1, running_loss, 100 * correct)
                )
                train_loss_history.append(running_loss)
                train_acc_history.append(correct)
                running_loss = 0.0
                correct = 0.0
                total = 0

    print("FINISH.")

    return train_loss_history, train_acc_history
