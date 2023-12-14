import torch
from utils.evaluation import evaluate_recording
import numpy as np

def train_step(recording, X_train_recording, Y_train_recording, model, optimizer, loss_fn):
    optimizer.zero_grad()
    
    X_train_recording = torch.from_numpy(X_train_recording).float()
    Y_train_recording = torch.from_numpy(Y_train_recording).float()

    Y_pred = model(recording, X_train_recording)
    
    loss = loss_fn(Y_pred, Y_train_recording)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_one_epoch(X_train, Y_train, X_test, Y_test, model, optimizer, loss_fn):
    train_loss_iter = [0 for _ in range(len(X_train))]
    test_loss_iter = [0 for _ in range(len(X_train))]

    all_train_indices = [np.arange(X_train[i].shape[0]) for i in range(len(X_train))]

    for recording in range(len(X_train)):
        train_loss_item = train_step(recording, X_train[recording], Y_train[recording], model, optimizer, loss_fn)
        test_loss_item = evaluate_recording(recording, X_test[recording], Y_test[recording], model, loss_fn, plot_num=0)
        
        train_loss_iter[recording] = train_loss_item
        test_loss_iter[recording] = test_loss_item
        
        # if training_iteration % 100 == 0:
        #     print(f'Iteration {training_iteration}, Recording {recording}, Train Loss {train_loss_item}, Test Loss {test_loss_item}')

    return train_loss_iter, test_loss_iter

# def train_one_epoch(X_train, Y_train, X_test, Y_test, model, optimizer, loss_fn):
#     train_loss_iter = [0 for _ in range(len(X_train))]
#     test_loss_iter = [0 for _ in range(len(X_train))]

#     for recording in range(len(X_train)):
#         train_loss_item = train_step(recording, X_train[recording], Y_train[recording], model, optimizer, loss_fn)
#         test_loss_item = evaluate_recording(recording, X_test[recording], Y_test[recording], model, loss_fn, plot_num=0)
        
#         train_loss_iter[recording] = train_loss_item
#         test_loss_iter[recording] = test_loss_item
        
#         # if training_iteration % 100 == 0:
#         #     print(f'Iteration {training_iteration}, Recording {recording}, Train Loss {train_loss_item}, Test Loss {test_loss_item}')

#     return train_loss_iter, test_loss_iter