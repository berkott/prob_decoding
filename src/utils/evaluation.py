import torch
import plotly.graph_objects as go

def evaluate_recording(recording, X_test_recording, Y_test_recording, model, loss_fn, plot_num=10):
    # Test the model
    model.eval()
    
    X_test_recording = torch.from_numpy(X_test_recording).float()
    Y_test_recording = torch.from_numpy(Y_test_recording).float()
    Y_pred = model(recording, X_test_recording)
    loss = loss_fn(Y_pred, Y_test_recording)
    
    for t in range(plot_num):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=Y_test_recording[t], name='Actual'))
        fig.add_trace(go.Scatter(y=Y_pred[t].detach().numpy(), name='Predicted'))
        fig.update_layout(title=f'Actual and Predicted Wheel Speeds, Recording {recording}, Trial {t}', xaxis_title='Time', yaxis_title='Value')
        fig.show()

    return loss.item()