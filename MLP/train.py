import itertools
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Train the model and validate after each epoch, including training accuracy.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        epochs (int): Number of training epochs
        
    Returns:
        history (dict): Training and validation loss and accuracy history
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            
            # Training accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return history


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        
    Returns:
        avg_loss (float): Average loss over dataset
        accuracy (float): Classification accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed for evaluation
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)  # sum over batch
            preds = outputs.argmax(dim=1)  # get predicted class
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def grid_search(model_class, train_loader, val_loader, param_grid, epochs=10):
    """
    Perform grid search over hyperparameters.

    Args:
        model_class: The neural network class to instantiate
        train_dataset: Training dataset (PyTorch Dataset)
        val_dataset: Validation dataset (PyTorch Dataset)
        param_grid (dict): Dictionary with hyperparameters to search
        epochs (int): Number of training epochs for each configuration
    Returns:
        best_params (dict): Best hyperparameter configuration
        best_val_acc (float): Best validation accuracy achieved
    """
    keys, values = zip(*param_grid.items())
    best_val_acc = 0
    best_params = None

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        print(f"\nTesting config: {params}")
        
        # DataLoaders
        
        # Model
        model = model_class(input_size=784, hidden_size=params["hidden_size"], output_size=10)
        
        # Optimizer
        if params["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        elif params["optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
        else:
            raise ValueError(f"Unknown optimizer {params['optimizer']}")
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        
        # Train
        history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)
        
        # Check best validation accuracy
        val_acc = history["val_acc"][-1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
    
    print("\nBest hyperparameters:", best_params)
    print("Validation accuracy:", best_val_acc)
    
    return best_params, best_val_acc   