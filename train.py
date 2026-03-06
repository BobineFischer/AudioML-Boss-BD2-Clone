import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataset import GuitarPedalDataset
from model import BossBD2Model

# Hyperparameters
BATCH_SIZE = 32         # Number of slices to feed the model each time
SEQUENCE_LENGTH = 4096  # Length of each slice
EPOCHS = 50             # Number of training epochs
LEARNING_RATE = 1e-3    # Learning rate

# Auto-detect if your Mac GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M-series chip (MPS) to train")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️ Training with CPU.")

def main():
    # Prepare data
    # Load the dataset you prepared (ensure DI.wav and BD.wav are in the same directory)
    dataset = GuitarPedalDataset("DI.wav", "BD.wav", sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = BossBD2Model(hidden_size=32).to(device) # Push the model to GPU
    
    # Loss function: Mean Squared Error (compare the difference between generated and real BD-2 waveforms)
    criterion = nn.MSELoss()
    
    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Start training loop
    print("\n Starting training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train() # Set to training mode
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Push data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Clear the gradient from the previous step
            optimizer.zero_grad()
            
            # Forward propagation (let the model guess what the sound is like)
            predictions = model(inputs)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backpropagation (calculate how to modify parameters)
            loss.backward()
            
            # Update weights (the model learning process)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Calculate the average loss for the entire epoch
        avg_loss = epoch_loss / len(dataloader)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] | Loss: {avg_loss:.6f}")
            
    # Save training results
    end_time = time.time()
    print(f"\n Training completed! Total time: {(end_time - start_time) / 60:.2f} minutes")
    
    torch.save(model.state_dict(), "boss_bd2_model.pth")
    print("Model weights saved as 'boss_bd2_model.pth'")

if __name__ == "__main__":
    main()