import torch
import torch.nn as nn

class BossBD2Model(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        """
        :param hidden_size: Number of LSTM hidden layer neurons. 32 is suitable for quick testing. Can be increased to 64 or 96 for better audio quality.
        :param num_layers: Number of LSTM layers. 1 layer is sufficient. 2 layers can learn more complex dynamics but train slower.
        """
        super(BossBD2Model, self).__init__()
        
        # 1. Core component: LSTM layer
        # input_size=1 because audio has only one amplitude value at each time point (mono)
        # batch_first=True means the input data dimension order is (Batch_Size, Sequence_Length, Input_Size)
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # 2. Output layer: Fully connected layer (Linear Layer)
        # Compresses the multi-dimensional features (hidden_size) from LSTM output back to 1-dimensional audio amplitude values
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward propagation
        x dimensions: (batch_size, sequence_length, 1)
        """
        # lstm_out dimensions: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # predictions dimensions: (batch_size, sequence_length, 1)
        predictions = self.linear(lstm_out)
        
        return predictions

# TEST CODE
if __name__ == "__main__":
    # Instantiate the model
    model = BossBD2Model(hidden_size=32)
    print("Model built successfully!")
    
    # Simulate a batch of dummy data to test if the model runs correctly
    # Assume Batch_Size=16, Sequence_Length=4096, Input_Size=1
    dummy_input = torch.randn(16, 4096, 1)
    
    # Forward propagation test
    dummy_output = model(dummy_input)
    
    print(f"\nInput dummy data dimensions: {dummy_input.shape}")
    print(f"Model output data dimensions: {dummy_output.shape}")
    
    if dummy_input.shape == dummy_output.shape:
        print("correct")