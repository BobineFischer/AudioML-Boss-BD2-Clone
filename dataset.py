import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class GuitarPedalDataset(Dataset):
    def __init__(self, input_file, target_file, sequence_length=4096):
        print("Loading and preprocessing audio data, please wait...")
        
        # Abandon torchaudio entirely (wrong version), use more stable soundfile for direct reading
        # Data read by soundfile is directly in standard numpy array format
        data_in, self.sr_in = sf.read(input_file, dtype='float32')
        data_tar, self.sr_tar = sf.read(target_file, dtype='float32')
        
        # Check if sampling rates match
        if self.sr_in != self.sr_tar:
            raise ValueError(f"Sampling rate mismatch! Input: {self.sr_in}, Target: {self.sr_tar}")
        self.sr = self.sr_in
        
        # Handle dimensions: soundfile mono reading produces (length,), stereo is (length, channels)
        # We need to unify it to PyTorch's preferred (channels, length)
        if data_in.ndim == 1:
            data_in = data_in.reshape(-1, 1)
        if data_tar.ndim == 1:
            data_tar = data_tar.reshape(-1, 1)
            
        # Convert to PyTorch Tensor and transpose (T)
        self.input_waveform = torch.from_numpy(data_in).T
        self.target_waveform = torch.from_numpy(data_tar).T
        
        # 2. Force conversion to mono (if multiple channels, average them)
        if self.input_waveform.shape[0] > 1:
            self.input_waveform = torch.mean(self.input_waveform, dim=0, keepdim=True)
        if self.target_waveform.shape[0] > 1:
            self.target_waveform = torch.mean(self.target_waveform, dim=0, keepdim=True)
            
        # 3. Truncate and align lengths
        min_len = min(self.input_waveform.shape[1], self.target_waveform.shape[1])
        self.input_waveform = self.input_waveform[:, :min_len]
        self.target_waveform = self.target_waveform[:, :min_len]
        
        self.sequence_length = sequence_length
        self.num_chunks = min_len // sequence_length
        
        print(f"Loading successful!")
        print(f"Sampling rate: {self.sr} Hz")
        print(f"Available audio length: {min_len / self.sr:.2f} seconds")
        print(f"Generated {self.num_chunks} training chunks (each chunk has {sequence_length} samples)")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        
        x = self.input_waveform[0, start_idx:end_idx]
        y = self.target_waveform[0, start_idx:end_idx]
        
        # RNN requires dimensions of (sequence_length, input_size)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        
        return x, y

def visualize_chunk(x, y):
    """Plot a waveform comparison graph of a chunk, so you can intuitively see the data"""
    plt.figure(figsize=(10, 4))
    plt.plot(x.numpy(), label='Input (DI Clean)', alpha=0.7)
    plt.plot(y.numpy(), label='Target (BD-2 Distorted)', alpha=0.7)
    plt.title('Waveform of a single training chunk')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Test our dataset
    dataset = GuitarPedalDataset("DI.wav", "BD.wav", sequence_length=4096)
    
    # Extract the first chunk to see
    x_sample, y_sample = dataset[0]
    print(f"\nInput chunk dimensions: {x_sample.shape} -> represents (4096 time steps, 1 feature)")
    print(f"Target chunk dimensions: {y_sample.shape}")
    
    # Plot
    visualize_chunk(x_sample, y_sample)