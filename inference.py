import torch
import soundfile as sf
import numpy as np
from model import BossBD2Model

def generate_ai_pedal_sound_ola_batched(test_audio_path, output_audio_path):
    print("Waking up AI effects processor (enabled OLA seamless stitching + memory safety protection)...")
    
    # 1. Hardware acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    # 2. Load model
    model = BossBD2Model(hidden_size=32).to(device)
    model.load_state_dict(torch.load("boss_bd2_model.pth", weights_only=True))
    model.eval() 
    
    # 3. Read audio
    data_in, sr = sf.read(test_audio_path, dtype='float32')
    if data_in.ndim > 1:
        data_in = np.mean(data_in, axis=1)
        
    # OLA Core Algorithm Start (smoothly process long audio without memory overflow)
    seq_len = 4096
    hop_size = 2048  
    
    data_in = np.pad(data_in, (seq_len, seq_len), 'constant')
    window = np.hanning(seq_len)
    out_sig = np.zeros(len(data_in))
    
    # Split long audio into chunks
    chunks = []
    starts = []
    for i in range(0, len(data_in) - seq_len, hop_size):
        chunks.append(data_in[i:i+seq_len])
        starts.append(i)
        
    all_chunks_np = np.array(chunks)
    num_total_chunks = len(all_chunks_np)
    
    # Core fix: Set a safe batch size to prevent GPU memory overflow
    batch_size = 32 
    output_chunks_list = []
    
    print(f"Audio is long, split into {num_total_chunks} chunks, rendering in batches...")
    
    with torch.no_grad():
        # Feed data to GPU in batches
        for i in range(0, num_total_chunks, batch_size):
            batch_np = all_chunks_np[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_np).view(len(batch_np), seq_len, 1).to(device)
            
            # Model inference
            out_tensor = model(batch_tensor)
            
            # Remove the last dimension (input_size=1), change from (batch, seq_len, 1) to (batch, seq_len)
            out_np = out_tensor.cpu().numpy().squeeze(axis=-1)
            
            # Edge case handling: if the last batch has only 1 chunk, squeeze will also remove the batch dimension, need to restore
            if out_np.ndim == 1:
                out_np = np.expand_dims(out_np, axis=0)
                
            output_chunks_list.append(out_np)
            
    # Concatenate all batch results on the batch dimension
    output_chunks = np.concatenate(output_chunks_list, axis=0)
    
    # OLA overlap-add
    for i, start in enumerate(starts):
        windowed_chunk = output_chunks[i] * window
        out_sig[start:start+seq_len] += windowed_chunk
        
    out_sig = out_sig[seq_len:-seq_len]
    # OLA Core Algorithm End

    # 4. Export
    sf.write(output_audio_path, out_sig, sr)
    print(f"audio saved to: {output_audio_path}")

if __name__ == "__main__":
    # Execute generation
    generate_ai_pedal_sound_ola_batched("test_di.wav", "ai_bd2_output.wav")