import numpy as np
import os

# merge 
def merge_embeddings(data_dir):
    file_names = os.listdir(data_dir)
    embeddings = {}
    for file_name in file_names:
        if file_name.endswith('.npz'):
            data = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
            embeddings.update(data['embeddings'].item())
    save_path = os.path.join(data_dir, 'embeddings.npz')
    np.savez(save_path, embeddings=embeddings)
    print(f"Saved embeddings to {save_path}")
    return embeddings

data_dirs = [
   "SOLAMI_data/DLP-MoCap/embeddings",
    "SOLAMI_data/HumanML3D/embeddings",
    "SOLAMI_data/Inter-X/embeddings",
]


for data_dir in data_dirs:
    merge_embeddings(data_dir)
