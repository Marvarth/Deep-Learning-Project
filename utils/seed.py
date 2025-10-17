import random
import numpy as np
import torch

# Funzione per fissare i semi casuali (reproducibilità)
# Garantisce che ogni esecuzione produca gli stessi risultati
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)