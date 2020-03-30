import numpy as np
from pathlib import Path, PurePath
import sys
sys.path.insert(1,str(PurePath(Path.cwd()).parent))
sys.path.insert(1,str(PurePath(Path.cwd())))
from NeuralGraph.util import dev, tanimoto_similarity


def example():
    """ demostrate the continuous tanimoto similarity

    x: a neural fingerprint of length 128
    y: a neural fingerprint of length 128
    z: 8 neural fingerprints of length 128. (8x128 numpy matrix)
    """
    x = np.random.rand(128)
    y = np.random.rand(128)
    z = np.random.rand(4, 128)
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")
    print(f"z.shape = {z.shape}")
    print("tanimoto(x,y) =", tanimoto_similarity(x,y))
    print("tanimoto(x,z) =", tanimoto_similarity(x,z))

if __name__ == "__main__":
    example()
