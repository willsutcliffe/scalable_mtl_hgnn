import numpy as np

# Remplace par ton chemin réel vers un exemple
graph = np.load("/eos/user/e/ebornand/DFEI/FullMC/npy_array/magdown/training_dataset/input_0.npy", allow_pickle=True).item()
target = np.load("/eos/user/e/ebornand/DFEI/FullMC/npy_array/magdown/training_dataset/target_0.npy", allow_pickle=True).item()

# Pour voir les clés disponibles dans chaque dictionnaire
print(graph.keys())
print(target.keys())


print("NODES:")
print("Shape:", graph["nodes"].shape)
print("Sample:", graph["nodes"][126])  # 5 premiers nœuds

print("\nEDGES:")
print("Shape:", graph["edges"].shape)
print("Sample:", graph["edges"][:5])  # 5 premières arêtes

print("\nSENDERS:", graph["senders"][:100])
print("RECEIVERS:", graph["receivers"][:100])

print("\nTARGET EDGE LABELS (from target['edges']):")
print("Shape:", target["edges"].shape)
print("Sample:", target["edges"][:500])