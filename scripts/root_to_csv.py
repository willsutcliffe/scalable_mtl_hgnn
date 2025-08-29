import uproot
import pandas as pd

# --- Paramètres ---
input_root_file = "/eos/user/e/ebornand/DFEI/PYTHIA/BtoKSpimumu/P_and_R/Event_data_particles_bquark_Bu_KSpimumu_nu7_6_2000_4999_5000events_allbkgtracks_events_4900_to_4999.root"
output_excel_file = "/afs/cern.ch/user/e/ebornand/DFEI_HGNN/csv_outputs/Particules_Bu_KSpimumu.csv"

tree_name = "Particles"

# --- Lecture du fichier ROOT ---
with uproot.open(input_root_file) as file:
    tree = file[tree_name]
    df = tree.arrays(library="pd")

# --- Export vers CSV ---
df.to_csv(output_excel_file, index=False)

print(f"Fichier csv écrit avec succès : {output_excel_file}")
