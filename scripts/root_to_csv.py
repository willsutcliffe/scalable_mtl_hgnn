import uproot
import pandas as pd

# --- Paramètres ---
input_root_file = "/eos/user/e/ebornand/DFEI/FullMC//P_and_R/magdown/Event_data_particles_bquark_Run3_inclusiveb_magdown_beautiful_ancestors_90754events_allbkgtracks_events_0_to_499.root"
output_excel_file = "/afs/cern.ch/user/e/ebornand/DFEI_HGNN/csv_outputs/Particules.csv"

tree_name = "Particles"

# --- Lecture du fichier ROOT ---
with uproot.open(input_root_file) as file:
    tree = file[tree_name]
    df = tree.arrays(library="pd")

# --- Export vers CSV ---
df.to_csv(output_excel_file, index=False)

print(f"Fichier csv écrit avec succès : {output_excel_file}")
