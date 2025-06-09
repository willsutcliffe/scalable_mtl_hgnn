# Scalable Multi-Task Learning for Particle Collision Event Reconstruction with HGNNs Dataset

This dataset was used in the paper _“Scalable Multi-Task Learning for Particle Collision Event Reconstruction with Heterogeneous Graph Neural Networks”_. This paper presents a scalable heterogeneous graph network with integrated pruning layers, which jointly determines:

- If tracks originate from decays of beauty hadrons.
- The association of each track to a proton-proton collision point known as a primary vertex (PV).

---

## Generated Events

The events in this dataset are based on simulation generated with **PYTHIA** and **EvtGen**, replicating the particle-collision conditions expected for **LHC Run 3**, as shown in the table:

| LHCb Period            | Num. vis. pp Collisions | Num. Tracks | Num. b Hadrons | Num. c Hadrons |
|------------------------|--------------------------|-------------|----------------|----------------|
| Runs 3–4 (Upgrade I)   | ~5                       | ~150        | <1             | ~1             |

An approximate emulation of LHCb detection and reconstruction effects is applied, as described in the paper (Appendix: "Simulation"). Each event includes **at least one b-hadron**, which may decay freely via standard modes present in PYTHIA8. Notably:

- ~40% of events contain more than one b-hadron decay.
- Max decay multiplicity: 5 b-hadrons.
- Only **charged stable particles** within LHCb geometrical acceptance and Vertex Locator region are included.

---

## Datasets

The datasets are divided into three categories:

### Inclusive Training and Validation

- `inclusive_training_validation_dataset.tar.gz`:  
  - **Training**: 40,000 events  
  - **Validation/Test**: 10,000 events

### Inclusive Test

- `inclusive_test_dataset.tar.gz`:  
  - **Evaluation**: 10,000 events

### Exclusive Test and Training

We provide 5,000-event samples with **exclusive decays**. For some modes, the data is split into:

- **Training**: 1,000 events  
- **Test**: 4,000 events

#### Exclusive Decays:

- `Bd_DD_dataset.tar.gz`
- `Bd_Kstmumu_dataset.tar.gz`
- `Bd_Kpi_dataset.tar.gz`
- `Bu_Kmumu_dataset.tar.gz`
- `Bu_Kpipimumu_dataset.tar.gz`
- `Bu_KKpi_dataset.tar.gz`
- `Lb_Lcpi_dataset.tar.gz`
- `Lb_pK_dataset.tar.gz`
- `Lb_pKmumu_dataset.tar.gz`
- `Bs_Dspi_dataset.tar.gz`
- `Bs_Jpsiphi_dataset.tar.gz`

---

## Data Format

A Cartesian right-handed coordinate system is used:

- **z-axis**: Along the beamline  
- **x-axis**: Horizontal  
- **y-axis**: Vertical

### Event Storage

Events are stored in graph format:

- Input files: `input_<i>.npy`
- Target files: `target_<i>.npy`

Each input file contains a dictionary with the following:

#### Node Features

Key: `'nodes'`, shape `(n_nodes, 13)`:

1. `Ox, Oy, Oz`: Origin coordinates  
2. `px, py, pz`: Momentum vector  
3. `PVᴵᴾx, PVᴵᴾy, PVᴵᴾz`: PV from minimum impact parameter (for homogeneous GNNs)  
4. `q`: Charge (±1)  
5. `PVx, PVy, PVz`: True associated reconstructed PV (for heterogeneous GNNs and edge target generation)

#### Edge Features

Key: `'edges'`, shape `(n_edges, 4)`:

1. Opening angle (θ)  
2. Transverse momentum distance (d⊥⃗P)  
3. Distance along beamline (Δz)  
4. FromSamePV_MinIP: Boolean, same PV via MinIP

#### Track Edge Relations

- `'senders'` and `'receivers'`: Arrays of indices for directed edges.

#### Global Features

- Number of unique reconstructed PVs per graph.

---

## Targets

The target files contain **Lowest Common Ancestor Generation (LCAG)** edge labels:

- Key: `'edges'`, shape `(n_edges, 4)`
- Format: One-hot encoded for 4 LCAG classes (0, 1, 2, 3)

---

## Additional Truth Information for Evaluation

For performance metrics in the paper, additional truth labels are provided:

| Key               | Description |
|------------------|-------------|
| `part_ids`        | Particle IDs after loose preselection |
| `ids`             | Mother particle IDs (beauty decay chains, post-preselection) |
| `init_part_ids`   | Particle IDs before preselection |
| `init_ids`        | Mother IDs before preselection |
| `init_y`          | LCAG targets before preselection |
| `truth_part_ids`  | IDs of beauty-chain particles only |
| `truth_ids`       | Mother IDs of beauty-chain particles only |
| `truth_senders`   | Sender indices (beauty particles only) |
| `truth_receivers` | Receiver indices (beauty particles only) |
| `truth_y`         | LCAG target values (beauty particles only) |
| `lca_chain`       | Truth full-chain LCA values (used to determine max depth) |

---

## Loading Data

Example:

```python
import numpy as np

# Load graph features and LCAG edge targets for event 0
graph_input_features = np.load("input_0.npy", allow_pickle=True).item()
graph_target = np.load("target_0.npy", allow_pickle=True).item()
```

---

## GitHub Support

We provide functionality to load the datasets in the GitHub repository.
