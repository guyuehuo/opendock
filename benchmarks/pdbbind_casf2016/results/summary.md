# PDBbind CASF-2016 docking-power success rates

Success rate = fraction of complexes with heavy-atom RMSD <= threshold for the top-ranked pose (thresholds: [1.0, 2.0, 2.5] A).

| tool | cfg | source | mode | n | top1<=1.0 | top1<=2.0 | top1<=2.5 | best-any<=2.0 | mean top1 RMSD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| idock | default | crystal | blind | 165 | 5.5% | 7.3% | 7.9% | 10.3% | 20.05 |
| idock | default | crystal | pocket | 165 | 46.7% | 59.4% | 63.0% | 81.8% | 2.53 |
| idock | default | rdkit | blind | 165 | 1.2% | 4.8% | 6.1% | 7.3% | 19.21 |
| idock | default | rdkit | pocket | 165 | 26.1% | 43.0% | 49.1% | 69.7% | 3.45 |
| opendock | ga-lbfgs | crystal | blind | 165 | 8.5% | 16.4% | 19.4% | 20.6% | 14.51 |
| opendock | ga-lbfgs | crystal | pocket | 165 | 53.9% | 67.3% | 75.2% | 73.3% | 1.65 |
| opendock | ga-lbfgs | rdkit | blind | 165 | 0.0% | 1.2% | 2.4% | 2.4% | 16.67 |
| opendock | ga-lbfgs | rdkit | pocket | 164 | 3.7% | 7.9% | 11.0% | 11.0% | 5.92 |
| opendock | ga-nomin | crystal | blind | 163 | 0.0% | 5.5% | 6.1% | 6.7% | 17.22 |
| opendock | ga-nomin | crystal | pocket | 165 | 22.4% | 42.4% | 46.1% | 50.3% | 3.33 |
| opendock | ga-nomin | rdkit | blind | 162 | 0.0% | 1.9% | 1.9% | 1.9% | 18.14 |
| opendock | ga-nomin | rdkit | pocket | 162 | 1.2% | 4.9% | 9.3% | 7.4% | 6.77 |
| opendock | mc-lbfgs | crystal | blind | 163 | 2.5% | 3.7% | 4.3% | 3.7% | 24.49 |
| opendock | mc-lbfgs | crystal | pocket | 165 | 21.2% | 24.8% | 29.7% | 26.1% | 176.94 |
| opendock | mc-lbfgs | rdkit | blind | 164 | 0.0% | 0.0% | 0.0% | 0.0% | 24.42 |
| opendock | mc-lbfgs | rdkit | pocket | 165 | 1.2% | 4.8% | 6.1% | 6.1% | 7.61 |
| opendock | mc-nomin | crystal | blind | 165 | 0.0% | 1.2% | 1.8% | 1.2% | 15.29 |
| opendock | mc-nomin | crystal | pocket | 165 | 0.0% | 7.9% | 15.8% | 9.1% | 4.86 |
| opendock | mc-nomin | rdkit | blind | 165 | 0.0% | 0.0% | 0.0% | 0.0% | 15.95 |
| opendock | mc-nomin | rdkit | pocket | 165 | 0.0% | 0.0% | 1.2% | 0.0% | 6.33 |
| opendock | pso-lbfgs | crystal | blind | 165 | 3.6% | 4.2% | 6.1% | 6.1% | 18.16 |
| opendock | pso-lbfgs | crystal | pocket | 165 | 40.0% | 54.5% | 60.6% | 70.3% | 2.70 |
| opendock | pso-lbfgs | rdkit | blind | 165 | 0.6% | 1.2% | 1.8% | 1.2% | 19.92 |
| opendock | pso-lbfgs | rdkit | pocket | 165 | 3.0% | 8.5% | 10.9% | 17.6% | 5.70 |
| opendock | pso-nomin | crystal | blind | 165 | 1.8% | 6.7% | 7.3% | 7.9% | 14.93 |
| opendock | pso-nomin | crystal | pocket | 165 | 25.5% | 46.1% | 53.9% | 60.0% | 3.18 |
| opendock | pso-nomin | rdkit | blind | 165 | 0.0% | 0.6% | 0.6% | 1.2% | 15.11 |
| opendock | pso-nomin | rdkit | pocket | 165 | 1.2% | 5.5% | 6.7% | 8.5% | 6.02 |

Notes (best-any shown at 2.0 A):
- OpenDock ``box_size`` is a half extent; idock/Vina ``size`` is full length (harness converts).
- idock performs its own stochastic global search: the crystal/rdkit axis differs only in the input geometry, not the search box.
- OpenDock blind runs use reduced sampling effort (see benchmark README / conditions.json).
