CrossSpread-ODE

Testing whether connectome-based spreading dynamics learned on Alzheimer's
transfer to Parkinson's.

Files
- `prep_oasis.py` — builds the model-ready CSV from the OASIS-3 FreeSurfer
  export and the UDSd1 clinical file
- `build_connectome_82region.py` — remaps the UCL tractography connectome to
  our 82-region order
- `pipeline.py` — mean-stage baseline, NDM, FKPP, and a Graph-Constrained
  Neural ODE

Run: `python pipeline.py --data-source {synthetic,real}`

Status
Real-data results are on 37 AD patients with 2+ longitudinal scans —
preliminary. Known open items: z-scores are computed against controls
without age/sex adjustment, and site harmonization is not yet applied.
