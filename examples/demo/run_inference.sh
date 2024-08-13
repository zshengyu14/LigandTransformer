
# -----------Variables to change--------------------------------
OUTPUT_DIR="./"
output_csv="affnity.csv"
protein_csv="Abeta.csv"
mol_library_pickle="mols.pkl"

python ../../src/inference.py --output_dir $OUTPUT_DIR --output_csv $output_csv --protein_csv $protein_csv --mol_library_pickle $mol_library_pickle --ckpt_path modelA.ckpt