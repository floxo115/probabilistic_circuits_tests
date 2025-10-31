echo "running hctl evaluation on artificial data"
python train_and_evaluate_hclt_models.py

echo "running own model evaluation on artificial data"
python train_and_evaluate_own_models.py

echo "running own model on hepatitis data"
python train_and_evaluate_hepatitis_ds_multiple_cats.py > results_for_hepatitis.txt
