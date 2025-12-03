Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
python -m venv venv_ds
venv_ds\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv_ds --display-name="Python (Data Science)"