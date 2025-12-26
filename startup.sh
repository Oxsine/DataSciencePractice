#!/bin/bash
python -m venv .venv_ds
source venv_ds/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv_ds --display-name="Python (Data Science)"