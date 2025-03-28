#!/usr/bin/env bash

echo "Creating virtual environment"
python3.10 -m venv sinc-env
echo "Activating virtual environment"

source $PWD/sinc-env/bin/activate

$PWD/sinc-env/bin/pip install --upgrade pip setuptools

$PWD/sinc-env/bin/pip install "torch==2.0.1" "torchvision==0.15.2" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cu118

# torch 2.0.0 cu117

$PWD/sinc-env/bin/pip install "pytorch-lightning==2.0.1"
$PWD/sinc-env/bin/pip install "torchmetrics==0.11.4"
$PWD/sinc-env/bin/pip install -r requirements.txt
