# graphconformal

## SpecBridge
(Model repo)[https://github.com/HassounLab/SpecBridge/tree/main]
### Instalation Guide
 ```bash
cd Models/SpecBridge/
uv venv --python 3.11
source .venv/bin/activate

#Chose the best CUDA to ensure compatibility 11.8 or 
uv pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt

cd DreaMS
uv pip install -e .
cd ..
uv pip install -e .

# Check Python version
python --version  # Should be 3.11.

# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check SpecBridge
python -c "import specbridge; print('SpecBridge installed successfully')"

# Check DreaMS
python -c "import dreams; print('DreaMS installed successfully')"
```
