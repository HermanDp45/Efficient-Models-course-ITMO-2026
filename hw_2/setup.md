mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh


~/miniconda3/bin/conda init bash
source ~/.bashrc


conda create -n cce_bench python=3.10 -y
conda activate cce_bench

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); import triton; print('Triton version:', triton.__version__)"

cd test_cce
git clone https://github.com/apple/ml-cross-entropy.git
cd ml-cross-entropy

# Флаг -e ставит пакет в режиме editable, [all] дотягивает зависимости для тестов и transformers
pip install -e ".[all]"

pip install jupyterlab ipykernel
python -m ipykernel install --user --name cce_bench --display-name "Python (cce_bench)"

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-cache-dir
pip uninstall flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

import torch
print(torch._C._GLIBCXX_USE_CXX11_ABI)  # False или True