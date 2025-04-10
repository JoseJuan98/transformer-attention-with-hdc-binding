echo -e "\n\n_____________________________ General Information about the server _____________________________ \n"
echo -e "NVIDIA GPUs available on the server:\n$(nvidia-smi --list-gpus) with $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) Mib VRAM\n"
echo -e "CPU cores available on the server: $(nproc)\n"
echo -e "Memory available on the server: $(free -h | grep Mem | awk '{print $2}')\n\n"

echo -e "\n\n_____________________________ Starting Setup _____________________________ \n"
apt-get update
apt-get upgrade -y
apt-get install -y neovim btop htop

echo -e "\n\n_____________________________ Miniconda3 Setup _____________________________ \n"
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

source ~/.bashrc

echo -e "Conda installed successfully. Please run the following command to activate conda:\n\n\tsource ~/.bashrc\n\n"
