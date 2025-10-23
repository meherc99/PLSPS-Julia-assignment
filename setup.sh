
# Load modules on DAS-5
module load slurm/17.02.2
module load gcc/6.4.0
module load julia/1.10.3 2>/dev/null

if ! command -v jq &> /dev/null; then
    echo "jq not found, installing locally..."
    mkdir -p $HOME/bin
    wget -O $HOME/bin/jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
    chmod +x $HOME/bin/jq
    echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
else
    echo "jq is already installed"
fi

# Setup Julia environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -O3 --check-bounds=no -e 'using Pkg; Pkg.precompile()'