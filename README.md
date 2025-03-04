# Sturgeon: Open-Source Deep Learning for Simultaneous Base Calling and Methylation Detection in Nanopore Sequencing  

This project introduces *Sturgeon*, a deep learning framework enabling accurate base calling and DNA methylation quantification using nanopore sequencing data. We address limitations in protein accessibility and model training by engineering commercial-grade Pif1 helicase and CsgG/HfaB proteins via random amino acid substitution and providing an open-source platform. Sturgeon combines convolutional neural networks (CNNs) and reverse LSTMs to process raw signals, achieving 88.76%-91.88% base calling accuracy and high methylation correlation with BS-seq (Pearson ≥0.96) across species. Designed for extensibility, Sturgeon serves as a foundation for advancing nanopore sequencing of genetic and epigenetic modifications.  

*Key innovations*: Novel protein designs, open-source framework, and simultaneous sequence/modification analysis.



## Building and Using Sturgeon

### Preparing the Python Environment

Create a virtual environment using Conda. The Python scripts require numpy (version 20.0 or higher) and pytorch (version 2.0 or higher) with CUDA above 12.0

```bash
conda create -n Sturgeon python=3.10
conda activate Sturgeon
pip install numpy torch==2.6.0+cu124
```

### Building the cpp_components

Sturgeon utilize a C++ code to accelerate CTC decoding process. It should be built before running basecalling.

```bash
git clone https://github.com/xiaochuanle/Sturgeon.git
cd Sturgeon/cpp_components
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPLIER="your nvcc path" -DCMAKE_PREFIX_PATH="your libtorch path"
```

You should change the cuda complier and libtorch path in `CMakeLists.txt` or add the correct path while running cmake.

## Running Sturgeon Basecalling and Methylation detection

Suturgeon combines basecalling and methylation detection together in one run. And our trained model with Qitan Sequencing data is in `./Saved_Models`

Here is an example.

```bash
python ./basecall.py 
./Saved_Models/cnn_reverselstm_ctc.pt
/h5_dir/
-o 
/output_file_path
--batch_size 1024 
--num_get_chunk 16 
--num_decode 8 
--num_stitch 8 
--output_moves 
--call_mods // optional
--ref_genome 
/your_ref_genome_path
--mod_module_path 
./Saved_Models/mod_bilstm_attn.pt
--mod_result_path 
/path_to_save_mod_res/mod_result.txt
```

