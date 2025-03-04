import gzip
import argparse
from itertools import islice

def calculate_fastq_base_count(fastq_file):
    total_bases = 0
    open_func = gzip.open if fastq_file.endswith('.gz') else open
    with open_func(fastq_file, 'rt') as f:
        while True:
            lines = list(islice(f, 4))
            if not lines:
                break
            if len(lines) < 4:
                print("Warning: Incomplete FASTQ record encountered.")
                break
            seq_line = lines[1].strip()
            total_bases += len(seq_line)
    return total_bases

def calculate_genome_length(fasta_file):
    total_length = 0
    open_func = gzip.open if fasta_file.endswith('.gz') else open
    with open_func(fasta_file, 'rt') as f:
        for line in f:
            if not line.startswith('>'):
                total_length += len(line.strip())
    return total_length

def calculate_sequencing_depth(fastq_file, fasta_file):
    total_bases = calculate_fastq_base_count(fastq_file)
    genome_length = calculate_genome_length(fasta_file)
    if genome_length == 0:
        print("Error: Genome length is zero.")
        return None
    return total_bases / genome_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate sequencing depth from FASTQ and FASTA files.'
    )
    parser.add_argument(
        'fastq_file',
        help='Input FASTQ file (can be gzipped).'
    )
    parser.add_argument(
        'fasta_file',
        help='Input genome FASTA file (can be gzipped).'
    )
    args = parser.parse_args()

    depth = calculate_sequencing_depth(args.fastq_file, args.fasta_file)
    if depth is not None:
        print(f"Sequencing depth: {depth:.2f}")
