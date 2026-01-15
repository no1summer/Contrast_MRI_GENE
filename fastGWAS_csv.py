import os
import multiprocessing as mp
from tqdm import tqdm
import argparse
from functools import partial
from glob import glob
from subprocess import check_output, STDOUT
from itertools import zip_longest
import numpy as np
import pandas as pd
from pathlib import Path
import time
import psutil

def run_fastgwa(pheno, out_file_dir, pheno_dir, gcta_threads=32):
    os.makedirs(out_file_dir, exist_ok=True)
    out_file = f"{out_file_dir}/{pheno}.fastGWA"
    if os.path.exists(out_file):
        return 0  # skip if already done
    cmd = f"""
    /data4012/zxie3/gcta/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1 \
      --bgen /data484_4/txia2/gwas_practice/UKB_bgen/all_filtered.bgen \
      --maf 0.01 \
      --sample /data484_4/txia2/gwas_practice/UKB_bgen/MRI_samples_chr1.sample \
      --grm-sparse /data484_4/txia2/gwas_practice/grm/gcta/ukb_grm_discovery \
      --fastGWA-mlm \
      --pheno {pheno_dir}/{pheno} \
      --covar /data484_4/txia2/gwas_practice/T1_ccovar_discovery \
      --qcovar /data484_4/txia2/gwas_practice/T1_qcovar_discovery \
      --thread-num {gcta_threads} \
      --seed 0 \
      --out {out_file_dir}/{pheno} 
    """
    return os.system(cmd)


def extract_col(x, offset=0):
    out = check_output(f"awk '{{print $(NF-{offset})}}' {x}", universal_newlines=True, shell=True, stderr=STDOUT)
    return np.array(list(map(float, out.strip('\n').split('\n')[1:]))), x


def create_minP(glob_list, pcol=0, mode='min', batch_size=128):
    """
    Create minP by aligning SNPs across files.
    Handles files with different numbers of variants by aligning on SNP ID.
    """
    print(f"Processing {len(glob_list)} files for minP computation...")
    
    # First pass: collect all SNP IDs to create a master SNP list
    print("Step 1: Collecting all SNP IDs...")
    all_snps = set()
    file_snp_sets = {}
    
    for file_path in tqdm(glob_list, desc="Reading SNP lists"):
        try:
            df = pd.read_table(file_path, sep='\t', usecols=['SNP'])
            snps = set(df['SNP'].values)
            all_snps.update(snps)
            file_snp_sets[file_path] = snps
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue
    
    # Create sorted master SNP list
    master_snps = sorted(list(all_snps))
    print(f"Total unique SNPs across all files: {len(master_snps)}")
    
    # Second pass: extract P-values and align to master SNP list
    print("Step 2: Extracting and aligning P-values...")
    p_matrix = []
    file_names = []
    
    for file_path in tqdm(glob_list, desc="Extracting P-values"):
        try:
            df = pd.read_table(file_path, sep='\t')
            
            # Get P column index (pcol from right, 0 = last column)
            p_col_idx = len(df.columns) - 1 - pcol
            p_col_name = df.columns[p_col_idx]
            
            # Create a dictionary mapping SNP to P-value
            snp_to_p = dict(zip(df['SNP'].values, df[p_col_name].values))
            
            # Align to master SNP list, filling missing SNPs with NaN
            aligned_p = np.array([snp_to_p.get(snp, np.nan) for snp in master_snps])
            
            p_matrix.append(aligned_p)
            file_names.append(file_path)
            
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    if len(p_matrix) == 0:
        raise ValueError("No files could be processed")
    
    # Convert to numpy array
    p_matrix = np.array(p_matrix)  # Shape: (n_files, n_snps)
    print(f"P-value matrix shape: {p_matrix.shape}")
    
    # Compute min/max P across files
    if mode == 'min':
        # For minP, we want the minimum P-value across files for each SNP
        # But we should ignore NaN values
        with np.errstate(invalid='ignore'):
            p_result = np.nanmin(p_matrix, axis=0)
            file_indices = np.nanargmin(p_matrix, axis=0)
    elif mode == 'max':
        with np.errstate(invalid='ignore'):
            p_result = np.nanmax(p_matrix, axis=0)
            file_indices = np.nanargmax(p_matrix, axis=0)
    else:
        raise Exception('Mode must be "min" or "max"')
    
    # Get file names for each SNP
    f_result = np.array([file_names[i] for i in file_indices])
    
    # Handle SNPs that are NaN in all files
    all_nan_mask = np.isnan(p_result)
    if np.any(all_nan_mask):
        print(f"Warning: {np.sum(all_nan_mask)} SNPs are missing in all files")
        # Set these to NaN and use first file name as placeholder
        f_result[all_nan_mask] = file_names[0]
    
    return p_result, f_result, master_snps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fastGWA on multiple phenotypes, with optional minP post-processing.')
    parser.add_argument('--pheno_dir', type=str, required=True, help='Directory of the phenotype files.')
    parser.add_argument('--output_dir', type=str, default='/data484_4/txia2/gwas_practice/results/128_T1_vit/', help='Directory to save the output files.')
    # minP options
    parser.add_argument('--compute_minp', action='store_true', help='After running, compute min/max P across generated *.fastGWA files.')
    parser.add_argument('--minp_pattern', type=str, default='*.fastGWA', help='Glob pattern for input files when computing minP (default: *.fastGWA).')
    parser.add_argument('--minp_pcol', type=int, default=1, help='P column index from the right (0 = last col; default 1).')
    parser.add_argument('--minp_mode', choices=['min', 'max'], default='min', help='Operation across files (default: min).')
    parser.add_argument('--minp_output_file', type=str, default='minP.tsv', help='Output filename for minP (default: minP.tsv).')
    # Performance tuning arguments
    parser.add_argument('--parallel_jobs', type=int, default=8, help='Number of parallel jobs to run (default: 8).')
    parser.add_argument('--gcta_threads', type=int, default=32, help='Number of threads per GCTA job (default: 32).')
    parser.add_argument('--minp_batch_size', type=int, default=128, help='Batch size for minP processing (default: 128).')
    args = parser.parse_args()

    # Print system information and optimization settings
    print(f"System CPU cores: {mp.cpu_count()}")
    print(f"Parallel jobs: {args.parallel_jobs}")
    print(f"GCTA threads per job: {args.gcta_threads}")
    print(f"Total theoretical threads: {args.parallel_jobs * args.gcta_threads}")
    print(f"MinP batch size: {args.minp_batch_size}")
    print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")
    print("-" * 50)

    qt_columns= ['Feature_'+str(i)+'.csv' for i in range(0,384)]  # your phenotype files
    # Number of jobs to run at once (optimized for 256-core AMD EPYC 7763)
    # Using configurable parallel jobs with configurable threads each
    N_PARALLEL = args.parallel_jobs
    
    # Create a partial function with the output_dir fixed
    run_func = partial(run_fastgwa, out_file_dir=args.output_dir, pheno_dir=args.pheno_dir, gcta_threads=args.gcta_threads)

    # Start timing
    start_time = time.time()
    
    # Use maxtasksperchild=10 for better performance on high-core systems
    with mp.Pool(N_PARALLEL, maxtasksperchild=10) as pool:
        list(tqdm(pool.imap(run_func, qt_columns), total=len(qt_columns)))
    
    end_time = time.time()
    print(f"FastGWA processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per phenotype: {(end_time - start_time) / len(qt_columns):.2f} seconds")

    if args.compute_minp:
        # Gather all fastGWA outputs and compute min/max P
        glob_list = sorted(glob(os.path.join(args.output_dir, args.minp_pattern)))
        if len(glob_list) == 0:
            raise FileNotFoundError(f"No files matched pattern {args.minp_pattern} in {args.output_dir}")
        
        p, f, master_snps = create_minP(glob_list, args.minp_pcol, args.minp_mode, args.minp_batch_size)
        
        # Create output DataFrame aligned to master SNP list
        template_path = glob_list[0]
        template = pd.read_table(template_path)
        template_dict = template.set_index('SNP').to_dict('index')
        
        # Build output DataFrame
        output_data = []
        for snp in master_snps:
            if snp in template_dict:
                row = template_dict[snp].copy()
                row['SNP'] = snp
            else:
                # If SNP not in template, try to find it in another file
                row = None
                for file_path in glob_list[1:]:
                    try:
                        df = pd.read_table(file_path, sep='\t')
                        df_dict = df.set_index('SNP').to_dict('index')
                        if snp in df_dict:
                            row = df_dict[snp].copy()
                            row['SNP'] = snp
                            break
                    except:
                        continue
                if row is None:
                    # Create a row with NaN for other columns
                    row = {
                        'CHR': np.nan,
                        'SNP': snp,
                        'POS': np.nan,
                        'A1': np.nan,
                        'A2': np.nan,
                        'AF1': np.nan,
                        'N': np.nan
                    }
            output_data.append(row)
        
        output_df = pd.DataFrame(output_data)
        output_df['P'] = p
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(args.output_dir, args.minp_output_file)
        output_df[['CHR', 'SNP', 'POS', 'A1', 'A2', 'AF1', 'N', 'P']].to_csv(out_path, sep='\t', index=False)
        print(f"minP output saved to: {out_path}")
