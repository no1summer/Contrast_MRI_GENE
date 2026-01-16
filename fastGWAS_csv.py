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
    # pcol is indexed from right to left, 0 means last col
    p = []
    f = []
    if mode == 'min':
        op = np.argmin
    elif mode == 'max':
        op = np.argmax
    else:
        raise Exception('not implemented')
    for i in tqdm(range(0, len(glob_list), batch_size)):
        with mp.Pool(batch_size) as q:
            result = q.starmap(extract_col, zip_longest(glob_list[i:i+batch_size], (), fillvalue=pcol))
        pnew, fnew = list(zip(*result))
        if i == 0:
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            f = np.array(fnew)[idx]
        else:
            pnew = list(pnew)
            pnew.append(p)
            pnew = np.vstack(pnew)
            idx = op(pnew, 0)
            mask = (idx != (pnew.shape[0]-1))
            f[mask] = np.array(fnew)[idx[mask]]
        p = pnew[idx, np.arange(pnew.shape[1])]
    return p, f

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
        p, f = create_minP(glob_list, args.minp_pcol, args.minp_mode, args.minp_batch_size)
        template_path = glob_list[0]
        template = pd.read_table(template_path)
        template["P"] = p
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(args.output_dir, args.minp_output_file)
        template[['CHR', 'SNP', 'POS', 'A1', 'A2', 'AF1', 'N', 'P']].to_csv(out_path, sep='\t', index=False)
        print(f"minP output saved to: {out_path}")
