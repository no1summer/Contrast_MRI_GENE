#!/usr/bin/env python3
"""
Merged script: Process MAGMA genes output and run GREP analysis.
1. Processes MAGMA file to extract significant genes
2. Runs GREP analysis with ATC and ICD tests
"""

import argparse
import os
import sys
import scipy.stats
import numpy as np
import pandas as pd

# GREP data file paths
BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.normpath(os.path.join(BASEDIR, 'data'))
ALL_GENES_FNAME = os.path.join(DATADIR, 'DrugBank_TTD_target_genelist.txt')
ATC_TARGETS_FNAME = os.path.join(DATADIR, 'DrugBank_TTD_targets_by_ATC_v2.txt')
ATC_ANNOT_FNAME = os.path.join(DATADIR, 'ATC_annotation.txt')
ICD_TARGETS_FNAME = os.path.join(DATADIR, 'DrugBank_TTD_targets_by_ICD10.txt')
ICD_ANNOT_FNAME = os.path.join(DATADIR, 'icd10_annotation_category.txt')

# Fixed output directory
OUTPUT_BASE_DIR = "/data484_4/txia2/gwas_practice/drug_GREP/GREP/output"


def process_magma_file(input_file, output_file=None):
    """
    Process MAGMA genes output file.
    
    Parameters:
    -----------
    input_file : str
        Path to magma.genes.out file
    output_file : str, optional
        Path to output file. If None, prints to stdout.
    
    Returns:
    --------
    tuple: (num_genes, p_threshold, num_passing, output_file)
    """
    # Read the MAGMA file
    df = pd.read_csv(input_file, sep='\t')
    
    # Calculate number of genes (total lines minus header)
    num_genes = len(df)
    
    # Calculate p-value threshold: 5e-8 / number of genes
    p_threshold = 5e-8 / num_genes
    
    print(f"Total genes: {num_genes}", file=sys.stderr)
    print(f"P-value threshold: {p_threshold:.2e}", file=sys.stderr)
    
    # Filter genes by p-value threshold
    filtered_df = df[df['P'] < p_threshold]
    
    print(f"Genes passing threshold: {len(filtered_df)}", file=sys.stderr)
    
    # Extract SYMBOL column
    symbols = filtered_df['SYMBOL'].tolist()
    
    # Output symbols
    if output_file:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}", file=sys.stderr)
        
        # Write symbols to file
        with open(output_file, 'w') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        print(f"Output written to: {output_file}", file=sys.stderr)
        print(f"Total genes written: {len(symbols)}", file=sys.stderr)
    else:
        # Print to stdout
        for symbol in symbols:
            print(symbol)
    
    return num_genes, p_threshold, len(filtered_df), output_file


# Fisher exact test function
def grep(target, key, annot, out_drug=False):
    global target_gene, all_genes, target_analysis_genes
    
    group = target[key].iloc[0]
    this_group_genes = target.TargetGene.unique()
    joined_genes = np.intersect1d(target_gene, this_group_genes)
    classified_genes = np.intersect1d(all_genes, this_group_genes)
    a = len(joined_genes)
    b = len(classified_genes) - a
    c = len(target_analysis_genes) - a
    d = len(all_genes) - len(target_analysis_genes) - b
    oddsratio, pvalue = scipy.stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    out_cols = ['#Group', 'GroupName', 'OddsRatio', 'FisherExactP']
    if out_drug:
        out_cols.append('TargetGene:DrugNames')
        gene_druglist = target[target.TargetGene.isin(joined_genes)].groupby('TargetGene').apply(
            lambda x: x.TargetGene.iloc[0] + ':' + ','.join(x.Drug.unique()))
        drugnames = ";".join(gene_druglist)
        ret = [group, annot.Annot[group], oddsratio, pvalue, drugnames]
    else:
        ret = [group, annot.Annot[group], oddsratio, pvalue]
    return pd.Series(ret, out_cols)


def run_grep_analysis(genelist_file, output_prefix, test_modes, output_drug_name=True):
    """
    Run GREP analysis on gene list.
    
    Parameters:
    -----------
    genelist_file : str
        Path to gene list file
    output_prefix : str
        Output filename prefix
    test_modes : list
        List of test modes to run (e.g., ['ATC', 'ICD'])
    output_drug_name : bool
        Whether to output drug names (default: True)
    """
    global target_gene, all_genes, target_analysis_genes
    
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"Running GREP analysis", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Gene list: {genelist_file}", file=sys.stderr)
    print(f"Output prefix: {output_prefix}", file=sys.stderr)
    print(f"Test modes: {', '.join(test_modes)}", file=sys.stderr)
    print(f"Output drug names: {output_drug_name}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Load gene list
    target_gene = pd.read_csv(genelist_file, header=None)[0].unique()
    all_genes = pd.read_table(ALL_GENES_FNAME, header=None)[0]
    target_analysis_genes = np.intersect1d(all_genes, target_gene)
    
    print(f"Target genes: {len(target_gene)}", file=sys.stderr)
    print(f"All genes in database: {len(all_genes)}", file=sys.stderr)
    print(f"Overlapping genes: {len(target_analysis_genes)}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Run ATC analysis
    if 'ATC' in test_modes:
        print("Running ATC analysis...", file=sys.stderr)
        ATC = pd.read_csv(ATC_TARGETS_FNAME, header=None, sep='\t',
                names=['Code', 'Drug', 'TargetGene', 'XXX']).dropna(subset=['TargetGene'])
        ATC['large'] = ATC.Code.str.slice(0, 1)
        ATC_ANNOT = pd.read_csv(ATC_ANNOT_FNAME, header=None, sep='\t', index_col=0, names=['Code', 'Annot'])

        # Analysis for large group
        print("  - Processing ATC large groups...", file=sys.stderr)
        ATC.groupby('large').apply(
            grep, annot=ATC_ANNOT, key='large',
            out_drug=output_drug_name).to_csv(
                output_prefix + ".ATC.large.txt", index=False, sep='\t')
        print(f"  - Saved: {output_prefix}.ATC.large.txt", file=sys.stderr)

        # Analysis for detailed group
        print("  - Processing ATC detailed groups...", file=sys.stderr)
        ATC.groupby('Code').apply(
            grep, annot=ATC_ANNOT, key='Code',
            out_drug=output_drug_name).to_csv(
                output_prefix + ".ATC.detail.txt", index=False, sep='\t')
        print(f"  - Saved: {output_prefix}.ATC.detail.txt", file=sys.stderr)
        print("", file=sys.stderr)

    # Run ICD analysis
    if 'ICD' in test_modes:
        print("Running ICD analysis...", file=sys.stderr)
        annot_icd = {}
        subcat_icd = {}
        with open(ICD_ANNOT_FNAME, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                annot_icd[line[0]] = line[1]
                [subcat_icd.update({v: line[0]}) for v in line[2:]]
        ICD_ANNOT = pd.DataFrame.from_dict(
            annot_icd, orient='index').rename(columns={0: 'Annot'})

        ICD = pd.read_csv(ICD_TARGETS_FNAME, header=None, sep='\t', names=['Code', 'Drug', 'TargetGene'])
        ICD['large'] = ICD.Code.apply(
            lambda x: subcat_icd.setdefault(x, np.nan))
        ICD = ICD.dropna()

        print("  - Processing ICD groups...", file=sys.stderr)
        ICD.groupby('large').apply(
            grep, annot=ICD_ANNOT, key='large',
            out_drug=output_drug_name).to_csv(
                output_prefix + ".ICD.txt", index=False, sep='\t')
        print(f"  - Saved: {output_prefix}.ICD.txt", file=sys.stderr)
        print("", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process MAGMA genes output and run GREP analysis')
    parser.add_argument('--magma_file', type=str,
        default='/data484_4/txia2/gwas_practice/fuma_result/mocov2_partial_fit_5std/magma.genes.out',
        help='Path to MAGMA genes output file (default: mocov2_partial_fit_5std)')
    parser.add_argument('--test', '-t', type=str, nargs='+',
        choices=['ATC', 'ICD'], default=['ATC', 'ICD'],
        help='Test modes to run: ATC, ICD, or both (default: both)')
    parser.add_argument('--output-drug-name', '-d', 
        action='store_true',
        help='Output drug names (always enabled by default)')
    
    args = parser.parse_args()
    
    # Always enable output-drug-name
    args.output_drug_name = True
    
    # Step 1: Process MAGMA file
    print(f"{'='*80}", file=sys.stderr)
    print(f"Step 1: Processing MAGMA file", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Input file: {args.magma_file}", file=sys.stderr)
    
    # Extract directory name from input file path to use as prefix
    input_dir = os.path.basename(os.path.dirname(args.magma_file))
    
    # Generate output filename: {input_dir}.genes
    genelist_file = os.path.join(OUTPUT_BASE_DIR, f"{input_dir}.genes")
    output_prefix = os.path.join(OUTPUT_BASE_DIR, input_dir)
    
    print(f"Output gene list: {genelist_file}", file=sys.stderr)
    print(f"Output prefix: {output_prefix}", file=sys.stderr)
    print("", file=sys.stderr)
    
    num_genes, p_threshold, num_passing, _ = process_magma_file(
        args.magma_file, genelist_file)
    
    # Step 2: Run GREP analysis
    run_grep_analysis(
        genelist_file=genelist_file,
        output_prefix=output_prefix,
        test_modes=args.test,
        output_drug_name=args.output_drug_name
    )
    
    print(f"{'='*80}", file=sys.stderr)
    print("All analyses completed!", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
