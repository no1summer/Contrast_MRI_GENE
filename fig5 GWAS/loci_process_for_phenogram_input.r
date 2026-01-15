# ------------------------------------------------------------------
# 1. SETUP & LIBRARIES
# ------------------------------------------------------------------
if (!require("GenomicRanges")) install.packages("GenomicRanges")
library(GenomicRanges)

# Define Input Files
file_a <- "/data484_4/txia2/gwas_practice/fuma_result/mocov2_partial_fit_5std/GenomicRiskLoci.txt"
file_b <- "/data484_4/txia2/gwas_practice/fuma_result/mocov2_t2_5std/GenomicRiskLoci.txt"

# Output File
output_file <- "/data484_4/txia2/mocov2/loci_chromosome_R/phenogram_input.txt"

# ------------------------------------------------------------------
# 2. DOWNLOAD CYTOBAND DATA (UCSC hg19)
# ------------------------------------------------------------------
# We use hg19 because standard FUMA results are based on GRCh37/hg19.
print("Downloading Cytoband database from UCSC...")
cyto_url <- "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz"
temp_cyto <- tempfile()
download.file(cyto_url, temp_cyto, quiet = TRUE)

# Read Cytoband Data
# Columns: 1=chr, 2=start, 3=end, 4=band_name, 5=stain
cyto_df <- read.table(temp_cyto, sep="\t", header=FALSE, stringsAsFactors=FALSE)
colnames(cyto_df) <- c("chrom", "start", "end", "name", "gieStain")

# Convert Cytobands to GRanges object for fast mapping
cyto_gr <- GRanges(
  seqnames = cyto_df$chrom,
  ranges = IRanges(start = cyto_df$start, end = cyto_df$end),
  band = cyto_df$name
)

# ------------------------------------------------------------------
# 3. FUNCTION TO MAP LOCI TO BANDS
# ------------------------------------------------------------------
get_phenogram_df <- function(file_path, phenotype_label, color_group) {
  
  # Read FUMA Data
  df <- read.table(file_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
  
  # Ensure 'chr' column is just numbers (remove 'chr' prefix if present)
  df$chr <- gsub("chr", "", as.character(df$chr))
  
  # Create GRanges for the FUMA loci
  # We use the 'pos' column. If missing, use 'start'.
  pos_col <- if("pos" %in% colnames(df)) df$pos else df$start
  
  loci_gr <- GRanges(
    seqnames = paste0("chr", df$chr),
    ranges = IRanges(start = pos_col, end = pos_col)
  )
  
  # Find Overlaps: Which band does each locus fall into?
  hits <- findOverlaps(loci_gr, cyto_gr)
  
  # Extract the band name (e.g., "p13.1")
  # We initialize with NA, then fill in matches
  df$band_suffix <- NA
  df$band_suffix[queryHits(hits)] <- mcols(cyto_gr)$band[subjectHits(hits)]
  
  # Create the final Label: Chromosome + Band (e.g., "1" + "p13.1" = "1p13.1")
  df$final_annotation <- paste0(df$chr, df$band_suffix)
  
  # Construct the Data Frame required by PhenoGram
  out_df <- data.frame(
    CHR = df$chr,
    POS = pos_col,
    PHENOTYPE = phenotype_label,
    COLORGROUP = color_group,
    ANNOTATION = df$final_annotation # <--- This is now "1p13.1"
  )
  
  return(out_df)
}

# ------------------------------------------------------------------
# 4. PROCESS AND SAVE
# ------------------------------------------------------------------
print("Processing datasets...")

# Process Dataset A
pheno_a <- get_phenogram_df(file_a, "MoCoV2_T1", "T1")

# Process Dataset B
pheno_b <- get_phenogram_df(file_b, "MoCoV2_T2", "T2")

# Combine
final_output <- rbind(pheno_a, pheno_b)

# Save
write.table(final_output, output_file, sep = "\t", quote = FALSE, row.names = FALSE)

# Clean up
unlink(temp_cyto)

print(paste("Success! File saved to:", output_file))
print("Example lines:")
print(head(final_output))