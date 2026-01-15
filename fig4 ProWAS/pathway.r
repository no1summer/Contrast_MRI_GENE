# -----------------------------
# 1️⃣ Non-interactive package installation
# -----------------------------

# Set CRAN and Bioconductor repos
options(repos = c(CRAN="https://cloud.r-project.org"))
options(BioC_mirror="https://bioconductor.org")

# Install CRAN packages if missing
install_if_missing <- function(pkg){
  if (!requireNamespace(pkg, quietly=TRUE)) {
    install.packages(pkg, repos="https://cloud.r-project.org", dependencies=TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only=TRUE))
}

cran_pkgs <- c("pheatmap","ggplot2","dplyr","httr","igraph","GOplot")
for (p in cran_pkgs) install_if_missing(p)

# Try to install GO.db for GO term names
if (!requireNamespace("GO.db", quietly=TRUE)) {
  tryCatch({
    BiocManager::install("GO.db", update=FALSE, ask=FALSE, dependencies=FALSE)
  }, error = function(e) {
    cat("Warning: GO.db not available, GO term names may be missing\n")
  })
}

# Install Bioconductor packages if missing
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager", repos="https://cloud.r-project.org")

install_if_missing_bioc <- function(pkg){
  if (!requireNamespace(pkg, quietly=TRUE)) {
    tryCatch({
      BiocManager::install(pkg, update=FALSE, ask=FALSE, dependencies=FALSE)
    }, error = function(e) {
      cat(sprintf("Warning: Failed to install %s: %s\n", pkg, e$message))
    })
  }
  if (requireNamespace(pkg, quietly=TRUE)) {
  suppressPackageStartupMessages(library(pkg, character.only=TRUE))
    return(TRUE)
  } else {
    cat(sprintf("Warning: Package %s not available\n", pkg))
    return(FALSE)
  }
}

bioc_pkgs <- c("clusterProfiler","org.Hs.eg.db","ReactomePA","DOSE","dorothea","viper","enrichplot")
bioc_status <- sapply(bioc_pkgs, install_if_missing_bioc)

# -----------------------------
# 2️⃣ Now continue with your normal script
# -----------------------------
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings=FALSE)

# Rest of your enrichment / heatmap / DoRothEA code here


if (!requireNamespace("org.Hs.eg.db", quietly=TRUE)) {
  stop("org.Hs.eg.db is required but not installed")
}

library(org.Hs.eg.db)
if (requireNamespace("clusterProfiler", quietly=TRUE)) {
  library(clusterProfiler)
  use_clusterProfiler <- TRUE
} else {
  cat("Warning: clusterProfiler not available, using basic GO term lookup\n")
  use_clusterProfiler <- FALSE
}
if (requireNamespace("enrichplot", quietly=TRUE)) library(enrichplot)
if (requireNamespace("DOSE", quietly=TRUE)) library(DOSE)
if (requireNamespace("ReactomePA", quietly=TRUE)) library(ReactomePA)
library(pheatmap)

dir.create(output_dir, showWarnings = FALSE)

# Read proteins
proteins <- read.csv(input_file, stringsAsFactors = FALSE)
gene_list <- proteins$common_protein  # change column name if needed

# Convert to Entrez IDs
entrez_ids <- mapIds(org.Hs.eg.db, keys=gene_list, column="ENTREZID", keytype="SYMBOL", multiVals="first")
entrez_ids <- na.omit(entrez_ids)

if (use_clusterProfiler) {
# GO Biological Process
go_bp <- enrichGO(entrez_ids, OrgDb=org.Hs.eg.db, ont="BP", pAdjustMethod="BH", readable=TRUE)
  write.csv(as.data.frame(go_bp), file=paste0(output_dir,"/GO_BP_enrichment.csv"))
  
  # Create GO bar plot with clusterProfiler results
  if (nrow(as.data.frame(go_bp)) > 0) {
    go_df <- as.data.frame(go_bp)
    go_df <- go_df[order(go_df$p.adjust), ]
    top_n <- min(10, nrow(go_df))
    go_plot_df <- go_df[1:top_n, ]
    go_plot_df$Description <- factor(go_plot_df$Description, levels=rev(go_plot_df$Description))
    
    # Order ascendingly (smallest at top, largest at bottom)
    go_plot_df <- go_plot_df[order(go_plot_df$Count, decreasing=FALSE), ]
    go_plot_df$Description <- factor(go_plot_df$Description, levels=go_plot_df$Description)
    
    max_count <- max(go_plot_df$Count)
    x_breaks <- seq(0, max_count, by=1)  # Integer breaks
    
    p <- ggplot2::ggplot(go_plot_df, ggplot2::aes(x=Count, y=Description)) +
      ggplot2::geom_bar(stat="identity", fill="steelblue", alpha=0.7) +
      ggplot2::scale_x_continuous(breaks=x_breaks, limits=c(0, max_count + 0.5)) +
      ggplot2::labs(title="Top GO biological process term",
                    x="Number of Proteins", y="GO Biological Process Term") +
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold", size=20),
                     axis.text.x=ggplot2::element_text(size=20),
                     axis.text.y=ggplot2::element_text(size=20),
                     axis.title.x=ggplot2::element_text(size=20),
                     axis.title.y=ggplot2::element_text(size=20))
    
    ggplot2::ggsave(paste0(output_dir,"/GO_BP_barplot.pdf"), p, width=14, height=8)
    ggplot2::ggsave(paste0(output_dir,"/GO_BP_barplot.png"), p, width=14, height=8, dpi=300)
    cat(sprintf("GO bar plot saved (top %d terms)\n", top_n))
  }
  
  if (requireNamespace("enrichplot", quietly=TRUE)) {
    pdf(paste0(output_dir,"/GO_BP_dotplot.pdf"), width=8, height=6); dotplot(go_bp, showCategory=10); dev.off()
    pdf(paste0(output_dir,"/GO_BP_cnetplot.pdf"), width=10, height=8); cnetplot(go_bp, showCategory=10); dev.off()
  }
  
  # Create GO bubble plot using GOplot package
  if (requireNamespace("GOplot", quietly=TRUE)) {
    library(GOplot)
    go_df <- as.data.frame(go_bp)
    if (nrow(go_df) > 0) {
      # Prepare data for GOplot
      # GOplot needs: ID, term, count, genes, adj_pval
      # Check column names
      cat("GO dataframe columns:", paste(colnames(go_df), collapse=", "), "\n")
      
      go_plot_data <- data.frame(
        ID = go_df$ID,
        term = go_df$Description,
        count = go_df$Count,
        genes = if("geneID" %in% colnames(go_df)) go_df$geneID else go_df$gene,
        adj_pval = go_df$p.adjust,
        stringsAsFactors = FALSE
      )
      
      # Select top terms
      top_n <- min(10, nrow(go_plot_data))
      go_plot_data <- go_plot_data[order(go_plot_data$adj_pval), ][1:top_n, ]
      
      cat("GO plot data rows:", nrow(go_plot_data), "\n")
      cat("GO plot data columns:", paste(colnames(go_plot_data), collapse=", "), "\n")
      
      # Create bubble plot with error handling
      tryCatch({
        pdf(paste0(output_dir, "/GO_BP_bubbleplot.pdf"), width=14, height=10)
        GOBubble(go_plot_data, labels=15, table.legend=FALSE, table.col=c("white", "lightblue"))
        dev.off()
        
        png(paste0(output_dir, "/GO_BP_bubbleplot.png"), width=14*300, height=10*300, res=300)
        GOBubble(go_plot_data, labels=15, table.legend=FALSE, table.col=c("white", "lightblue"))
        dev.off()
        
        cat("GO bubble plot saved using GOplot package\n")
      }, error = function(e) {
        cat("Error creating GO bubble plot:", e$message, "\n")
        # Try alternative: create simple bubble plot with ggplot2
        library(ggplot2)
        p <- ggplot2::ggplot(go_plot_data, ggplot2::aes(x=count, y=-log10(adj_pval), size=count, label=term)) +
          ggplot2::geom_point(alpha=0.6, color="steelblue") +
          ggplot2::scale_size_continuous(range=c(3, 12)) +
          ggplot2::labs(title="GO Biological Process Enrichment",
                        x="Number of Genes", y="-log10(Adjusted P-value)") +
          ggplot2::theme_minimal() +
          ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold", size=20),
                         axis.text=ggplot2::element_text(size=16),
                         axis.title=ggplot2::element_text(size=18))
        
        ggplot2::ggsave(paste0(output_dir, "/GO_BP_bubbleplot.pdf"), p, width=14, height=10)
        ggplot2::ggsave(paste0(output_dir, "/GO_BP_bubbleplot.png"), p, width=14, height=10, dpi=300)
        cat("GO bubble plot saved using ggplot2 fallback\n")
      })
    }
  } else {
    cat("GOplot package not available, skipping bubble plot\n")
  }

# KEGG
kegg <- enrichKEGG(entrez_ids, organism='hsa', pvalueCutoff=0.05)
  write.csv(as.data.frame(kegg), file=paste0(output_dir,"/KEGG_enrichment.csv"))
  
  # Create KEGG bar plot with clusterProfiler results
  if (nrow(as.data.frame(kegg)) > 0) {
    kegg_df <- as.data.frame(kegg)
    kegg_df <- kegg_df[order(kegg_df$p.adjust), ]
    top_n <- min(15, nrow(kegg_df))
    kegg_plot_df <- kegg_df[1:top_n, ]
    kegg_plot_df$Description <- factor(kegg_plot_df$Description, levels=rev(kegg_plot_df$Description))
    
    # Truncate long pathway names
    kegg_plot_df$Description_short <- sapply(kegg_plot_df$Description, function(x) {
      x_char <- as.character(x)
      if (nchar(x_char) > 60) {
        paste0(substr(x_char, 1, 57), "...")
      } else {
        x_char
      }
    })
    kegg_plot_df$Description_short <- factor(kegg_plot_df$Description_short, levels=rev(kegg_plot_df$Description_short))
    
    max_count_kegg <- max(kegg_plot_df$Count)
    x_breaks_kegg <- seq(0, max_count_kegg, by=1)
    
    p <- ggplot2::ggplot(kegg_plot_df, ggplot2::aes(x=Count, y=Description_short)) +
      ggplot2::geom_bar(stat="identity", fill="darkgreen", alpha=0.7) +
      ggplot2::scale_x_continuous(breaks=x_breaks_kegg, limits=c(0, max_count_kegg + 0.5)) +
      ggplot2::labs(title="Top KEGG Pathways",
                    x="Number of Proteins", y="KEGG Pathway") +
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold", size=20),
                     axis.text.x=ggplot2::element_text(size=20),
                     axis.text.y=ggplot2::element_text(size=20),
                     axis.title.x=ggplot2::element_text(size=20),
                     axis.title.y=ggplot2::element_text(size=20))
    
    ggplot2::ggsave(paste0(output_dir,"/KEGG_barplot.pdf"), p, width=14, height=8)
    ggplot2::ggsave(paste0(output_dir,"/KEGG_barplot.png"), p, width=14, height=8, dpi=300)
    cat(sprintf("KEGG bar plot saved (top %d pathways)\n", top_n))
  }

# Reactome
  if (requireNamespace("ReactomePA", quietly=TRUE)) {
reactome <- enrichPathway(entrez_ids, organism="human", pvalueCutoff=0.05, readable=TRUE)
    write.csv(as.data.frame(reactome), file=paste0(output_dir,"/Reactome_enrichment.csv"))
  }
} else {
  # Basic GO term lookup without enrichment statistics
  cat("Performing basic GO term lookup (enrichment statistics require clusterProfiler)\n")
  go_terms <- select(org.Hs.eg.db, keys=as.character(entrez_ids), columns=c("GO", "ONTOLOGY"), keytype="ENTREZID")
  go_bp_terms <- go_terms[go_terms$ONTOLOGY == "BP", ]
  write.csv(go_bp_terms, file=paste0(output_dir,"/GO_BP_terms.csv"), row.names=FALSE)
  cat("Basic GO terms saved to GO_BP_terms.csv\n")
  
  # Get GO term names and create plots
  if (nrow(go_bp_terms) > 0) {
    # Count genes per GO term
    go_counts <- table(go_bp_terms$GO)
    go_counts_df <- data.frame(GO=names(go_counts), Count=as.numeric(go_counts), stringsAsFactors=FALSE)
    go_counts_df <- go_counts_df[order(go_counts_df$Count, decreasing=TRUE), ]
    
    # Get GO term names
    if (requireNamespace("GO.db", quietly=TRUE)) {
      library(GO.db)
      go_names <- select(GO.db, keys=go_counts_df$GO, columns="TERM", keytype="GOID")
      go_counts_df <- merge(go_counts_df, go_names, by.x="GO", by.y="GOID", all.x=TRUE)
      go_counts_df$Term <- go_counts_df$TERM
    } else {
      # Try to get names from org.Hs.eg.db
      go_names <- select(org.Hs.eg.db, keys=go_counts_df$GO, columns="TERM", keytype="GO")
      if (nrow(go_names) > 0) {
        go_counts_df <- merge(go_counts_df, go_names, by="GO", all.x=TRUE)
        go_counts_df$Term <- go_counts_df$TERM
      } else {
        go_counts_df$Term <- go_counts_df$GO
      }
    }
    
    # Select top terms for plotting (already sorted descending, so take first top_n)
    top_n <- min(10, nrow(go_counts_df))
    go_plot_df <- go_counts_df[1:top_n, ]
    # Order ascendingly (smallest at top, largest at bottom)
    go_plot_df <- go_plot_df[order(go_plot_df$Count, decreasing=FALSE), ]
    go_plot_df$Term <- factor(go_plot_df$Term, levels=go_plot_df$Term)
    
    # Create bar plot
    max_count <- max(go_plot_df$Count)
    x_breaks <- seq(0, max_count, by=1)  # Integer breaks
    
    p <- ggplot2::ggplot(go_plot_df, ggplot2::aes(x=Count, y=Term)) +
      ggplot2::geom_bar(stat="identity", fill="steelblue", alpha=0.7) +
      ggplot2::scale_x_continuous(breaks=x_breaks, limits=c(0, max_count + 0.5)) +
      ggplot2::labs(title="Top GO biological process term",
                    x="Number of Proteins", y="GO Biological Process Term") +
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold", size=20),
                     axis.text.x=ggplot2::element_text(size=20),
                     axis.text.y=ggplot2::element_text(size=20),
                     axis.title.x=ggplot2::element_text(size=20),
                     axis.title.y=ggplot2::element_text(size=20))
    
    ggplot2::ggsave(paste0(output_dir,"/GO_BP_barplot.pdf"), p, width=14, height=8)
    ggplot2::ggsave(paste0(output_dir,"/GO_BP_barplot.png"), p, width=14, height=8, dpi=300)
    cat(sprintf("GO bar plot saved (top %d terms)\n", top_n))
  }
  
  # Basic KEGG pathway lookup without enrichment statistics
  cat("Performing basic KEGG pathway lookup\n")
  kegg_terms <- select(org.Hs.eg.db, keys=as.character(entrez_ids), columns="PATH", keytype="ENTREZID")
  kegg_terms <- kegg_terms[!is.na(kegg_terms$PATH), ]
  if (nrow(kegg_terms) > 0) {
    # Get pathway names using KEGG REST API
    if (requireNamespace("httr", quietly=TRUE)) {
      library(httr)
      unique_pathways <- unique(kegg_terms$PATH)
      pathway_names <- sapply(unique_pathways, function(path_id) {
        tryCatch({
          # KEGG pathway IDs need "hsa" prefix for human pathways
          kegg_id <- paste0("hsa", path_id)
          url <- paste0("https://rest.kegg.jp/get/", kegg_id)
          response <- GET(url, timeout(10))
          if (status_code(response) == 200) {
            content <- content(response, "text")
            lines <- strsplit(content, "\n")[[1]]
            name_line <- grep("^NAME", lines, value=TRUE)[1]
            if (!is.na(name_line) && length(name_line) > 0) {
              name <- gsub("^NAME\\s+", "", name_line)
              # Remove " - Homo sapiens (human)" suffix if present
              name <- gsub("\\s*-\\s*Homo sapiens.*$", "", name)
              name <- gsub("\\s*$", "", name)  # trim whitespace
              if (nchar(name) > 0) {
                return(name)
              }
            }
          }
          return(paste0("hsa", path_id))
        }, error = function(e) {
          return(paste0("hsa", path_id))
        })
      })
      pathway_df <- data.frame(PATH=unique_pathways, PATHNAME=pathway_names, stringsAsFactors=FALSE)
      kegg_terms <- merge(kegg_terms, pathway_df, by="PATH", all.x=TRUE)
    } else {
      # If httr not available, just add hsa prefix
      kegg_terms$PATHNAME <- paste0("hsa", kegg_terms$PATH)
    }
    write.csv(kegg_terms, file=paste0(output_dir,"/KEGG_pathways.csv"), row.names=FALSE)
    cat(sprintf("Basic KEGG pathways saved to KEGG_pathways.csv (%d pathway-gene associations)\n", nrow(kegg_terms)))
    
    # Create KEGG pathway plots
    if (nrow(kegg_terms) > 0 && "PATHNAME" %in% colnames(kegg_terms)) {
      # Count genes per pathway
      kegg_counts <- table(kegg_terms$PATHNAME)
      kegg_counts_df <- data.frame(Pathway=names(kegg_counts), Count=as.numeric(kegg_counts), stringsAsFactors=FALSE)
      kegg_counts_df <- kegg_counts_df[order(kegg_counts_df$Count, decreasing=TRUE), ]
      
      # Select top pathways for plotting
      top_n <- min(15, nrow(kegg_counts_df))
      kegg_plot_df <- kegg_counts_df[1:top_n, ]
      kegg_plot_df$Pathway <- factor(kegg_plot_df$Pathway, levels=rev(kegg_plot_df$Pathway))
      
      # Truncate long pathway names for better display
      kegg_plot_df$Pathway <- as.character(kegg_plot_df$Pathway)
      kegg_plot_df$Pathway_short <- sapply(kegg_plot_df$Pathway, function(x) {
        if (!is.na(x) && nchar(x) > 60) {
          paste0(substr(x, 1, 57), "...")
        } else {
          x
        }
      })
      kegg_plot_df$Pathway_short <- factor(kegg_plot_df$Pathway_short, levels=rev(kegg_plot_df$Pathway_short))
      
      # Create bar plot
      max_count_kegg <- max(kegg_plot_df$Count)
      x_breaks_kegg <- seq(0, max_count_kegg, by=1)  # Integer breaks
      
      p <- ggplot2::ggplot(kegg_plot_df, ggplot2::aes(x=Count, y=Pathway_short)) +
        ggplot2::geom_bar(stat="identity", fill="darkgreen", alpha=0.7) +
        ggplot2::scale_x_continuous(breaks=x_breaks_kegg, limits=c(0, max_count_kegg + 0.5)) +
        ggplot2::labs(title="Top KEGG Pathways",
                      x="Number of Proteins", y="KEGG Pathway") +
        ggplot2::theme_minimal() +
        ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold", size=20),
                       axis.text.x=ggplot2::element_text(size=20),
                       axis.text.y=ggplot2::element_text(size=20),
                       axis.title.x=ggplot2::element_text(size=20),
                       axis.title.y=ggplot2::element_text(size=20))
      
      ggplot2::ggsave(paste0(output_dir,"/KEGG_barplot.pdf"), p, width=14, height=8)
      ggplot2::ggsave(paste0(output_dir,"/KEGG_barplot.png"), p, width=14, height=8, dpi=300)
      cat(sprintf("KEGG bar plot saved (top %d pathways)\n", top_n))
    }
  } else {
    cat("No KEGG pathways found for the input genes\n")
    # Create empty file for consistency
    write.csv(data.frame(ENTREZID=character(), PATH=character(), PATHNAME=character()), file=paste0(output_dir,"/KEGG_pathways.csv"), row.names=FALSE)
  }
}

# ----------------------------
# STRING Network Visualization
# ----------------------------
string_network_file <- paste0(output_dir, "/string_network.tsv")
if (file.exists(string_network_file)) {
  cat("Creating STRING network visualization...\n")
  if (requireNamespace("igraph", quietly=TRUE)) {
    library(igraph)
    
    # Read STRING network file
    string_data <- read.table(string_network_file, sep="\t", header=FALSE, stringsAsFactors=FALSE)
    # STRING format: protein1, protein2, protein1_name, protein2_name, species, score, ...
    if (ncol(string_data) >= 6) {
      # Extract protein names (columns 3 and 4)
      edges_df <- data.frame(
        from = string_data[, 3],
        to = string_data[, 4],
        weight = string_data[, 6],  # combined score
        stringsAsFactors = FALSE
      )
      
      # Create graph
      g <- graph_from_data_frame(edges_df, directed = FALSE)
      
      # Get GO categories for coloring
      protein_colors <- rep("lightblue", vcount(g))  # default color
      protein_categories <- rep("Other", vcount(g))
      
      # Try to map proteins to GO categories
      if (file.exists(paste0(output_dir, "/GO_BP_terms.csv"))) {
        go_terms <- read.csv(paste0(output_dir, "/GO_BP_terms.csv"), stringsAsFactors=FALSE)
        
        # Get GO term names for top categories
        if (requireNamespace("GO.db", quietly=TRUE)) {
          library(GO.db)
          # Count GO terms per protein
          go_counts <- table(go_terms$ENTREZID)
          top_go_per_protein <- sapply(names(go_counts), function(eid) {
            protein_gos <- go_terms[go_terms$ENTREZID == eid, "GO"]
            if (length(protein_gos) > 0) {
              # Get the most common GO term for this protein
              go_counts_protein <- table(protein_gos)
              top_go <- names(go_counts_protein)[which.max(go_counts_protein)]
              return(top_go)
            }
            return(NA)
          })
          
          # Map Entrez IDs to symbols
          entrez_to_symbol <- mapIds(org.Hs.eg.db, keys=names(top_go_per_protein), 
                                     column="SYMBOL", keytype="ENTREZID", multiVals="first")
          
          # Get GO term names
          go_names <- select(GO.db, keys=unique(top_go_per_protein[!is.na(top_go_per_protein)]), 
                            columns="TERM", keytype="GOID")
          
          # Create color scheme based on GO categories
          # Group GO terms into broad categories
          go_categories <- sapply(go_names$TERM, function(term) {
            term_lower <- tolower(term)
            if (grepl("extracellular|matrix|collagen|cartilage|bone", term_lower)) return("Extracellular Matrix")
            if (grepl("cell adhesion|adhesion", term_lower)) return("Cell Adhesion")
            if (grepl("signaling|signal", term_lower)) return("Signaling")
            if (grepl("metabolic|metabolism", term_lower)) return("Metabolism")
            if (grepl("development|differentiation|growth", term_lower)) return("Development")
            if (grepl("immune|inflammatory|response", term_lower)) return("Immune Response")
            return("Other")
          })
          
          # Color palette
          category_colors <- c(
            "Extracellular Matrix" = "#FF6B6B",
            "Cell Adhesion" = "#4ECDC4",
            "Signaling" = "#45B7D1",
            "Metabolism" = "#FFA07A",
            "Development" = "#98D8C8",
            "Immune Response" = "#F7DC6F",
            "Other" = "#D3D3D3"
          )
          
          # Assign colors to nodes
          for (i in 1:vcount(g)) {
            protein_name <- V(g)$name[i]
            # Try to find Entrez ID for this protein
            entrez_match <- names(entrez_to_symbol)[entrez_to_symbol == protein_name]
            if (length(entrez_match) > 0) {
              entrez_id <- entrez_match[1]
              if (entrez_id %in% names(top_go_per_protein)) {
                go_id <- top_go_per_protein[entrez_id]
                if (!is.na(go_id) && go_id %in% go_names$GOID) {
                  category <- go_categories[go_names$GOID == go_id][1]
                  if (!is.na(category)) {
                    protein_categories[i] <- category
                    protein_colors[i] <- category_colors[category]
                  }
                }
              }
            }
          }
        }
      }
      
      # Set vertex size based on degree (number of connections/interactions)
      # Larger nodes = more protein-protein interactions (higher connectivity)
      # This helps identify hub proteins in the network
      V(g)$size <- degree(g) * 10 + 25  # Much larger nodes, size represents connectivity
      V(g)$label <- V(g)$name
      V(g)$label.cex <- 1.5  # Larger font (will be scaled further in plot)
      V(g)$label.color <- "black"
      V(g)$color <- protein_colors
      V(g)$frame.color <- "darkblue"
      
      # Set edge width based on weight (interaction confidence score)
      # Thicker edges = higher confidence interaction from STRING database
      E(g)$width <- E(g)$weight * 3 + 1
      E(g)$color <- "gray50"
      
      # Layout - use multiple algorithms and choose best, add much more spacing to prevent overlaps
      # Try different layouts and use the one with best spacing
      layouts <- list(
        layout_with_fr(g, niter=10000, start.temp=10),
        layout_with_kk(g, maxiter=5000),
        layout_with_dh(g),
        layout_with_gem(g)
      )
      # Calculate minimum distance between nodes for each layout
      min_dists <- sapply(layouts, function(l) {
        if (nrow(l) < 2) return(0)
        dists <- as.matrix(dist(l))
        diag(dists) <- Inf
        min(dists, na.rm=TRUE)
      })
      # Use layout with maximum minimum distance
      best_idx <- which.max(min_dists)
      layout <- layouts[[best_idx]]
      
      # Scale layout much more aggressively to add spacing (3x instead of 1.5x)
      layout <- layout * 3.0
      
      # Save as PDF with larger plot size and fonts
      pdf(paste0(output_dir, "/string_network.pdf"), width=20, height=16)
      par(cex.main=2.0, cex.lab=1.8, mar=c(5, 4, 4, 2) + 0.1)
      plot(g, layout=layout, 
           vertex.color=V(g)$color,
           vertex.frame.color=V(g)$frame.color,
           main="STRING Protein-Protein Interaction Network\n(Node size = number of connections, Color = GO category)",
           vertex.label.cex=1.8,  # Larger font size
           vertex.label.font=2,
           rescale=TRUE,
           xlim=c(min(layout[,1])*1.1, max(layout[,1])*1.1),
           ylim=c(min(layout[,2])*1.1, max(layout[,2])*1.1))
      # Add legend with larger font
      if (length(unique(protein_categories)) > 1) {
        legend("bottomright", 
               legend=names(category_colors)[names(category_colors) %in% unique(protein_categories)],
               fill=category_colors[names(category_colors) %in% unique(protein_categories)],
               cex=1.8, bty="n", text.font=2)
      }
      dev.off()
      
      # Save as PNG with much larger plot size and fonts
      #png(paste0(output_dir, "/string_network.png"), width=20*300, height=16*300, res=300)
      par(cex.main=2.5, cex.lab=2.0, cex.axis=2.0, mar=c(5, 4, 4, 2) + 0.1)
      plot(g, layout=layout,
           vertex.color=V(g)$color,
           vertex.frame.color=V(g)$frame.color,
           main="STRING Protein-Protein Interaction Network\n(Node size = number of connections, Color = GO category)",
           vertex.label.cex=2.0,  # Larger font size (approximately 20pt)
           vertex.label.font=2,
           rescale=TRUE,
           xlim=c(min(layout[,1])*1.1, max(layout[,1])*1.1),
           ylim=c(min(layout[,2])*1.1, max(layout[,2])*1.1))
      # Add legend with larger font
      if (length(unique(protein_categories)) > 1) {
        legend("bottomright", 
               legend=names(category_colors)[names(category_colors) %in% unique(protein_categories)],
               fill=category_colors[names(category_colors) %in% unique(protein_categories)],
               cex=2.0, bty="n", text.font=2)
      }
      dev.off()
      
      cat(sprintf("STRING network plot saved (%d nodes, %d edges)\n", vcount(g), ecount(g)))
    } else {
      cat("Warning: STRING network file format not recognized\n")
    }
  } else {
    cat("Warning: igraph package not available for network visualization\n")
  }
} else {
  cat("STRING network file not found, skipping network visualization\n")
}

# Heatmap (if expression data exists)
expr_file <- paste0(dirname(input_file), "/protein_expression.csv")
if(file.exists(expr_file)){
    expr_mat <- read.csv(expr_file, row.names=1)
    pheatmap(expr_mat[gene_list,], scale="row", cluster_rows=TRUE, cluster_cols=TRUE,
             filename=paste0(output_dir,"/protein_heatmap.pdf"))
}


# ----------------------------
# Upstream regulator analysis with DoRothEA
# ----------------------------
if (requireNamespace("dorothea", quietly=TRUE) && requireNamespace("viper", quietly=TRUE)) {
library(dorothea)
library(viper)
library(dplyr)

# Load human TF regulons
data(dorothea_hs, package = "dorothea")
regulons <- dorothea_hs %>% filter(confidence %in% c("A","B","C"))

# Create a gene vector for viper (simple presence/absence)
gene_vec <- rep(1, length(entrez_ids))
names(gene_vec) <- names(entrez_ids)

# Estimate TF activity using VIPER
tf_activity <- viper(gene_vec, regulons, verbose=FALSE)

# Save TF activity table
  write.csv(tf_activity, file=paste0(output_dir, "/DoRothEA_TF_activity.csv"))
} else {
  cat("DoRothEA analysis skipped: dorothea or viper packages not available\n")
}
