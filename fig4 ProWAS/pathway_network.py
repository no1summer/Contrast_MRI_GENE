import os
import subprocess
import pandas as pd
import requests
from py2cytoscape import cyrest

# ---------------------------
# Paths
# ---------------------------
base_dir = "/data484_4/txia2/mocov2/proWAS/protein_common_analysis"
protein_file = "/data484_4/txia2/mocov2/proWAS/protein_common_analysis/common_proteins_LOD_min.csv"
output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)
r_script = os.path.join(base_dir, "scripts", "enrichment_analysis.r")

# ---------------------------
# Step 1: Run R enrichment script
# ---------------------------
print("Running R enrichment analysis...")
subprocess.run(["Rscript", r_script, protein_file, output_dir], check=True)
print("R enrichment completed!")

# ---------------------------
# Step 2: Get STRING network
# ---------------------------
df = pd.read_csv(protein_file)
proteins = df['common_protein'].tolist()  # adjust column name

string_api_url = "https://string-db.org/api"
params = {
    "identifiers": "%0d".join(proteins),  # %0d is newline delimiter for STRING API
    "species": 9606,
    "required_score": 700,
    "add_nodes": 0
}
try:
    response = requests.get(f"{string_api_url}/tsv-no-header/network", params=params, timeout=30)
    response.raise_for_status()
except Exception as e:
    print(f"Warning: STRING API request failed: {e}")
    print("Continuing without STRING network...")
    response = None

if response and response.text.strip():

    network_file = os.path.join(output_dir, "string_network.tsv")
    with open(network_file, "w") as f:
        f.write(response.text)
    print(f"STRING network saved to {network_file}")
else:
    print("Warning: STRING API returned empty response. Network file not created.")
    network_file = None

# ---------------------------
# Step 3: Connect Cytoscape
# ---------------------------
if network_file and os.path.exists(network_file):
    try:
        cy = cyrest.cyclient()
        # Test connection
        cy.version()
        print("Connected to Cytoscape")
        
        # Import network file
        network = cy.network.import_file(network_file, data_type='tsv')
        network_suid = network.get('networkSUID')
        print(f"Network imported to Cytoscape (SUID: {network_suid})")
        
        # ---------------------------
        # Step 4: Run MCODE
        # ---------------------------
        try:
            cy.commands.commands_post('mcode cluster clusterOnlySelected=false')
            print("MCODE clustering completed")
        except Exception as e:
            print("MCODE failed:", e)
        
        # ---------------------------
        # Step 5: Run CytoHubba
        # ---------------------------
        try:
            cy.commands.commands_post('cytohubba hubDetect method="Degree" topK=10')
            print("CytoHubba hub detection completed")
        except Exception as e:
            print("CytoHubba failed:", e)
        
        # ---------------------------
        # Step 6: Export network image
        # ---------------------------
        try:
            export_file = os.path.join(output_dir, "cytoscape_network.png")
            cy.network.export(network_suid, export_file, type='PNG', width=1200, height=1000)
            print(f"Network image exported to {export_file}")
        except Exception as e:
            print(f"Network export failed: {e}")
            print("Note: Network file is available at", network_file)
            
    except Exception as e:
        print(f"Warning: Could not connect to Cytoscape: {e}")
        print("Note: Cytoscape must be running to import networks and run analyses.")
        if network_file:
            print("The STRING network file has been saved to:", network_file)
            print("You can manually import this file into Cytoscape if needed.")
else:
    print("Skipping Cytoscape steps: STRING network file not available.")
