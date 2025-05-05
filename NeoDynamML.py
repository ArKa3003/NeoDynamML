

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
import mdtraj as md
import mdtraj.compute as compute
import subprocess
import pickle
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from scipy.spatial.distance import cdist
import itertools
import warnings
from collections import defaultdict
import random
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neodynamml.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NeoDynamML")

class ConfigManager:
    """Manages configuration for the NeoDynamML pipeline."""
    
    def __init__(self):
        self.config = {
            # Input/output paths
            "output_dir": "neodynamml_results",
            "temp_dir": "temp",
            
            # Files and databases
            "reference_genome": "references/human_genome.fasta",
            "hla_database": "references/hla_sequences.fasta",
            "iedb_database": "references/iedb_epitopes.csv",
            
            # Tools paths
            "vep_path": "tools/ensembl-vep",
            "netmhcpan_path": "tools/netMHCpan",
            "blast_path": "tools/blast",
            "gromacs_path": "tools/gromacs/bin",
            
            # Analysis parameters
            "peptide_lengths": [8, 9, 10, 11],
            "binding_threshold": 500,  # IC50 nM
            "strong_binder_threshold": 50,  # IC50 nM
            "md_simulation_time": 100,  # ns
            "md_timestep": 0.002,      # ps
            
            # ML parameters
            "bert_model": "bert_models/scibert",
            "gnn_hidden_channels": 128,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            
            # Molecular dynamics parameters
            "forcefield": "amber99sb-ildn",
            "water_model": "tip3p",
            "box_distance": 1.0,  # nm
            "temperature": 310,    # K
            "pressure": 1,         # bar
            
            # Analysis thresholds
            "rmsd_threshold": 0.3,     # nm
            "rmsf_threshold": 0.25,    # nm
            "hbond_threshold": 3,      # number of stable H-bonds
            "contact_distance": 0.45,  # nm
            
            # Number of parallel processes
            "n_jobs": max(1, os.cpu_count() - 2)
        }
    
    def update_config(self, **kwargs):
        """Update configuration with user-provided parameters."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Create output and temp directories if they don't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True)
    
    def get_config(self):
        """Get the current configuration."""
        return self.config


class VariantProcessor:
    """Process genomic variants to identify potential neoantigens."""
    
    def __init__(self, config):
        self.config = config
        self.vep_path = config["vep_path"]
        self.reference_genome = config["reference_genome"]
        self.output_dir = config["output_dir"]
    
    def annotate_variants(self, vcf_file):
        """Annotate variants using Ensembl VEP."""
        logger.info("Annotating variants with VEP")
        output_file = os.path.join(self.output_dir, "vep_output.tsv")
        
        command = [
            os.path.join(self.vep_path, "vep"),
            "--input_file", vcf_file,
            "--output_file", output_file,
            "--format", "vcf",
            "--fasta", self.reference_genome,
            "--tab",
            "--canonical",
            "--protein",
            "--symbol",
            "--hgvs",
            "--coding_only",
            "--no_intergenic",
            "--pick"
        ]
        
        logger.debug(f"Running command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True)
            logger.info("VEP annotation completed successfully")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"VEP annotation failed: {e.stderr.decode('utf-8')}")
            raise
    
    def extract_missense_variants(self, vep_output):
        """Extract missense variants from VEP output."""
        logger.info("Extracting missense variants")
        
        missense_variants = []
        with open(vep_output, 'r') as f:
            # Skip header lines
            for line in f:
                if not line.startswith("#"):
                    fields = line.strip().split("\t")
                    # Check if the variant is a missense variant
                    if "missense_variant" in fields[6]:
                        variant_info = {
                            "chromosome": fields[0],
                            "position": int(fields[1]),
                            "reference": fields[2],
                            "alternative": fields[3],
                            "gene": fields[4],
                            "protein_change": fields[10] if len(fields) > 10 else "Unknown"
                        }
                        missense_variants.append(variant_info)
        
        logger.info(f"Found {len(missense_variants)} missense variants")
        return missense_variants
    
    def generate_peptides(self, variants, transcript_db):
        """Generate mutated peptides for each missense variant."""
        logger.info("Generating mutated peptides")
        
        peptides = []
        for variant in variants:
            # Extract protein sequence from transcript database
            gene = variant["gene"]
            if gene in transcript_db:
                protein_seq = transcript_db[gene]
                
                # Parse protein change (e.g., p.Ala123Ser)
                if "p." in variant["protein_change"]:
                    change = variant["protein_change"].split("p.")[1]
                    ref_aa = change[0:3]
                    pos = int(change[3:-3])
                    alt_aa = change[-3:]
                    
                    # Convert 3-letter amino acid code to 1-letter
                    ref_aa_1 = seq1(ref_aa)
                    alt_aa_1 = seq1(alt_aa)
                    
                    # Verify the reference amino acid matches the transcript
                    if 0 <= (pos-1) < len(protein_seq) and protein_seq[pos-1] == ref_aa_1:
                        # Generate mutated sequence
                        mutated_seq = protein_seq[:pos-1] + alt_aa_1 + protein_seq[pos:]
                        
                        # Generate peptides of different lengths around the mutation
                        for length in self.config["peptide_lengths"]:
                            half_window = length // 2
                            
                            for i in range(max(0, pos-length), min(len(protein_seq)-length+1, pos)):
                                wt_peptide = protein_seq[i:i+length]
                                mut_peptide = mutated_seq[i:i+length]
                                
                                # Only include peptides that contain the mutation
                                if wt_peptide != mut_peptide:
                                    peptide_info = {
                                        "gene": gene,
                                        "mutation": variant["protein_change"],
                                        "position": pos,
                                        "wt_peptide": wt_peptide,
                                        "mut_peptide": mut_peptide,
                                        "length": length
                                    }
                                    peptides.append(peptide_info)
        
        logger.info(f"Generated {len(peptides)} potential neoantigen peptides")
        return peptides

    def load_transcript_db(self):
        """Load transcript database from reference genome."""
        transcript_db = {}
        try:
            for record in SeqIO.parse(self.reference_genome, "fasta"):
                gene_id = record.id.split("|")[0]
                transcript_db[gene_id] = str(record.seq)
            return transcript_db
        except Exception as e:
            logger.error(f"Failed to load transcript database: {e}")
            return {}


class HLAProcessor:
    """Process HLA types and prepare for neoantigen prediction."""
    
    def __init__(self, config):
        self.config = config
        self.hla_database = config["hla_database"]
    
    def parse_hla_types(self, hla_file):
        """Parse HLA types from input file."""
        logger.info("Parsing HLA types")
        
        hla_types = []
        try:
            with open(hla_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Format: HLA-A*02:01
                        if line.startswith("HLA-"):
                            hla_types.append(line)
                        else:
                            # Try to format: A*02:01
                            parts = line.split("*")
                            if len(parts) == 2:
                                hla_types.append(f"HLA-{line}")
        except Exception as e:
            logger.error(f"Failed to parse HLA types: {e}")
        
        logger.info(f"Found {len(hla_types)} HLA types")
        return hla_types
    
    def calculate_hla_divergence(self, hla_types):
        """Calculate HLA evolutionary divergence (HED)."""
        logger.info("Calculating HLA evolutionary divergence")
        
        # Load HLA sequences
        hla_sequences = {}
        try:
            for record in SeqIO.parse(self.hla_database, "fasta"):
                hla_id = record.id
                if hla_id.startswith("HLA-"):
                    hla_sequences[hla_id] = str(record.seq)
        except Exception as e:
            logger.error(f"Failed to load HLA sequences: {e}")
            return {}
        
        # Calculate pairwise sequence divergence
        hed_scores = {}
        for hla1, hla2 in itertools.combinations(hla_types, 2):
            if hla1 in hla_sequences and hla2 in hla_sequences:
                seq1 = hla_sequences[hla1]
                seq2 = hla_sequences[hla2]
                
                # Calculate sequence identity
                identity = sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)
                divergence = 1 - identity
                
                hed_scores[(hla1, hla2)] = divergence
        
        # Calculate average HED for each HLA
        avg_hed = {}
        for hla in hla_types:
            scores = [hed_scores[(hla, other)] if (hla, other) in hed_scores 
                      else hed_scores[(other, hla)] for other in hla_types if other != hla]
            avg_hed[hla] = np.mean(scores) if scores else 0
        
        logger.info("HLA divergence calculation completed")
        return avg_hed


class NeoantigenPredictor:
    """Predict neoantigens using binding affinity and immunogenicity."""
    
    def __init__(self, config):
        self.config = config
        self.netmhcpan_path = config["netmhcpan_path"]
        self.iedb_database = config["iedb_database"]
        self.output_dir = config["output_dir"]
        self.temp_dir = config["temp_dir"]
        self.binding_threshold = config["binding_threshold"]
        self.strong_binder_threshold = config["strong_binder_threshold"]
        
        # Load IEDB database for similarity analysis
        self.iedb_epitopes = self._load_iedb_database()
    
    def _load_iedb_database(self):
        """Load IEDB database for epitope comparison."""
        epitopes = []
        try:
            df = pd.read_csv(self.iedb_database)
            if 'Epitope' in df.columns:
                epitopes = df['Epitope'].tolist()
            logger.info(f"Loaded {len(epitopes)} epitopes from IEDB database")
        except Exception as e:
            logger.error(f"Failed to load IEDB database: {e}")
        
        return epitopes
    
    def predict_binding_affinity(self, peptides, hla_types):
        """Predict binding affinity using NetMHCpan."""
        logger.info("Predicting HLA binding affinity with NetMHCpan")
        
        # Prepare input file
        input_file = os.path.join(self.temp_dir, "peptides_for_netmhcpan.fasta")
        with open(input_file, 'w') as f:
            for i, peptide in enumerate(peptides):
                f.write(f">mut_{i}\n{peptide['mut_peptide']}\n")
                f.write(f">wt_{i}\n{peptide['wt_peptide']}\n")
        
        # Format HLA types for NetMHCpan
        formatted_hla = []
        for hla in hla_types:
            # Format: HLA-A*02:01 -> HLA-A02:01
            if "*" in hla:
                formatted_hla.append(hla.replace("*", ""))
            else:
                formatted_hla.append(hla)
        
        # Run NetMHCpan for each HLA type
        binding_results = []
        
        for hla in formatted_hla:
            output_file = os.path.join(self.temp_dir, f"netmhcpan_{hla.replace(':', '_')}.txt")
            
            command = [
                self.netmhcpan_path,
                "-a", hla,
                "-f", input_file,
                "-xls",
                "-xlsfile", output_file
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True)
                
                # Parse results
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        if not line.startswith("#") and not line.startswith("Pos"):
                            fields = line.strip().split()
                            if len(fields) >= 11:
                                peptide_id = fields[1]
                                peptide = fields[2]
                                binding_affinity = float(fields[11])  # IC50 nM
                                rank = float(fields[12])
                                
                                is_mutant = peptide_id.startswith("mut_")
                                peptide_idx = int(peptide_id.split("_")[1])
                                
                                binding_result = {
                                    "peptide_idx": peptide_idx,
                                    "peptide": peptide,
                                    "hla": hla,
                                    "is_mutant": is_mutant,
                                    "binding_affinity": binding_affinity,
                                    "rank": rank,
                                    "is_binder": binding_affinity <= self.binding_threshold,
                                    "is_strong_binder": binding_affinity <= self.strong_binder_threshold
                                }
                                binding_results.append(binding_result)
            
            except subprocess.CalledProcessError as e:
                logger.error(f"NetMHCpan failed for HLA {hla}: {e.stderr.decode('utf-8')}")
        
        logger.info(f"Completed binding affinity prediction for {len(binding_results)} peptide-HLA pairs")
        return binding_results
    
    def calculate_differential_affinity(self, binding_results, peptides):
        """Calculate differential binding affinity between mutant and wild-type peptides."""
        logger.info("Calculating differential binding affinity")
        
        differential_results = []
        
        # Group by peptide index and HLA
        grouped_results = {}
        for result in binding_results:
            key = (result["peptide_idx"], result["hla"])
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Calculate differential affinity
        for key, group in grouped_results.items():
            peptide_idx, hla = key
            
            mut_result = next((r for r in group if r["is_mutant"]), None)
            wt_result = next((r for r in group if not r["is_mutant"]), None)
            
            if mut_result and wt_result:
                # Calculate fold change (wt / mut)
                fold_change = wt_result["binding_affinity"] / mut_result["binding_affinity"] if mut_result["binding_affinity"] > 0 else float('inf')
                
                differential_result = {
                    "peptide_idx": peptide_idx,
                    "peptide": peptides[peptide_idx],
                    "hla": hla,
                    "mut_affinity": mut_result["binding_affinity"],
                    "wt_affinity": wt_result["binding_affinity"],
                    "fold_change": fold_change,
                    "mut_is_binder": mut_result["is_binder"],
                    "wt_is_binder": wt_result["is_binder"],
                    "improved_binding": fold_change > 1 and mut_result["is_binder"]
                }
                differential_results.append(differential_result)
        
        logger.info(f"Identified {sum(1 for r in differential_results if r['improved_binding'])} peptides with improved binding")
        return differential_results
    
    def calculate_similarity_to_known_epitopes(self, peptides):
        """Calculate similarity to known epitopes in IEDB."""
        logger.info("Calculating similarity to known epitopes")
        
        similarity_scores = []
        
        for peptide_info in peptides:
            mut_peptide = peptide_info["mut_peptide"]
            
            # Calculate Levenshtein distance to known epitopes
            min_distance = float('inf')
            most_similar_epitope = ""
            
            for epitope in self.iedb_epitopes:
                if len(epitope) >= 8:  # Only compare with epitopes of sufficient length
                    distance = self._levenshtein_distance(mut_peptide, epitope)
                    normalized_distance = distance / max(len(mut_peptide), len(epitope))
                    
                    if normalized_distance < min_distance:
                        min_distance = normalized_distance
                        most_similar_epitope = epitope
            
            similarity_score = 1 - min_distance
            
            similarity_info = {
                "peptide": mut_peptide,
                "similarity_score": similarity_score,
                "most_similar_epitope": most_similar_epitope
            }
            similarity_scores.append(similarity_info)
        
        logger.info("Epitope similarity calculation completed")
        return similarity_scores
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def rank_neoantigens(self, differential_results, similarity_scores):
        """Rank neoantigens based on multiple criteria."""
        logger.info("Ranking neoantigens")
        
        # Combine results
        ranked_neoantigens = []
        
        for diff_result in differential_results:
            if diff_result["improved_binding"]:
                peptide_idx = diff_result["peptide_idx"]
                mut_peptide = diff_result["peptide"]["mut_peptide"]
                
                # Find similarity score
                similarity_info = next((s for s in similarity_scores if s["peptide"] == mut_peptide), None)
                similarity_score = similarity_info["similarity_score"] if similarity_info else 0
                
                # Calculate combined score
                # Higher score for:
                # - Higher fold change in binding affinity
                # - Lower mutant binding affinity (stronger binding)
                # - Higher similarity to known epitopes
                fold_change_score = min(diff_result["fold_change"] / 10, 1)  # Cap at 1
                binding_score = 1 - (diff_result["mut_affinity"] / self.binding_threshold)
                
                combined_score = (0.5 * fold_change_score) + (0.3 * binding_score) + (0.2 * similarity_score)
                
                neoantigen_info = {
                    "peptide_idx": peptide_idx,
                    "gene": diff_result["peptide"]["gene"],
                    "mutation": diff_result["peptide"]["mutation"],
                    "mut_peptide": mut_peptide,
                    "wt_peptide": diff_result["peptide"]["wt_peptide"],
                    "hla": diff_result["hla"],
                    "mut_affinity": diff_result["mut_affinity"],
                    "wt_affinity": diff_result["wt_affinity"],
                    "fold_change": diff_result["fold_change"],
                    "similarity_score": similarity_score,
                    "combined_score": combined_score
                }
                ranked_neoantigens.append(neoantigen_info)
        
        # Sort by combined score
        ranked_neoantigens.sort(key=lambda x: x["combined_score"], reverse=True)
        
        logger.info(f"Ranked {len(ranked_neoantigens)} potential neoantigens")
        return ranked_neoantigens


class MolecularDynamicsSimulator:
    """Run molecular dynamics simulations on peptide-HLA complexes."""
    
    def __init__(self, config):
        self.config = config
        self.gromacs_path = config["gromacs_path"]
        self.output_dir = config["output_dir"]
        self.temp_dir = config["temp_dir"]
        self.forcefield = config["forcefield"]
        self.water_model = config["water_model"]
        self.simulation_time = config["md_simulation_time"]  # ns
        self.timestep = config["md_timestep"]  # ps
        self.temperature = config["temperature"]  # K
        self.pressure = config["pressure"]  # bar
        self.box_distance = config["box_distance"]  # nm
    
    def prepare_peptide_hla_complex(self, peptide, hla, template_pdb=None):
        """Prepare peptide-HLA complex structure for simulation."""
        logger.info(f"Preparing peptide-HLA complex for {peptide} and {hla}")
        
        # Create output directory
        complex_name = f"{peptide}_{hla.replace(':', '_')}"
        complex_dir = os.path.join(self.temp_dir, complex_name)
        os.makedirs(complex_dir, exist_ok=True)
        
        # If template PDB is provided, use it as a starting structure
        if template_pdb:
            pdb_file = os.path.join(complex_dir, "complex.pdb")
            self._modify_template_with_peptide(template_pdb, peptide, pdb_file)
        else:
            # Use homology modeling to create a structure
            pdb_file = self._perform_homology_modeling(peptide, hla, complex_dir)
        
        return pdb_file, complex_dir
    
    def _modify_template_with_peptide(self, template_pdb, peptide, output_pdb):
        """Modify template PDB structure with the new peptide sequence."""
        # This is a simplified version - in practice, you would use tools like PyRosetta
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("template", template_pdb)
        
        # Find peptide chain
        peptide_chain = None
        for chain in structure[0]:
            # Assuming peptide chain is typically shorter than HLA chains
            if len(list(chain.get_residues())) < 30:
                peptide_chain = chain
                break
        
        if peptide_chain:
            # Replace residues with new peptide sequence
            # This is extremely simplified and won't produce accurate structures
            # In a real implementation, you would use more sophisticated methods
            for i, residue in enumerate(peptide_chain):
                if i < len(peptide):
                    residue.resname = peptide[i]
        
        # Save modified structure
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb)
        
        return output_pdb
    
    def _perform_homology_modeling(self, peptide, hla, output_dir):
        """Perform homology modeling to create a peptide-HLA complex."""
        # This is a placeholder - in a real implementation, you would use tools like MODELLER
        # Here we'll just create a dummy PDB file
        pdb_file = os.path.join(output_dir, "homology_model.pdb")
        
        with open(pdb_file, 'w') as f:
            f.write("REMARK This is a placeholder for homology modeling\n")
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        
        return pdb_file
    
    def run_md_simulation(self, pdb_file, complex_dir):
        """Run molecular dynamics simulation using GROMACS."""
        logger.info(f"Setting up MD simulation for {pdb_file}")
        
        # Current working directory
        cwd = os.getcwd()
        
        try:
            # Change to complex directory
            os.chdir(complex_dir)
            
            # 1. Convert PDB to GROMACS format
            self._run_gmx_command("pdb2gmx", 
                                 f"-f {pdb_file} -o processed.gro -water {self.water_model} -ff {self.forcefield}")
            
            # 2. Define simulation box
            self._run_gmx_command("editconf", 
                                 f"-f processed.gro -o box.gro -c -d {self.box_distance} -bt cubic")
            
            # 3. Solvate the box
            self._run_gmx_command("solvate", 
                                 f"-cp box.gro -cs spc216.gro -o solvated.gro -p topol.top")
            
            # 4. Add ions
            self._create_mdp_file("ions.mdp", {"integrator": "steep", "nsteps": 0})
            self._run_gmx_command("grompp", 
                                 f"-f ions.mdp -c solvated.gro -p topol.top -o ions.tpr")
            
            self._run_gmx_command("genion", 
                                 f"-s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral")
            
            # 5. Energy minimization
            self._create_mdp_file("em.mdp", {
                "integrator": "steep",
                "emtol": 1000.0,
                "nsteps": 50000
            })
            
            self._run_gmx_command("grompp", 
                                 f"-f em.mdp -c ionized.gro -p topol.top -o em.tpr")
            
            self._run_gmx_command("mdrun", 
                                 f"-v -deffnm em")
            
            # 6. NVT equilibration
            self._create_mdp_file("nvt.mdp", {
                "integrator": "md",
                "dt": self.timestep,
                "nsteps": int(100 / self.timestep),  # 100 ps
                "tcoupl": "V-rescale",
                "ref_t": self.temperature,
                "pcoupl": "no"
            })
            
            self._run_gmx_command("grompp", 
                                 f"-f nvt.mdp -c em.gro -p topol.top -o nvt.tpr")
            
            self._run_gmx_command("mdrun", 
                                 f"-v -deffnm nvt")
            
            # 7. NPT equilibration
            self._create_mdp_file("npt.mdp", {
                "integrator": "md",
                "dt": self.timestep,
                "nsteps": int(100 / self.timestep),  # 100 ps
                "tcoupl": "V-rescale",
                "ref_t": self.temperature,
                "pcoupl": "Parrinello-Rahman",
                "ref_p": self.pressure
            })
            
            self._run_gmx_command("grompp", 
                                 f"-f npt.mdp -c nvt.gro -p topol.top -o npt.tpr")
            
            self._run_gmx_command("mdrun", 
                                 f"-v -deffnm npt")
            
            # 8. Production MD
            steps = int((self.simulation_time * 1000) / self.timestep)  # Convert ns to ps
            self._create_mdp_file("md.mdp", {
                "integrator": "md",
                "dt": self.timestep,
                "nsteps": steps,
                "tcoupl": "V-rescale",
                "ref_t": self.temperature,
                "pcoupl": "Parrinello-Rahman",
                "ref_p": self.pressure,
                "nstxout": int(10 / self.timestep),  # Save coordinates every 10 ps
                "nstvout": int(10 / self.timestep),
                "nstfout": int(10 / self.timestep),
                "nstxtcout": int(1 / self.timestep)   # Save compressed trajectory every 1 ps
            })
            
            self._run_gmx_command("grompp", 
                                 f"-f md.mdp -c npt.gro -p topol.top -o md.tpr")
            
            self._run_gmx_command("mdrun", 
                                 f"-v -deffnm md")
            
            # Return to original directory
            trajectory_file = os.path.join(complex_dir, "md.xtc")
            structure_file = os.path.join(complex_dir, "md.gro")
            
            logger.info(f"MD simulation completed successfully: {trajectory_file}")
            return trajectory_file, structure_file
            
        except Exception as e:
            logger.error(f"MD simulation failed: {e}")
            raise
        finally:
            # Return to original directory
            os.chdir(cwd)
    
    def _run_gmx_command(self, command, args):
        """Run a GROMACS command."""
        full_command = f"{os.path.join(self.gromacs_path, 'gmx')} {command} {args}"
        logger.debug(f"Running GROMACS command: {full_command}")
        
        try:
            process = subprocess.run(full_command, shell=True, check=True, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True, input="\n")
            return process.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"GROMACS command failed: {e.stderr}")
            raise
    
    def _create_mdp_file(self, filename, parameters):
        """Create GROMACS MDP parameter file."""
        with open(filename, 'w') as f:
            for key, value in parameters.items():
                f.write(f"{key} = {value}\n")
            
            # Add default parameters if not specified
            default_params = {
                "cutoff-scheme": "Verlet",
                "coulombtype": "PME",
                "rcoulomb": 1.0,
                "rvdw": 1.0,
                "pbc": "xyz",
                "constraints": "all-bonds",
                "constraint-algorithm": "LINCS"
            }
            
            for key, value in default_params.items():
                if key not in parameters:
                    f.write(f"{key} = {value}\n")
    
    def analyze_trajectory(self, trajectory_file, structure_file, peptide_residues=None):
        """Analyze MD trajectory for stability and binding metrics."""
        logger.info(f"Analyzing MD trajectory: {trajectory_file}")
        
        # Load trajectory
        try:
            traj = md.load(trajectory_file, top=structure_file)
            
            # Calculate RMSD
            reference = traj[0]
            rmsd = md.rmsd(traj, reference)
            
            # Calculate RMSF
            rmsf = md.rmsf(traj, reference)
            
            # Calculate hydrogen bonds
            hbonds = md.baker_hubbard(traj, periodic=False)
            n_hbonds = len(hbonds)
            
            # Calculate contacts (if peptide residues are specified)
            contacts = None
            if peptide_residues:
                # Get atom indices for peptide and HLA
                peptide_indices = traj.topology.select(f"resid {' '.join(map(str, peptide_residues))}")
                hla_indices = traj.topology.select(f"not resid {' '.join(map(str, peptide_residues))}")
                
                # Calculate minimum distances between peptide and HLA
                contacts = md.compute_contacts(traj, contacts=[(peptide_indices, hla_indices)], scheme='closest-heavy')[0]
            
            # Compute secondary structure
            dssp = md.compute_dssp(traj)
            
            analysis_results = {
                "rmsd": rmsd,
                "rmsf": rmsf,
                "hbonds": n_hbonds,
                "contacts": contacts,
                "dssp": dssp,
                "trajectory": traj
            }
            
            logger.info(f"Trajectory analysis completed: RMSD range [{np.min(rmsd):.3f}, {np.max(rmsd):.3f}] nm")
            return analysis_results
        
        except Exception as e:
            logger.error(f"Trajectory analysis failed: {e}")
            return None


class GNNPeptideHLAModel(nn.Module):
    """Graph Neural Network model for peptide-HLA binding prediction."""
    
    def __init__(self, hidden_channels, num_features=20, num_classes=1):
        super(GNNPeptideHLAModel, self).__init__()
        self.hidden_channels = hidden_channels
        
        # GNN layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Readout and classification
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x


class MLPredictor:
    """ML-based predictions for peptide-HLA binding and immunogenicity."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.bert_model_path = config["bert_model"]
        self.gnn_hidden_channels = config["gnn_hidden_channels"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        
        # Initialize models
        self.bert_tokenizer = None
        self.bert_model = None
        self.gnn_model = None
        self.stability_classifier = None
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            # Load BERT model and tokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
            self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_model_path)
            self.bert_model.to(self.device)
            
            # Load GNN model
            self.gnn_model = GNNPeptideHLAModel(hidden_channels=self.gnn_hidden_channels)
            gnn_path = os.path.join(self.config["output_dir"], "gnn_model.pt")
            if os.path.exists(gnn_path):
                self.gnn_model.load_state_dict(torch.load(gnn_path, map_location=self.device))
            self.gnn_model.to(self.device)
            
            # Load stability classifier
            stability_path = os.path.join(self.config["output_dir"], "stability_classifier.pkl")
            if os.path.exists(stability_path):
                self.stability_classifier = joblib.load(stability_path)
            
            logger.info("Models loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Initialize new models
            self.gnn_model = GNNPeptideHLAModel(hidden_channels=self.gnn_hidden_channels)
            self.gnn_model.to(self.device)
            self.stability_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train_gnn_model(self, training_data):
        """Train the GNN model on peptide-HLA binding data."""
        logger.info("Training GNN model for peptide-HLA binding")
        
        # Prepare data
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = self._create_graph_data_loader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = self._create_graph_data_loader(val_data, batch_size=self.batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            # Training
            self.gnn_model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.view(-1), batch.y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs
            
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.gnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out.view(-1), batch.y)
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss /= len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.gnn_model.state_dict(), os.path.join(self.config["output_dir"], "gnn_model.pt"))
        
        # Load best model
        self.gnn_model.load_state_dict(torch.load(os.path.join(self.config["output_dir"], "gnn_model.pt")))
        logger.info("GNN model training completed")
    
    def _create_graph_data_loader(self, data_list, batch_size=32, shuffle=True):
        """Create a graph data loader from a list of peptide-HLA data."""
        graphs = []
        
        for item in data_list:
            peptide = item["peptide"]
            hla = item["hla"]
            binding = item["binding"]
            
            # Create graph from peptide and HLA
            x, edge_index = self._peptide_hla_to_graph(peptide, hla)
            y = torch.tensor([float(binding)], dtype=torch.float)
            
            graph = Data(x=x, edge_index=edge_index, y=y)
            graphs.append(graph)
        
        return torch.utils.data.DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
    
    def _peptide_hla_to_graph(self, peptide, hla):
        """Convert peptide and HLA sequences to a graph."""
        # This is a simplified version - in practice, you would use more sophisticated methods
        # to create a graph representing the peptide-HLA complex
        
        # Amino acid features (one-hot encoding)
        aa_dict = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        # Create nodes for peptide and HLA
        nodes = []
        for aa in peptide:
            features = torch.zeros(20)
            if aa in aa_dict:
                features[aa_dict[aa]] = 1.0
            nodes.append(features)
        
        # Add some HLA representative nodes (simplified)
        hla_type = hla.split("-")[1] if "-" in hla else hla
        for i in range(5):  # Add 5 representative nodes
            features = torch.zeros(20)
            if i < len(hla_type) and hla_type[i] in aa_dict:
                features[aa_dict[hla_type[i]]] = 1.0
            nodes.append(features)
        
        x = torch.stack(nodes)
        
        # Create edges (fully connected within peptide and HLA, plus connections between them)
        edges = []
        peptide_len = len(peptide)
        total_nodes = len(nodes)
        
        # Connect peptide residues
        for i in range(peptide_len):
            for j in range(peptide_len):
                if i != j:
                    edges.append([i, j])
        
        # Connect HLA residues
        for i in range(peptide_len, total_nodes):
            for j in range(peptide_len, total_nodes):
                if i != j:
                    edges.append([i, j])
        
        # Connect peptide to HLA
        for i in range(peptide_len):
            for j in range(peptide_len, total_nodes):
                edges.append([i, j])
                edges.append([j, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return x, edge_index
    
    def train_stability_classifier(self, md_results):
        """Train a classifier to predict peptide-HLA complex stability from MD results."""
        logger.info("Training stability classifier from MD results")
        
        # Extract features from MD results
        X = []
        y = []
        
        for result in md_results:
            # Extract features
            rmsd_mean = np.mean(result["rmsd"])
            rmsd_std = np.std(result["rmsd"])
            rmsf_mean = np.mean(result["rmsf"])
            rmsf_std = np.std(result["rmsf"])
            hbonds = result["hbonds"]
            
            # Additional features from contacts if available
            contact_mean = np.mean(result["contacts"]) if result["contacts"] is not None else 0
            contact_std = np.std(result["contacts"]) if result["contacts"] is not None else 0
            
            # Create feature vector
            features = [rmsd_mean, rmsd_std, rmsf_mean, rmsf_std, hbonds, contact_mean, contact_std]
            X.append(features)
            
            # Set label (stable if RMSD < threshold)
            stable = rmsd_mean < self.config["rmsd_threshold"]
            y.append(int(stable))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        self.stability_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.stability_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.stability_classifier.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        logger.info(f"Stability classifier trained with accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        joblib.dump(self.stability_classifier, os.path.join(self.config["output_dir"], "stability_classifier.pkl"))
        joblib.dump(scaler, os.path.join(self.config["output_dir"], "feature_scaler.pkl"))
        
        return accuracy
    
    def predict_tcell_reactivity(self, peptides, hla_types):
        """Predict T-cell reactivity using BERT-based model."""
        logger.info("Predicting T-cell reactivity")
        
        if self.bert_model is None or self.bert_tokenizer is None:
            logger.error("BERT model not available for T-cell reactivity prediction")
            return []
        
        predictions = []
        
        self.bert_model.eval()
        with torch.no_grad():
            for peptide in peptides:
                for hla in hla_types:
                    # Format input for BERT
                    input_text = f"peptide: {peptide} hla: {hla}"
                    
                    # Tokenize
                    inputs = self.bert_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Predict
                    outputs = self.bert_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                    
                    prediction = {
                        "peptide": peptide,
                        "hla": hla,
                        "reactivity_score": float(probabilities[0])
                    }
                    predictions.append(prediction)
        
        logger.info(f"T-cell reactivity prediction completed for {len(predictions)} peptide-HLA pairs")
        return predictions
    
    def predict_peptide_hla_binding(self, peptides, hla_types):
        """Predict peptide-HLA binding using GNN model."""
        logger.info("Predicting peptide-HLA binding with GNN")
        
        if self.gnn_model is None:
            logger.error("GNN model not available for binding prediction")
            return []
        
        predictions = []
        
        self.gnn_model.eval()
        with torch.no_grad():
            for peptide in peptides:
                for hla in hla_types:
                    # Create graph
                    x, edge_index = self._peptide_hla_to_graph(peptide, hla)
                    data = Data(x=x, edge_index=edge_index)
                    data = data.to(self.device)
                    
                    # Predict
                    out = self.gnn_model(data.x, data.edge_index, torch.zeros(1, dtype=torch.long, device=self.device))
                    binding_prob = torch.sigmoid(out).item()
                    
                    prediction = {
                        "peptide": peptide,
                        "hla": hla,
                        "binding_prob": binding_prob
                    }
                    predictions.append(prediction)
        
        logger.info(f"GNN binding prediction completed for {len(predictions)} peptide-HLA pairs")
        return predictions


class NeoantigenDynamicsAnalyzer:
    """Analyze dynamics of neoantigens and integrate ML predictions."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config["output_dir"]
        self.md_analyzer = MolecularDynamicsSimulator(config)
        self.ml_predictor = MLPredictor(config)
    
    def analyze_neoantigen_candidates(self, candidates, hla_types):
        """Perform comprehensive analysis on top neoantigen candidates."""
        logger.info(f"Analyzing dynamics for {len(candidates)} neoantigen candidates")
        
        analysis_results = []
        
        # Process top candidates
        for candidate in candidates:
            peptide = candidate["mut_peptide"]
            hla = candidate["hla"]
            
            # 1. Prepare peptide-HLA complex
            pdb_file, complex_dir = self.md_analyzer.prepare_peptide_hla_complex(peptide, hla)
            
            # 2. Run MD simulation
            trajectory_file, structure_file = self.md_analyzer.run_md_simulation(pdb_file, complex_dir)
            
            # 3. Analyze trajectory
            md_analysis = self.md_analyzer.analyze_trajectory(trajectory_file, structure_file)
            
            # 4. ML-based predictions
            binding_prediction = self.ml_predictor.predict_peptide_hla_binding([peptide], [hla])
            reactivity_prediction = self.ml_predictor.predict_tcell_reactivity([peptide], [hla])
            
            # 5. Extract key metrics
            rmsd_mean = np.mean(md_analysis["rmsd"])
            rmsf_mean = np.mean(md_analysis["rmsf"])
            hbonds = md_analysis["hbonds"]
            
            binding_prob = binding_prediction[0]["binding_prob"] if binding_prediction else 0
            reactivity_score = reactivity_prediction[0]["reactivity_score"] if reactivity_prediction else 0
            
            # 6. Calculate stability score
            stability_features = [
                rmsd_mean, 
                np.std(md_analysis["rmsd"]),
                rmsf_mean,
                np.std(md_analysis["rmsf"]),
                hbonds,
                np.mean(md_analysis["contacts"]) if md_analysis["contacts"] is not None else 0,
                np.std(md_analysis["contacts"]) if md_analysis["contacts"] is not None else 0
            ]
            
            # Check if stability classifier is available
            stability_score = 0
            if self.ml_predictor.stability_classifier is not None:
                # Load scaler
                scaler_path = os.path.join(self.output_dir, "feature_scaler.pkl")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    scaled_features = scaler.transform([stability_features])
                    stability_score = self.ml_predictor.stability_classifier.predict_proba(scaled_features)[0][1]
            
            # 7. Combine scores
            combined_score = (
                0.3 * binding_prob +
                0.3 * stability_score +
                0.3 * reactivity_score +
                0.1 * candidate["combined_score"]  # Original neoantigen score
            )
            
            result = {
                "peptide": peptide,
                "hla": hla,
                "gene": candidate["gene"],
                "mutation": candidate["mutation"],
                "binding_affinity": candidate["mut_affinity"],
                "fold_change": candidate["fold_change"],
                "rmsd_mean": float(rmsd_mean),
                "rmsf_mean": float(rmsf_mean),
                "hbonds": int(hbonds),
                "binding_prob": float(binding_prob),
                "stability_score": float(stability_score),
                "reactivity_score": float(reactivity_score),
                "combined_score": float(combined_score),
                "trajectory_file": trajectory_file,
                "is_stable": rmsd_mean < self.config["rmsd_threshold"]
            }
            
            analysis_results.append(result)
            
            # Save analysis plot
            self._save_analysis_plot(md_analysis, peptide, hla)
        
        # Sort by combined score
        analysis_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        logger.info("Neoantigen dynamics analysis completed")
        return analysis_results
    
    def _save_analysis_plot(self, md_analysis, peptide, hla):
        """Create and save analysis plot for MD trajectory."""
        plot_file = os.path.join(self.output_dir, f"md_plot_{peptide}_{hla.replace(':', '_')}.png")
        
        plt.figure(figsize=(12, 10))
        
        # Plot RMSD
        plt.subplot(2, 2, 1)
        plt.plot(md_analysis["rmsd"])
        plt.title("RMSD")
        plt.xlabel("Frame")
        plt.ylabel("RMSD (nm)")
        
        # Plot RMSF
        plt.subplot(2, 2, 2)
        plt.plot(md_analysis["rmsf"])
        plt.title("RMSF")
        plt.xlabel("Residue")
        plt.ylabel("RMSF (nm)")
        
        # Plot hydrogen bonds
        plt.subplot(2, 2, 3)
        plt.hist(md_analysis["hbonds"], bins=20)
        plt.title("Hydrogen Bonds")
        plt.xlabel("Number of H-bonds")
        plt.ylabel("Frequency")
        
        # Plot secondary structure
        if md_analysis["dssp"] is not None:
            plt.subplot(2, 2, 4)
            dssp_counts = np.zeros((len(md_analysis["dssp"]), 8))
            for i, frame in enumerate(md_analysis["dssp"]):
                for j, ss in enumerate(frame):
                    dssp_counts[i, ss] += 1
            
            plt.imshow(dssp_counts.T, aspect='auto', interpolation='none')
            plt.title("Secondary Structure")
            plt.xlabel("Frame")
            plt.ylabel("Secondary Structure")
            plt.colorbar(label="Count")
        
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()


class ReportGenerator:
    """Generate comprehensive reports for neoantigen analysis."""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config["output_dir"]
    
    def generate_summary_report(self, neoantigens, analysis_results):
        """Generate a summary report of neoantigen analysis."""
        logger.info("Generating summary report")
        
        report_file = os.path.join(self.output_dir, "neoantigen_summary_report.html")
        
        with open(report_file, 'w') as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<title>Neoantigen Analysis Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1, h2 { color: #2c3e50; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #f2f2f2; }\n")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
            f.write(".good { color: green; }\n")
            f.write(".bad { color: red; }\n")
            f.write("</style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            
            f.write("<h1>Neoantigen Analysis Report</h1>\n")
            f.write(f"<p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Summary statistics
            f.write("<h2>Summary Statistics</h2>\n")
            f.write("<ul>\n")
            f.write(f"<li>Total neoantigens analyzed: {len(neoantigens)}</li>\n")
            f.write(f"<li>Top candidates analyzed with MD: {len(analysis_results)}</li>\n")
            
            stable_count = sum(1 for result in analysis_results if result["is_stable"])
            f.write(f"<li>Stable candidates: {stable_count} ({stable_count/len(analysis_results)*100:.1f}%)</li>\n")
            
            high_binding_count = sum(1 for result in analysis_results if result["binding_prob"] > 0.7)
            f.write(f"<li>High binding probability: {high_binding_count} ({high_binding_count/len(analysis_results)*100:.1f}%)</li>\n")
            
           
