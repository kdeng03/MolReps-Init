import yaml
from PIL import Image
import selfies as sf
import deepsmiles
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw


CONFIG_FILE = "configs/mol_rep_schema.yaml"


class MolReps:
    @staticmethod
    def smiles_to_mol(smiles: str) -> Chem.rdchem.Mol|None:
        """Convert SMILES to RDKit Mol object."""
        return Chem.MolFromSmiles(smiles)

    @staticmethod
    def mol_to_can_smiles(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to canonical SMILES."""
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    @staticmethod
    def mol_to_can_selfies(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to canonical SELFIES."""
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return sf.encoder(can_smiles)

    @staticmethod
    def mol_to_can_deepsmiles(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to canonical DeepSMILES."""
        converter = deepsmiles.Converter(rings=True, branches=True)
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return converter.encode(can_smiles)

    @staticmethod
    def mol_to_inchi(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to InChI."""
        return Chem.MolToInchi(mol)

    @staticmethod
    def mol_to_inchikey(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to InChIKey."""
        return Chem.InchiToInchiKey(Chem.MolToInchi(mol))

    @staticmethod
    def mol_to_iupac(mol: Chem.rdchem.Mol) -> str|None:
        """Convert RDKit Mol object to IUPAC name using PubChemPy."""
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        compound = pcp.get_compounds(can_smiles, 'smiles')
        if compound:
            return compound[0].iupac_name
        else:
            return None

    @staticmethod
    def mol_to_cml(mol: Chem.rdchem.Mol) -> str:
        """Convert RDKit Mol object to CML."""
        return Chem.MolToCMLBlock(mol)

    @staticmethod
    def mol_to_image(mol: Chem.rdchem.Mol, size=(300, 300)) -> Image.Image:
        """Render RDKit Mol object to an image."""
        return Draw.MolToImage(mol, size=size)

    def __init__(self, smiles: str):
        self.raw_smiles = smiles
        self.mol = self.smiles_to_mol(smiles)

        with open(CONFIG_FILE, "r") as f:
            self.config_reps = yaml.safe_load(f)["representations"]

        if self.mol:
            self.can_smiles: str = self.mol_to_can_smiles(self.mol)
            # --- random SMILES (unique, no canonical dup) ---
            self.random_smiles: list[str] = []
            seen_smiles = {self.can_smiles}
            attempts = 0
            while len(self.random_smiles) < 5 and attempts < 1000:
                sm = Chem.MolToSmiles(self.mol, doRandom=True)
                attempts += 1
                if sm not in seen_smiles:
                    seen_smiles.add(sm)
                    self.random_smiles.append(sm)

            if attempts == 1000:
                print(f"Warning: Only generated {len(self.random_smiles)} unique random SMILES for {smiles} after 1000 attempts.")

            self.can_selfies: str = sf.encoder(self.can_smiles)
            self.random_selfies: list[str] = [sf.encoder(sm) for sm in self.random_smiles]
            converter = deepsmiles.Converter(rings=True, branches=True)
            self.can_deepsmiles: str = converter.encode(self.can_smiles)
            self.random_deepsmiles: list[str] = [converter.encode(sm) for sm in self.random_smiles]
            self.inchi: str = self.mol_to_inchi(self.mol)
            self.inchikey: str = self.mol_to_inchikey(self.mol)
            self.iupac: str|None = self.mol_to_iupac(self.mol)
            self.cml: str = self.mol_to_cml(self.mol)
            self.mol_image: Image.Image = self.mol_to_image(self.mol)
        else:
            print(f"Error: Invalid SMILES string '{smiles}'. Could not convert to RDKit Mol object.")
            self.can_smiles = None
            self.random_smiles = []
            self.can_selfies = None
            self.random_selfies = []
            self.can_deepsmiles = None
            self.random_deepsmiles = []
            self.inchi = None
            self.inchikey = None
            self.iupac = None
            self.cml = None
            self.mol_image = None


if __name__ == "__main__":
    smiles = "CCO"
    mol_reps = MolReps(smiles)   
    print()
