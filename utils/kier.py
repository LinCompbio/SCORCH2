#######################################################################
# This script contains functions from PyBioMed described in the       #
# published paper: Dong et al. J Cheminform  (2018) 10:16             #
# https://doi.org/10.1186/s13321-018-0270-2. It is intended to        #
# replicate their calculations of Kier Flexibilities of small         #
# molecules to provide features for our scoring functions.            #
#                                                                     #
# Format conversion function at the end of the script added by        #
# @milesmcgibbon                                                      #
#                                                                     #
#######################################################################


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import openbabel as ob
import subprocess
# from oddt import toolkit,shape
# from molent.molent import entropy, binary_similarity, atomic_smiles

def CalculateKappaAlapha1(mol):
    """
    #################################################################
    Calculation of molecular shape index for one bonded fragment
    with Alapha
    ---->kappam1
    Usage:
        result=CalculateKappaAlapha1(mol)
        Input: mol is a molecule object.
        Output: result is a numeric value.
    #################################################################
    """
    P1 = mol.GetNumBonds(onlyHeavy=1)
    A = mol.GetNumHeavyAtoms()
    alpha = rdMolDescriptors.CalcHallKierAlpha(mol)
    denom = P1 + alpha
    if denom:
        kappa = (A + alpha) * (A + alpha - 1) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)

def CalculateKappaAlapha2(mol):
    """
    #################################################################
    Calculation of molecular shape index for two bonded fragment
    with Alapha
    ---->kappam2
    Usage:
        result=CalculateKappaAlapha2(mol)
        Input: mol is a molecule object.
        Output: result is a numeric value.
    #################################################################
    """
    P2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    A = mol.GetNumHeavyAtoms()
    alpha = rdMolDescriptors.CalcHallKierAlpha(mol)
    denom = P2 + alpha
    if denom:
        kappa = (A + alpha - 1) * (A + alpha - 2) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)

def CalculateFlexibility(mol):
    """
    #################################################################
    Calculation of Kier molecular flexibility index
    ---->phi
    Usage:
        result = CalculateFlexibility(mol)
        Input: mol is a molecule object.
        Output: result is a numeric value.`
    #################################################################
    """
    kappa1 = CalculateKappaAlapha1(mol)
    kappa2 = CalculateKappaAlapha2(mol)
    A = mol.GetNumHeavyAtoms()
    phi = kappa1 * kappa2 / (A + 0.0)
    return phi


def SmilePrep(file, molecule_encoder=None):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    mol = ob.OBMol()

    # Read and convert the molecule
    if not obConversion.ReadString(mol, file):
        raise ValueError("Failed to read the PDBQT file with Open Babel.")

    outMDL = obConversion.WriteString(mol)

    # Convert to RDKit molecule
    refmol = Chem.MolFromPDBBlock(outMDL, removeHs=False, sanitize=False)
    if refmol is None:
        raise ValueError("Failed to convert PDB block to RDKit Mol.")

    # Sanitize the molecule if necessary
    # try:
    #     Chem.SanitizeMol(refmol)
    #
    # except Exception as e:
    #     raise ValueError(f"Sanitization failed: {str(e)}")

    # Check and use the molecule encoder if provided
    if molecule_encoder:
        mol_feature = molecule_encoder(refmol)
        return refmol, mol_feature
    else:
        return refmol



# def SmilePrep(file):
#     # Write the PDBQT content to a temporary file
#     with open("temp.pdbqt", "w") as f:
#         f.write(file)
#
#     # Convert PDBQT to PDB using obabel command-line tool
#     subprocess.run(["obabel", "-ipdbqt", "temp.pdbqt", "-opdb", "-O", "temp.pdb", '-h'], check=True)
#
#     # Read the generated PDB file
#     with open("temp.pdb", "r") as f:
#         outMDL = f.read()
#
#     # Clean up temporary files
#     subprocess.run(["rm", "temp.pdbqt", "temp.pdb"], check=True)
#
#     # Create RDKit molecule object from PDB block
#     refmol = Chem.MolFromPDBBlock(outMDL, removeHs=True, sanitize=False)
#
#     return refmol


def SmilePrep2(file):
    # 设置 OpenBabel 转换
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "pdb")
    mol = ob.OBMol()

    # 读取分子文件
    obConversion.ReadString(mol, file)  # Open Babel will uncompress automatically if needed

    # 在加氢之前打印原子总数
    # print("原子总数（加氢前）:", mol.NumAtoms())
    #
    # # 加氢
    # mol.AddHydrogens()
    #
    # # 在加氢之后打印原子总数
    # print("原子总数（加氢后）:", mol.NumAtoms())

    # 将 OpenBabel 分子转换为 PDB 格式的字符串
    outMDL = obConversion.WriteString(mol)

    # 使用 RDKit 从 PDB 格式字符串创建分子对象
    refmol = Chem.MolFromPDBBlock(outMDL, removeHs=False, sanitize=True)

    # 如果需要，这里可以进行 RDKit 分子的其他处理
    # 例如，计算分子的 SMILES 表示或其他特性

    # 这里假设 `atomic_smiles`, `binary_similarity`, 和 `entropy` 函数已经定义
    # 获取环境的 SMILES 表示（具体实现取决于 `atomic_smiles` 函数的定义）
    frag_smiles = atomic_smiles(refmol, max_radius=1)

    # 获取相似性矩阵（具体实现取决于 `binary_similarity` 函数的定义）
    sim = binary_similarity(frag_smiles)

    # 计算熵（具体实现取决于 `entropy` 函数的定义）
    enc = float(entropy(sim))

    return enc,refmol  # 根据您的描述，这里返回计算得到的熵值