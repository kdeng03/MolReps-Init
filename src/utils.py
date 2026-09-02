import yaml
from PIL import Image
import selfies as sf
import deepsmiles
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Mol rep utility functions
# ============================================================================

def _parse_to_mol(rep: str, rep_type: str) -> Chem.Mol | None:
    """
    Args:
        rep: 分子表示字符串
        rep_type: 表示类型 (can_smiles, can_selfies, can_deepsmiles, inchi)
    
    Returns:
        RDKit Mol 对象，解析失败返回 None
    """
    if not rep or not isinstance(rep, str):
        return None

    rep = rep.strip()
    
    try:
        match rep_type:
            case "can_smiles":
                return Chem.MolFromSmiles(rep)
            case "can_selfies":
                decoded_smiles = sf.decoder(rep)
                return Chem.MolFromSmiles(decoded_smiles)
            case "can_deepsmiles":
                converter = deepsmiles.Converter(rings=True, branches=True)
                decoded_smiles = converter.decode(rep)
                return Chem.MolFromSmiles(decoded_smiles)
            case "inchi":
                return Chem.MolFromInchi(rep)
            case _:
                raise ValueError(f"Unsupported rep_type: {rep_type}")
    except Exception:
        return None


def _get_canonical_smiles(mol: Chem.Mol) -> str | None:
    """获取分子的 canonical SMILES"""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _get_inchikey(mol: Chem.Mol) -> str | None:
    """获取分子的 InChIKey"""
    try:
        inchi = Chem.MolToInchi(mol)
        if inchi:
            # InChIKey 是 InChI 的哈希，格式为 14-10-1
            return Chem.InchiToInchiKey(inchi)
        return None
    except Exception:
        return None


def _compute_tanimoto(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    计算两个分子的 Tanimoto 相似度 (Morgan fingerprint, radius=2, nBits=2048)
    """
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ============================================================================
# 公开 API: 分子评估指标计算
# ============================================================================

def compute_mol_metrics(gt_rep: str, pred_rep: str, rep_type: str) -> dict:
    """
    计算分子表示评估指标
    
    Args:
        gt_rep: Ground truth 分子表示
        pred_rep: 模型预测的分子表示
        rep_type: 表示类型 (can_smiles, can_selfies, can_deepsmiles, inchi)
    
    Returns:
        dict 包含所有指标:
        - is_gt_valid: GT 是否可解析
        - is_pred_valid: Pred 是否可解析
        - is_em: Exact Match (字符串完全匹配)
        - is_can_smiles_match: Canonical SMILES 是否匹配
        - is_inchikey_match: InChIKey 是否匹配
        - tanimoto_sim: Tanimoto 相似度 (0.0-1.0)
        - _parse_status: 解析状态 ("success", "gt_failed", "pred_failed", "both_failed", "missing")
    """
    # 处理缺失输入
    if gt_rep is None or pred_rep is None:
        return {
            "is_gt_valid": False,
            "is_pred_valid": False,
            "is_em": False,
            "is_can_smiles_match": False,
            "is_inchikey_match": False,
            "tanimoto_sim": 0.0,
            "_parse_status": "missing"
        }
    
    # 1. Exact Match (纯字符串比较，不依赖解析)
    is_em = gt_rep.strip() == pred_rep.strip()
    
    # 2. 解析分子
    gt_mol = _parse_to_mol(gt_rep, rep_type)
    pred_mol = _parse_to_mol(pred_rep, rep_type)
    
    is_gt_valid = gt_mol is not None
    is_pred_valid = pred_mol is not None
    
    # 3. 确定解析状态
    if is_gt_valid and is_pred_valid:
        parse_status = "success"
    elif is_gt_valid:
        parse_status = "pred_failed"
    elif is_pred_valid:
        parse_status = "gt_failed"
    else:
        parse_status = "both_failed"
    
    # 4. 计算结构匹配指标
    if is_gt_valid and is_pred_valid:
        gt_smiles = _get_canonical_smiles(gt_mol)
        pred_smiles = _get_canonical_smiles(pred_mol)
        is_can_smiles_match = (gt_smiles is not None and pred_smiles is not None and 
                               gt_smiles == pred_smiles)
        
        gt_inchikey = _get_inchikey(gt_mol)
        pred_inchikey = _get_inchikey(pred_mol)
        is_inchikey_match = (gt_inchikey is not None and pred_inchikey is not None and 
                             gt_inchikey == pred_inchikey)
        
        tanimoto_sim = _compute_tanimoto(gt_mol, pred_mol)
    else:
        # 解析失败时，结构指标返回 False/0.0，计入分母避免虚高
        is_can_smiles_match = False
        is_inchikey_match = False
        tanimoto_sim = 0.0
    
    return {
        "is_gt_valid": is_gt_valid,
        "is_pred_valid": is_pred_valid,
        "is_em": is_em,
        "is_can_smiles_match": is_can_smiles_match,
        "is_inchikey_match": is_inchikey_match,
        "tanimoto_sim": tanimoto_sim,
        "_parse_status": parse_status
    }