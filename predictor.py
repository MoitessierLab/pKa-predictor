import sys
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure pKaLearn GNN code is on sys.path
_PKA_GNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "GNN"))
if _PKA_GNN_DIR not in sys.path:
    sys.path.insert(0, _PKA_GNN_DIR)

_pka_predict_fn = None

def _get_pka_predict_fn():
    global _pka_predict_fn
    if _pka_predict_fn is None:
        from predict import predict
        _pka_predict_fn = predict
    return _pka_predict_fn


def predict_protonation_states(smiles_list: list[str], pH: float = 7.4, task_name: str = None) -> list[str]:
    """
    Predict dominant protonation state SMILES for a list of SMILES strings at a given pH using pKaLearn.
    Prints and logs a summary of how many molecules were protonated, unchanged, or failed.
    """
    #test
    if not smiles_list:
        return []

    total = len(smiles_list)
    num_changed = 0
    num_unchanged = 0
    num_failed = 0

    predict_fn = _get_pka_predict_fn()
    df_input = pd.DataFrame({'Smiles': smiles_list})

    final_smiles = []

    try:
        _, prot_smiles_list = predict_fn(df_input, mode='pH', pH=pH)
        
        # Pad list if pKaLearn returned fewer predictions than inputs
        if len(prot_smiles_list) < total:
            prot_smiles_list = list(prot_smiles_list) + [None] * (total - len(prot_smiles_list))

        for orig, prot in zip(smiles_list, prot_smiles_list):
            if prot and isinstance(prot, str) and prot.strip() != "":
                if prot != orig:
                    num_changed += 1
                else:
                    num_unchanged += 1
                final_smiles.append(prot)
            else:
                logger.warning(f"pKaLearn returned empty output for SMILES '{orig}'. Falling back to original.")
                num_failed += 1
                final_smiles.append(orig)

    except Exception as e:
        logger.warning(f"pKaLearn failed with exception: {e}. Falling back to original SMILES.")
        num_failed = total
        final_smiles = list(smiles_list)

    task_label = f" | {task_name}" if task_name else ""
    summary_msg = f"[pKaLearn pH {pH}{task_label}] Processed {total} mols | Changed: {num_changed} ({num_changed/total*100:.1f}%) | Unchanged: {num_unchanged} ({num_unchanged/total*100:.1f}%) | Fallback: {num_failed}"
    print(summary_msg)
    logger.info(summary_msg)

    return final_smiles

