import json
import os
import celltypist
from celltypist import models

# =============================================================================
# 1. 범용 세포 클래스 추론 (Universal Logic)
# =============================================================================
def get_universal_class(cell_name):
    """
    세포 이름의 키워드를 분석하여 범용적인 Cell Class를 반환합니다.
    """
    c_lower = cell_name.lower()
    
    # [0] Doublet / Low Quality
    if 'doublet' in c_lower: return "Doublet"
    if 'low quality' in c_lower or 'debris' in c_lower: return "LowQuality"

    # [1] Immune
    if any(x in c_lower for x in ['b cell', 't cell', 'nk', 'plasma', 'lymphocyte', 'cd4', 'cd8', 'treg', 'thymocyte', 'innate lymphoid', 'ilc']):
        return "Immune_Lymphoid"
    if any(x in c_lower for x in ['macrophage', 'monocyte', 'dendritic', 'dc', 'neutrophil', 'mast', 'myeloid', 'kupffer', 'microglia', 'granulocyte', 'basophil', 'eosinophil']):
        return "Immune_Myeloid"
    if "immune" in c_lower or "leukocyte" in c_lower or "lymphoid" in c_lower:
        return "Immune_General"

    # [2] Vascular
    if any(x in c_lower for x in ['endothelial', 'endothelium', 'endo', 'vascular', 've_']):
        return "Endothelial"
    
    # [3] Stromal / Mural
    if any(x in c_lower for x in ['fibroblast', 'mesenchyme', 'stromal', 'pericyte', 'smooth muscle', 'smc', 'myofibroblast', 'mural', 'peri ', 'vlmc', 'abc ']):
        return "Stromal"
    
    # [4] Neural
    if any(x in c_lower for x in ['neuron', 'glut', 'gaba', 'dopa', 'excitatory', 'inhibitory', 'purkinje', 'granule', 'motor', 'neuroblast']):
        return "Neuron"
    if any(x in c_lower for x in ['astrocyte', 'oligodendrocyte', 'opc', 'glia', 'schwann', 'ependymal', 'tanycyte', 'satellite']):
        return "Glia"
        
    # [5] Epithelial
    if any(x in c_lower for x in ['epithelial', 'basal', 'ciliated', 'club', 'goblet', 'pneumocyte', 'keratinocyte', 'podocyte', 'tubule', 'loop of henle', 'duct', 'hepatocyte', 'cholangiocyte', 'mesothelial']):
        return "Epithelial"
    
    # [6] Erythroid
    if "erythro" in c_lower or "red blood" in c_lower or "hemoglobin" in c_lower:
        return "Erythroid"
        
    # [7] Progenitor
    if any(x in c_lower for x in ['stem', 'progenitor', 'precursor', 'blast']):
        return "Progenitor"
        
    # [8] Cycling
    if 'cycling' in c_lower or 'proliferating' in c_lower:
        return "Cycling"

    return "Parenchymal_or_Other"

# =============================================================================
# 2. 모델 기반 Tissue 결정
# =============================================================================
def get_model_tissue(model_name):
    m_lower = model_name.lower()
    
    if "brain" in m_lower or "hippocampus" in m_lower or "cortex" in m_lower or "dentate" in m_lower: return "Brain"
    if "lung" in m_lower or "airway" in m_lower: return "Lung"
    if "kidney" in m_lower: return "Kidney"
    if "liver" in m_lower: return "Liver"
    if "heart" in m_lower: return "Heart"
    if "intestine" in m_lower or "gut" in m_lower or "colon" in m_lower or "rectal" in m_lower: return "Intestine"
    if "skin" in m_lower: return "Skin"
    if "pancreas" in m_lower: return "Pancreas"
    if "blood" in m_lower or "pbmc" in m_lower: return "Blood"
    if "spleen" in m_lower: return "Spleen"
    if "spinal" in m_lower: return "SpinalCord"
    if "retina" in m_lower or "eye" in m_lower or "ear" in m_lower: return "SensoryOrgans"
    if "placenta" in m_lower or "yolksac" in m_lower or "gonad" in m_lower: return "Reproductive"
    if "thymus" in m_lower: return "Thymus"
    if "tonsil" in m_lower: return "Tonsil"
    if "immune_all" in m_lower: return "Immune_System"
    
    return model_name.replace(".pkl", "")

# =============================================================================
# 3. 메인 실행 함수
# =============================================================================
def main():
    # 저장 경로 설정 (src/AnnQ/data 폴더)
    target_dir = os.path.join("src", "AnnQ", "data")
    
    # 폴더가 존재하는지 확인, 없으면 생성
    if not os.path.exists(os.path.join("src", "AnnQ")):
        print(f"Error: 'src/AnnQ' directory does not exist.")
        print("Please run this script from the project root directory (where 'src' folder is located).")
        return
    
    # data 폴더 생성
    os.makedirs(target_dir, exist_ok=True)

    print("Fetching all available models from CellTypist...")
    
    try:
        all_model_names = models.get_all_models()
    except Exception as e:
        print(f"Error fetching model list: {e}")
        return

    print(f"Found {len(all_model_names)} models. Generating hierarchy...")
    
    hierarchy = {}
    total_cells = 0
    
    for model_name in all_model_names:
        try:
            # print(f"Processing {model_name}...")
            current_model = models.Model.load(model=model_name)
            cell_types = current_model.cell_types
            
            default_tissue = get_model_tissue(model_name)
            hierarchy[model_name] = {}
            
            for cell in cell_types:
                cell_class = get_universal_class(cell)
                hierarchy[model_name][cell] = {
                    "Tissue": default_tissue,
                    "Cell_Class": cell_class
                }
                total_cells += 1
                
        except Exception as e:
            print(f"  Warning: Failed to load {model_name} - {e}")
            
    # JSON 파일 저장 (src/AnnQ/data/AnnQ_Model_Hierarchy.json)
    output_filename = "AnnQ_Model_Hierarchy.json"
    output_path = os.path.join(target_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print(f"Success! Processed {len(hierarchy)} models.")
    print(f"JSON database saved directly to: {output_path}")
    print("-" * 50)
    print("You can now run 'annq' immediately.")

if __name__ == "__main__":
    main()