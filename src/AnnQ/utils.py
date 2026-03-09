import json
import os
import pkg_resources

# 패키지 내부의 JSON 파일 경로 (data/ 서브폴더)
DATA_DIR = "data"
JSON_FILENAME = "AnnQ_Model_Hierarchy.json"

def load_hierarchy_db():
    """패키지 내 JSON DB를 로드합니다."""
    json_relative_path = os.path.join(DATA_DIR, JSON_FILENAME)
    
    # 1. 패키지 설치 경로에서 찾기 (배포용)
    try:
        json_path = pkg_resources.resource_filename('AnnQ', json_relative_path)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    
    # 2. 현재 작업 경로(개발 중)에서 찾기
    local_path = os.path.join(os.path.dirname(__file__), DATA_DIR, JSON_FILENAME)
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    # 3. 파일이 없으면 빈 딕셔너리 반환 (에러 방지)
    print(f"Warning: '{json_relative_path}' not found. Classification will rely on defaults.")
    return {}


def load_custom_hierarchy(hierarchy_file, model_list=None):
    """
    사용자 지정 JSON hierarchy 파일을 로드합니다.
    
    Args:
        hierarchy_file (str): JSON 파일 경로
        model_list (list, optional): 사용할 모델 이름 리스트.
                                     None이면 모든 모델을 병합.
    
    Returns:
        dict: 병합된 계층 정보 딕셔너리 { "CellType": {"Tissue":..., "Cell_Class":...} }
    """
    if not os.path.exists(hierarchy_file):
        print(f"Error: Hierarchy file '{hierarchy_file}' not found.")
        return {}
    
    try:
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            db = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{hierarchy_file}': {e}")
        return {}
    
    combined_map = {}
    
    # model_list가 지정되지 않으면 모든 모델 사용
    if model_list is None or len(model_list) == 0:
        # _metadata 키는 제외하고 모든 모델 병합
        for model_name, model_data in db.items():
            if model_name.startswith('_'):  # _metadata 등 메타 정보 스킵
                continue
            if isinstance(model_data, dict):
                # 중첩 구조인 경우 (모델별로 분리된 경우)
                # 각 cell type의 정보가 dict인지 확인
                first_value = next(iter(model_data.values()), None)
                if isinstance(first_value, dict) and 'Tissue' in first_value:
                    combined_map.update(model_data)
                elif isinstance(first_value, dict):
                    # 더 깊은 중첩 (모델 > cell type > info)
                    for cell_type, cell_info in model_data.items():
                        if not cell_type.startswith('_'):
                            combined_map[cell_type] = cell_info
        
        # 만약 combined_map이 비어있다면, flat 구조일 수 있음
        if not combined_map:
            for key, value in db.items():
                if not key.startswith('_') and isinstance(value, dict):
                    if 'Tissue' in value and 'Cell_Class' in value:
                        combined_map[key] = value
    else:
        # model_list가 지정된 경우
        if isinstance(model_list, str):
            model_list = [model_list]
        
        for model_name in model_list:
            if model_name in db:
                model_data = db[model_name]
                # _metadata 등 메타 정보 제외
                for cell_type, cell_info in model_data.items():
                    if not cell_type.startswith('_'):
                        combined_map[cell_type] = cell_info
            else:
                print(f"Warning: Model '{model_name}' not found in hierarchy file.")
    
    if combined_map:
        print(f"Loaded {len(combined_map)} cell types from custom hierarchy.")
    else:
        print("Warning: No cell types loaded from hierarchy file.")
    
    return combined_map


def get_combined_hierarchy(model_list, hierarchy_file=None):
    """
    여러 모델의 계층 정보를 하나로 병합하여 반환합니다.
    
    Args:
        model_list (list): 모델 이름들의 리스트 
                           예: ["Mouse_Whole_Brain.pkl", "Developing_Mouse_Brain.pkl"]
        hierarchy_file (str, optional): 사용자 지정 JSON 파일 경로.
                                        지정하면 내장 DB 대신 이 파일 사용.
    
    Returns:
        dict: 병합된 계층 정보 딕셔너리 { "CellType": {"Tissue":..., "Cell_Class":...} }
    """
    # Custom hierarchy file이 지정된 경우
    if hierarchy_file:
        return load_custom_hierarchy(hierarchy_file, model_list)
    
    # 기본: 내장 DB 사용
    db = load_hierarchy_db()
    combined_map = {}
    
    if not model_list:
        return {}

    # 단일 문자열이 들어온 경우 리스트로 변환 (안전장치)
    if isinstance(model_list, str):
        model_list = [model_list]

    for model_name in model_list:
        if model_name in db:
            # 해당 모델의 딕셔너리(CellType -> Info)를 가져와서 병합
            # 여러 모델에 같은 이름의 세포가 있어도 Universal Logic에 의해 속성이 같으므로 덮어써도 무방
            combined_map.update(db[model_name])
        else:
            print(f"Warning: Model '{model_name}' not found in JSON DB.")
            
    return combined_map
