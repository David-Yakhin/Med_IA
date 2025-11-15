# create_symptoms_dataset.py
import json
from pathlib import Path

def create_symptoms_focused_dataset():
    """Создает датасет, оптимизированный для поиска по симптомам"""
    
    input_dir = Path("enhanced_dataset")
    output_dir = Path("symptoms_dataset")
    output_dir.mkdir(exist_ok=True)
    
    symptoms_data = []
    
    for json_file in input_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Создаем упрощенную структуру для поиска по симптомам
            simplified = {
                'title': data['title'],
                'main_symptoms': data['metadata'].get('symptoms', []),
                'categories': data['metadata'].get('categories', []),
                'clinical_picture': data['sections'].get('clinical_picture', ''),
                'complaints_anamnesis': data['sections'].get('complaints_anamnesis', ''),
                'diagnostics': data['sections'].get('diagnostics', ''),
                'full_text': ' '.join([data['title']] + 
                           [str(content) for content in data['sections'].values()])
            }
            
            # Сохраняем
            output_file = output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simplified, f, ensure_ascii=False, indent=2)
                
            symptoms_data.append(simplified)
            
        except Exception as e:
            print(f"Ошибка в {json_file.name}: {e}")
    
    # Создаем индекс симптомов
    create_symptoms_index(symptoms_data)
    
    print(f"Создан датасет для поиска симптомов: {len(symptoms_data)} заболеваний")

def create_symptoms_index(symptoms_data):
    """Создает индекс симптомов для быстрого поиска"""
    
    symptoms_index = {}
    
    for disease in symptoms_data:
        title = disease['title']
        symptoms = disease['main_symptoms']
        categories = disease['categories']
        
        for symptom in symptoms:
            if symptom not in symptoms_index:
                symptoms_index[symptom] = []
            symptoms_index[symptom].append({
                'title': title,
                'categories': categories
            })
    
    # Сохраняем индекс
    with open('symptoms_index.json', 'w', encoding='utf-8') as f:
        json.dump(symptoms_index, f, ensure_ascii=False, indent=2, ensure_ascii=False)
    
    print(f"Создан индекс для {len(symptoms_index)} симптомов")

if __name__ == "__main__":
    create_symptoms_focused_dataset()