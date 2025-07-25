#!/usr/bin/env python3
"""
Скрипт для миграции существующих результатов - генерация тестовых вопросов
"""

import sqlite3
import json
import os
import sys
from datetime import datetime

# Добавляем путь к модулям приложения
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Загружаем переменные окружения из .env файла
from dotenv import load_dotenv
load_dotenv()

def generate_test_questions(result_data):
    """Генерирует тестовые вопросы с вариантами ответов на основе материала"""
    try:
        # Проверяем наличие API ключа
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("⚠️ OpenAI API ключ не найден, используем демонстрационные вопросы")
            return get_demo_questions()
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Получаем текст материала
        full_text = result_data.get('full_text', '')
        summary = result_data.get('summary', '')
        topics_data = result_data.get('topics_data', {})
        
        # Формируем расширенный контекст для генерации вопросов
        # Берем больше текста для лучшего понимания материала
        text_sample = full_text[:5000] if len(full_text) > 5000 else full_text
        
        # Извлекаем ключевые темы и подтемы
        main_topics = []
        if isinstance(topics_data, dict):
            for topic, details in topics_data.items():
                if isinstance(details, dict) and 'subtopics' in details:
                    subtopics = details['subtopics'][:3]  # Берем первые 3 подтемы
                    main_topics.append(f"{topic}: {', '.join(subtopics)}")
                else:
                    main_topics.append(str(topic))
        
        context = f"""
        НАЗВАНИЕ МАТЕРИАЛА: {result_data.get('filename', 'Учебный материал')}
        
        КРАТКОЕ РЕЗЮМЕ:
        {summary}
        
        ОСНОВНЫЕ ТЕМЫ И ПОДТЕМЫ:
        {chr(10).join(main_topics) if main_topics else 'Темы не определены'}
        
        ФРАГМЕНТ ПОЛНОГО ТЕКСТА (для понимания стиля и деталей):
        {text_sample}
        {'...' if len(full_text) > 5000 else ''}
        
        ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:
        - Общий объем материала: {len(full_text)} символов
        - Количество основных тем: {len(main_topics)}
        """
        
        prompt = f"""
        На основе КОНКРЕТНОГО предоставленного учебного материала создай 25 тестовых вопросов разной сложности.
        
        ВАЖНО: Вопросы должны быть СТРОГО основаны на содержании данного материала, а не на общих знаниях по теме.
        
        Материал:
        {context}
        
        Требования к вопросам:
        1. 8 легких вопросов (конкретные определения и факты из материала)
        2. 12 средних вопросов (понимание концепций, формул, алгоритмов из материала)
        3. 5 сложных вопросов (анализ примеров, применение методов из материала)
        
        ОБЯЗАТЕЛЬНЫЕ требования:
        - Вопросы должны проверять знание ИМЕННО этого материала
        - Используй конкретные термины, формулы, примеры из текста
        - НЕ задавай общие вопросы по теме, которые можно ответить без чтения материала
        - Включай специфические детали, числа, названия из материала
        - Ссылайся на конкретные разделы, примеры, диаграммы из текста
        
        Каждый вопрос должен иметь:
        - Четкую формулировку на русском языке, основанную на материале
        - 4 варианта ответа (A, B, C, D) - все правдоподобные для данной темы
        - Только один правильный ответ из материала
        - Подробное объяснение со ссылкой на материал
        - Неправильные варианты должны быть близкими по теме, но четко неверными
        
        Примеры хороших вопросов:
        - "Согласно материалу, какая формула используется для расчета..."
        - "В приведенном примере автор демонстрирует..."
        - "Какой метод рекомендуется в материале для решения..."
        - "Согласно тексту, основное отличие между X и Y заключается в..."
        
        Верни результат в формате JSON:
        {{
            "questions": [
                {{
                    "id": 1,
                    "question": "Конкретный вопрос по материалу",
                    "options": {{
                        "A": "Вариант A из материала",
                        "B": "Вариант B из материала", 
                        "C": "Вариант C из материала",
                        "D": "Вариант D из материала"
                    }},
                    "correct_answer": "A",
                    "explanation": "Объяснение со ссылкой на конкретное место в материале",
                    "difficulty": 1,
                    "topic": "Конкретная тема из материала"
                }}
            ]
        }}
        
        Сложность: 1 = легко, 2 = средне, 3 = сложно
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты эксперт по созданию образовательных тестов, специализирующийся на создании вопросов строго по содержанию конкретного учебного материала. Твоя задача - создавать вопросы, которые можно ответить ТОЛЬКО прочитав данный материал, а не на основе общих знаний по теме. Фокусируйся на специфических деталях, примерах, формулах и концепциях из предоставленного текста."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Снижаем температуру для более точных вопросов
            max_tokens=4000
        )
        
        # Парсим ответ
        response_text = response.choices[0].message.content
        
        # Извлекаем JSON из ответа
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            try:
                questions_data = json.loads(json_text)
                return questions_data.get('questions', [])
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON: {e}")
                print(f"Пытаемся исправить JSON...")
                
                # Пытаемся исправить распространенные ошибки JSON
                try:
                    # Исправляем отсутствующие запятые между объектами
                    fixed_json = re.sub(r'}\s*{', '},{', json_text)
                    # Исправляем отсутствующие запятые после строк
                    fixed_json = re.sub(r'"\s*\n\s*"', '",\n"', fixed_json)
                    # Исправляем отсутствующие запятые после чисел
                    fixed_json = re.sub(r'(\d)\s*\n\s*"', r'\1,\n"', fixed_json)
                    
                    questions_data = json.loads(fixed_json)
                    print("✅ JSON успешно исправлен")
                    return questions_data.get('questions', [])
                except json.JSONDecodeError:
                    print("❌ Не удалось исправить JSON, используем демонстрационные вопросы")
                    return get_demo_questions()
        else:
            print("Не удалось извлечь JSON из ответа GPT")
            print(f"Ответ GPT: {response_text[:500]}...")
            return get_demo_questions()
            
    except Exception as e:
        print(f"Ошибка генерации тестовых вопросов: {str(e)}")
        return get_demo_questions()

def get_demo_questions():
    """Возвращает демонстрационные вопросы для тестирования"""
    return [
        {
            "id": 1,
            "question": "Что такое машинное обучение?",
            "options": {
                "A": "Способность машин физически обучаться новым движениям",
                "B": "Раздел ИИ, позволяющий компьютерам обучаться на данных",
                "C": "Процесс обучения людей работе с машинами",
                "D": "Автоматическое обновление программного обеспечения"
            },
            "correct_answer": "B",
            "explanation": "Машинное обучение — это раздел искусственного интеллекта, который позволяет компьютерам обучаться и принимать решения на основе данных без явного программирования.",
            "difficulty": 1,
            "topic": "Основы ML"
        },
        {
            "id": 2,
            "question": "Какие основные типы машинного обучения существуют?",
            "options": {
                "A": "Быстрое, медленное и среднее обучение",
                "B": "Обучение с учителем, без учителя и с подкреплением",
                "C": "Линейное, нелинейное и циклическое обучение",
                "D": "Простое, сложное и экспертное обучение"
            },
            "correct_answer": "B",
            "explanation": "Основные типы: supervised learning (с учителем), unsupervised learning (без учителя) и reinforcement learning (с подкреплением).",
            "difficulty": 2,
            "topic": "Типы обучения"
        },
        {
            "id": 3,
            "question": "Что происходит при переобучении модели?",
            "options": {
                "A": "Модель работает слишком быстро",
                "B": "Модель потребляет много памяти",
                "C": "Модель слишком хорошо запоминает обучающие данные",
                "D": "Модель обучается дольше обычного"
            },
            "correct_answer": "C",
            "explanation": "При переобучении модель слишком хорошо запоминает обучающие данные, включая шум, что ухудшает её работу на новых данных.",
            "difficulty": 2,
            "topic": "Проблемы обучения"
        },
        {
            "id": 4,
            "question": "Что такое нейронная сеть?",
            "options": {
                "A": "Сеть компьютеров для обработки данных",
                "B": "Математическая модель, имитирующая работу нейронов мозга",
                "C": "Программа для создания графиков",
                "D": "База данных для хранения информации"
            },
            "correct_answer": "B",
            "explanation": "Нейронная сеть — это математическая модель, построенная по принципу организации и функционирования биологических нейронных сетей.",
            "difficulty": 1,
            "topic": "Нейронные сети"
        },
        {
            "id": 5,
            "question": "Что такое градиентный спуск?",
            "options": {
                "A": "Метод физических упражнений",
                "B": "Алгоритм оптимизации для минимизации функции потерь",
                "C": "Способ сжатия данных",
                "D": "Техника визуализации данных"
            },
            "correct_answer": "B",
            "explanation": "Градиентный спуск — это итерационный алгоритм оптимизации, используемый для минимизации функции потерь путем движения в направлении наибольшего убывания градиента.",
            "difficulty": 2,
            "topic": "Оптимизация"
        }
    ]

def migrate_test_questions():
    """Миграция существующих результатов - добавление тестовых вопросов"""
    print("🔄 Начинаем миграцию тестовых вопросов...")
    
    conn = sqlite3.connect('ai_study.db')
    c = conn.cursor()
    
    # Находим результаты без тестовых вопросов
    c.execute('''
        SELECT id, filename, topics_json, summary, full_text
        FROM result 
        WHERE test_questions_json IS NULL OR test_questions_json = ''
    ''')
    
    results_to_migrate = c.fetchall()
    total_results = len(results_to_migrate)
    
    if total_results == 0:
        print("✅ Все результаты уже имеют тестовые вопросы")
        conn.close()
        return
    
    print(f"📊 Найдено {total_results} результатов для миграции")
    
    success_count = 0
    error_count = 0
    
    for i, (result_id, filename, topics_json, summary, full_text) in enumerate(results_to_migrate, 1):
        print(f"🔄 [{i}/{total_results}] Обрабатываем: {filename}")
        
        try:
            # Проверяем данные
            print(f"   📊 ID: {result_id}, topics_json: {bool(topics_json)}, summary: {bool(summary)}, full_text: {bool(full_text)}")
            
            if topics_json:
                try:
                    topics_data = json.loads(topics_json)
                    print(f"   ✅ Topics JSON успешно распарсен")
                except json.JSONDecodeError as e:
                    print(f"   ❌ Ошибка парсинга topics_json: {e}")
                    topics_data = {}
            else:
                topics_data = {}
            
            # Подготавливаем данные для генерации вопросов
            result_data = {
                'full_text': full_text or '',
                'summary': summary or '',
                'topics_data': topics_data
            }
            
            print(f"   📝 Данные для генерации: текст={len(result_data['full_text'])} символов, резюме={len(result_data['summary'])} символов")
            
            # Генерируем тестовые вопросы
            test_questions = generate_test_questions(result_data)
            
            if test_questions:
                # Сохраняем в базу данных
                test_questions_json = json.dumps(test_questions, ensure_ascii=False)
                c.execute('UPDATE result SET test_questions_json = ? WHERE id = ?', 
                         (test_questions_json, result_id))
                
                print(f"✅ [{i}/{total_results}] Сгенерировано {len(test_questions)} вопросов для {filename}")
                success_count += 1
            else:
                print(f"❌ [{i}/{total_results}] Не удалось сгенерировать вопросы для {filename}")
                error_count += 1
                
        except Exception as e:
            print(f"❌ [{i}/{total_results}] Ошибка при обработке {filename}: {str(e)}")
            error_count += 1
    
    # Сохраняем изменения
    conn.commit()
    conn.close()
    
    print(f"\n📈 Миграция завершена:")
    print(f"✅ Успешно обработано: {success_count}")
    print(f"❌ Ошибок: {error_count}")
    print(f"📊 Всего: {total_results}")

if __name__ == '__main__':
    migrate_test_questions()