import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import logging
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_MB', 200)) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Убедитесь, что папка для загрузки существует
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Настройка ведения журнала логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Допустимые форматы файла
ALLOWED_EXTENSIONS = {'pdf', 'mp4', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Инициализация БД SQLite"""
    conn = sqlite3.connect('ai_study.db')
    c = conn.cursor()
    
    # Таблица с результатом
    c.execute('''
        CREATE TABLE IF NOT EXISTS result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            topics_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            flashcards_json TEXT NOT NULL,
            mind_map_json TEXT,
            study_plan_json TEXT,
            quality_json TEXT,
            video_segments_json TEXT,
            key_moments_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Таблица прогресса пользователя
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER,
            flashcard_id INTEGER,
            last_review TIMESTAMP,
            next_review TIMESTAMP,
            ease_factor REAL DEFAULT 2.5,
            consecutive_correct INTEGER DEFAULT 0,
            FOREIGN KEY (result_id) REFERENCES result(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_result(filename, file_type, analysis_result):
    """Сохранение результата в БД"""
    conn = sqlite3.connect('ai_study.db')
    c = conn.cursor()
    
    # Сериализовываем данные
    topics_json = json.dumps(analysis_result['topics_data'], ensure_ascii=False)
    flashcards_json = json.dumps(analysis_result['flashcards'], ensure_ascii=False)
    mind_map_json = json.dumps(analysis_result.get('mind_map', {}), ensure_ascii=False)
    study_plan_json = json.dumps(analysis_result.get('study_plan', {}), ensure_ascii=False)
    quality_json = json.dumps(analysis_result.get('quality_assessment', {}), ensure_ascii=False)
    video_segments_json = json.dumps(analysis_result.get('video_segments', []), ensure_ascii=False)
    key_moments_json = json.dumps(analysis_result.get('key_moments', []), ensure_ascii=False)
    
    c.execute('''
        INSERT INTO result (
            filename, file_type, topics_json, summary, flashcards_json,
            mind_map_json, study_plan_json, quality_json,
            video_segments_json, key_moments_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename, file_type, topics_json, analysis_result['summary'], 
        flashcards_json, mind_map_json, study_plan_json, quality_json,
        video_segments_json, key_moments_json
    ))
    
    result_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return result_id

def get_result(result_id):
    """Получение результата из базы данных"""
    conn = sqlite3.connect('ai_study.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT filename, file_type, topics_json, summary, flashcards_json,
               mind_map_json, study_plan_json, quality_json,
               video_segments_json, key_moments_json, created_at
        FROM result WHERE id = ?
    ''', (result_id,))
    
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'filename': row[0],
            'file_type': row[1],
            'topics_data': json.loads(row[2]),
            'summary': row[3],
            'flashcards': json.loads(row[4]),
            'mind_map': json.loads(row[5]),
            'study_plan': json.loads(row[6]),
            'quality_assessment': json.loads(row[7]),
            'video_segments': json.loads(row[8]),
            'key_moments': json.loads(row[9]),
            'created_at': row[10]
        }
    return None

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка и обработка файла"""
    try:
        # Проверка загружен ли файл
        if 'file' not in request.files:
            flash('Выберите файл', 'danger')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # Проверка пустой ли файл
        if file.filename == '':
            flash('Выберите файл', 'danger')
            return redirect(url_for('index'))
        
        # Проверка формата файла
        if not allowed_file(file.filename):
            flash('Формат не поддерживается. Используйте PDF, MP4, MOV или MKV', 'danger')
            return redirect(url_for('index'))
        
        # Сохранение файла
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        # Обработка файла
        try:
            from ml import process_file
            analysis_result = process_file(filepath, filename)
            
            # Сохранение результата в БД
            file_type = Path(filename).suffix.lower()
            result_id = save_result(filename, file_type, analysis_result)
            
            # Удаление файла
            os.remove(filepath)
            
            logger.info(f"Advanced processing completed for: {filename}")
            
            return redirect(url_for('result', result_id=result_id))
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            # Удаление файла с ошибкой
            if os.path.exists(filepath):
                os.remove(filepath)
            flash('Ошибка обработки, попробуйте ещё раз', 'danger')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        flash('Ошибка загрузки файла', 'danger')
        return redirect(url_for('index'))

@app.route('/result/<int:result_id>')
def result(result_id):
    """Отображение результата"""
    data = get_result(result_id)
    if not data:
        flash('Результат не найден', 'danger')
        return redirect(url_for('index'))
    
    return render_template('result.html', **data, result_id=result_id)

@app.route('/api/flashcard_progress', methods=['POST'])
def update_flashcard_progress():
    """Обновление прогресса изучения флеш-карт"""
    try:
        data = request.json
        result_id = data.get('result_id')
        flashcard_id = data.get('flashcard_id')
        correct = data.get('correct', False)
        
        conn = sqlite3.connect('ai_study.db')
        c = conn.cursor()
        
        # Проверка существования прогресса
        c.execute('''
            SELECT id, ease_factor, consecutive_correct 
            FROM user_progress 
            WHERE result_id = ? AND flashcard_id = ?
        ''', (result_id, flashcard_id))
        
        progress = c.fetchone()
        
        if progress:
            # Обновление существующего прогресса
            prog_id, ease_factor, consecutive = progress
            
            if correct:
                # Повышение сложности при правильном ответе
                new_ease = min(2.5, ease_factor + 0.1)
                new_consecutive = consecutive + 1
                interval_days = int(new_consecutive * new_ease)
            else:
                # Понижение сложности при неправильном ответе
                new_ease = max(1.3, ease_factor - 0.2)
                new_consecutive = 0
                interval_days = 1
            
            c.execute('''
                UPDATE user_progress 
                SET last_review = CURRENT_TIMESTAMP,
                    next_review = datetime('now', '+' || ? || ' days'),
                    ease_factor = ?,
                    consecutive_correct = ?
                WHERE id = ?
            ''', (interval_days, new_ease, new_consecutive, prog_id))
        else:
            # Создание новой истории прогресса
            interval_days = 1 if not correct else 3
            c.execute('''
                INSERT INTO user_progress 
                (result_id, flashcard_id, last_review, next_review, ease_factor, consecutive_correct)
                VALUES (?, ?, CURRENT_TIMESTAMP, datetime('now', '+' || ? || ' days'), 2.5, ?)
            ''', (result_id, flashcard_id, interval_days, 1 if correct else 0))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "next_review_days": interval_days})
        
    except Exception as e:
        logger.error(f"Error updating flashcard progress: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download/<int:result_id>')
def download_flashcards(result_id):
    """Сохранение флеш-карт как JSON"""
    data = get_result(result_id)
    if not data:
        flash('Результат не найден', 'danger')
        return redirect(url_for('index'))
    
    anki_cards = []
    for i, card in enumerate(data['flashcards']):
        anki_card = {
            "id": i + 1,
            "question": card['q'],
            "answer": card['a'],
            "tags": [card['type']] + card.get('related_topics', []),
            "hint": card.get('hint', ''),
            "memory_hook": card.get('memory_hook', ''),
            "common_mistakes": card.get('common_mistakes', ''),
            "difficulty": card.get('difficulty', 1)
        }
        anki_cards.append(anki_card)
    
    # Метадата
    export_data = {
        "deck_name": f"AI_Study_{data['filename']}",
        "created": datetime.now().isoformat(),
        "total_cards": len(anki_cards),
        "cards": anki_cards,
        "study_plan": data.get('study_plan', {}),
        "mind_map": data.get('mind_map', {})
    }
    
    # Создание JSON файла
    json_content = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    temp_filename = f"ai_study_export_{result_id}.json"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(json_content)
    
    def remove_file(response):
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return response
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f"ai_study_{datetime.now().strftime('%Y%m%d')}.json",
        mimetype='application/json'
    )

@app.route('/api/mind_map/<int:result_id>')
def get_mind_map_data(result_id):
    """Получение Mind Map"""
    data = get_result(result_id)
    if not data:
        return jsonify({"error": "Not found"}), 404
    
    return jsonify(data.get('mind_map', {}))

@app.route('/api/study_progress/<int:result_id>')
def get_study_progress(result_id):
    """Получение прогресса пользователя"""
    try:
        conn = sqlite3.connect('ai_study.db')
        c = conn.cursor()
        
        # Получение прогресса флеш-карт
        c.execute('''
            SELECT flashcard_id, last_review, next_review, 
                   ease_factor, consecutive_correct
            FROM user_progress
            WHERE result_id = ?
        ''', (result_id,))
        
        progress_data = []
        for row in c.fetchall():
            progress_data.append({
                "flashcard_id": row[0],
                "last_review": row[1],
                "next_review": row[2],
                "ease_factor": row[3],
                "consecutive_correct": row[4]
            })
        
        # Подсчет прогресса
        total_cards = len(get_result(result_id)['flashcards'])
        reviewed_cards = len(progress_data)
        mastered_cards = sum(1 for p in progress_data if p['consecutive_correct'] >= 3)
        
        conn.close()
        
        return jsonify({
            "total_cards": total_cards,
            "reviewed_cards": reviewed_cards,
            "mastered_cards": mastered_cards,
            "progress_percentage": round((mastered_cards / total_cards * 100) if total_cards > 0 else 0, 1),
            "card_progress": progress_data
        })
        
    except Exception as e:
        logger.error(f"Error getting study progress: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    """Превышен максимальный размер файла"""
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    flash(f'Размер файла превышает лимит в {max_mb} МБ', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)