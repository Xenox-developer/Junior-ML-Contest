# Docker Setup для AI Study
## Сервис доступен по ссылке http://нотэ.рф/

## 🚀 Быстрый старт

### 1. Предварительные требования
- Docker
- Docker Compose

### 2. Подготовка
```bash
# Клонирование репозитория
git clone https://github.com/Xenox-developer/Junior-ML-Contest.git
cd Junior-ML-Contest

# Создание .env файла
cp .env.example .env
# Отредактируйте .env и добавьте ваш OPENAI_API_KEY

# Сделать скрипты исполняемыми
chmod +x setup.sh run.sh
```

### 3. Сборка и запуск

#### Вариант 1: CPU версия
```bash
# Сборка
docker-compose build

# Запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f app
```

#### Вариант 2: GPU версия
```bash
# Сборка и запуск
docker-compose -f docker-compose.gpu.yml up -d
```

#### Вариант 3: Режим разработки
```bash
# Запуск с hot-reload
docker-compose -f docker-compose.dev.yml up
```

#### Вариант 4: Production с Nginx
```bash
# Запуск с Nginx reverse proxy
docker-compose --profile production up -d
```

## 🔧 Конфигурация

### Переменные окружения (.env)
```env
# Обязательные
OPENAI_API_KEY=sk-...

# Опциональные
MAX_UPLOAD_MB=200
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

## 🛠️ Управление

### Основные команды
```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Просмотр логов
docker-compose logs -f app

# Вход в контейнер
docker-compose exec app bash

# Очистка volumes
docker-compose down -v
```

### Обновление
```bash
# Получить последние изменения
git pull

# Пересобрать образы
docker-compose build --no-cache

# Перезапустить
docker-compose up -d
```

## 🔍 Отладка

### Проверка статуса
```bash
# Статус контейнеров
docker-compose ps

# Использование ресурсов
docker stats

# Проверка состояния
docker-compose exec app curl http://localhost:5000/
```