.PHONY: help build run stop logs shell clean test lint

# Цвета для вывода
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
RESET  := $(shell tput -Txterm sgr0)

## Помощь
help:
	@echo '${GREEN}AI Study MVP - Docker Commands${RESET}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<command>${RESET}'
	@echo ''
	@echo 'Commands:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-15s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

## Сборка Docker образа
build:
	docker-compose build

## Сборка с GPU поддержкой
build-gpu:
	docker-compose -f docker-compose.gpu.yml build

## Запуск приложения
run:
	docker-compose up -d
	@echo "${GREEN}Application started at http://localhost:5000${RESET}"

## Запуск в режиме разработки
dev:
	docker-compose -f docker-compose.dev.yml up

## Запуск с GPU
run-gpu:
	docker-compose -f docker-compose.gpu.yml up -d

## Остановка приложения
stop:
	docker-compose down

## Просмотр логов
logs:
	docker-compose logs -f app

## Вход в контейнер
shell:
	docker-compose exec app bash

## Очистка (контейнеры, образы, volumes)
clean:
	docker-compose down -v
	docker system prune -f

## Полная очистка (включая кэш моделей)
clean-all:
	docker-compose down -v
	docker system prune -af
	rm -rf uploads/* data/*.db

## Проверка статуса
status:
	docker-compose ps
	@echo ""
	@echo "Memory usage:"
	docker stats --no-stream

## Запуск тестов
test:
	docker-compose exec app pytest

## Проверка кода
lint:
	docker-compose exec app flake8 .
	docker-compose exec app black --check .

## Форматирование кода
format:
	docker-compose exec app black .

## Резервное копирование данных
backup:
	mkdir -p backups
	docker-compose exec app tar -czf /tmp/backup.tar.gz /app/data /app/uploads
	docker cp $$(docker-compose ps -q app):/tmp/backup.tar.gz ./backups/backup-$$(date +%Y%m%d-%H%M%S).tar.gz
	@echo "${GREEN}Backup created in ./backups/${RESET}"

## Восстановление из резервной копии
restore:
	@echo "Available backups:"
	@ls -la backups/
	@echo ""
	@read -p "Enter backup filename: " backup; \
	docker cp ./backups/$$backup $$(docker-compose ps -q app):/tmp/backup.tar.gz && \
	docker-compose exec app tar -xzf /tmp/backup.tar.gz -C /
	@echo "${GREEN}Restore completed${RESET}"

## Обновление зависимостей
update:
	docker-compose exec app pip install --upgrade -r requirements.txt

## Мониторинг производительности
monitor:
	watch -n 2 docker stats --no-stream

## Проверка здоровья
health:
	@docker-compose exec app curl -f http://localhost:5000/ > /dev/null 2>&1 && \
	echo "${GREEN}✓ Application is healthy${RESET}" || \
	echo "${YELLOW}✗ Application is not responding${RESET}"

## Установка pre-commit hooks
install-hooks:
	pip install pre-commit
	pre-commit install

## Запуск production версии с Nginx
prod:
	docker-compose --profile production up -d
	@echo "${GREEN}Production deployment started${RESET}"
	@echo "Nginx: http://localhost"
	@echo "App: http://localhost:5000"