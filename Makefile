.PHONY: test smoke-test docker-build docker-run run dashboard clean

test:
	pytest tests/ -v

smoke-test:
	python run.py --bbox 85.2,27.9,85.4,28.1 --date 2026-08-26 --report

dashboard:
	streamlit run app/dashboard/app.py

docker-build:
	docker build -t spacetrack-flood:latest .

docker-run:
	docker run --rm -it -p 8501:8501 spacetrack-flood:latest streamlit run app/dashboard/app.py

clean:
	rm -rf .pytest_cache outputs/ __pycache__ */__pycache__
