.PHONY: install bootstrap run test evaluate benchmark docker up
install:   ; pip install -r requirements-dev.txt
bootstrap: ; python -m app.cli bootstrap
run:       ; uvicorn app.main:app --reload --port 8000
test:      ; pytest -q
evaluate:  ; python -m app.cli evaluate
benchmark: ; python -m app.cli benchmark
docker:    ; docker build -t movie-recommender . && docker run -p 8000:8000 movie-recommender
up:        ; docker compose up --build
