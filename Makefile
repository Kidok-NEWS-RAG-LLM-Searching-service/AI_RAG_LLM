run-local:
	ENV=local python3.12 -m uvicorn app.main:app --reload
run-dev:
	ENV=dev python3.12 -m uvicorn app.main:app --reload
run-prod:
	ENV=prod python3.12 -m uvicorn app.main:app --reload