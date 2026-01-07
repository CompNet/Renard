.PHONY: test
test:
	uv run --group dev python -m pytest tests

.PHONE: ui
ui:
	uv run --extra ui python -m renard.ui
