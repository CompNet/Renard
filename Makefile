.PHONY: test
test:
	uv run python -m pytest tests

.PHONE: ui
ui:
	uv run python -m renard.ui
