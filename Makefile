coverage:
	coverage run -m pytest
	coverage report
	coverage json

test:
	pytest -vv