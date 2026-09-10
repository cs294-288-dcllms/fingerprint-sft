.PHONY: env data teacher control adfp8 adfp16 adfp32 radioactive all test

env:
	./scripts/bootstrap_conda.sh

data:
	./scripts/prepare_science.sh

teacher:
	./scripts/run_teacher_sft.sh

control:
	./scripts/run_condition.sh configs/strategies/control.env

adfp8:
	./scripts/run_condition.sh configs/strategies/adfp-lambda8.env

adfp16:
	./scripts/run_condition.sh configs/strategies/adfp-lambda16.env

adfp32:
	./scripts/run_condition.sh configs/strategies/adfp-lambda32.env

radioactive:
	./scripts/run_condition.sh configs/strategies/radioactive-delta2.env

all:
	./scripts/run_all_sft.sh

test:
	PYTHONPATH=. /tmp/fingerprint-sft-conda/bin/python -m unittest discover -s tests -v
