.PHONY: install sft reward ppo dpo eval all clean

install:
	pip install -r requirements.txt

sft:
	python scripts/run_sft.py

reward:
	python scripts/run_reward.py

ppo:
	python scripts/run_ppo.py

dpo:
	python scripts/run_dpo.py

eval:
	python scripts/run_eval.py

all:
	$(MAKE) sft
	$(MAKE) reward
	$(MAKE) dpo
	$(MAKE) eval

clean:
	rm -rf checkpoints/
	rm -rf results/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
