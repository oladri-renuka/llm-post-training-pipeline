.PHONY: install sft reward ppo eval all clean

install:
	pip install -r requirements.txt

sft:
	python scripts/run_sft.py

reward:
	python scripts/run_reward.py

ppo:
	python scripts/run_ppo.py

eval:
	python scripts/run_eval.py

all:
	$(MAKE) sft
	$(MAKE) reward
	$(MAKE) ppo
	$(MAKE) eval

clean:
	rm -rf checkpoints/
	rm -rf results/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
