.PHONY: test demo train eval

test:
	python -m pytest

demo:
	python -m cs_massive_access.demo

train:
	python -m cs_massive_access.train --N 32 --K 128 --Ka 8 --E 10 --L 10 --num-samples 20000 --epochs 20

eval:
	python -m cs_massive_access.evaluate --checkpoint runs/lista.pt --detect topk

