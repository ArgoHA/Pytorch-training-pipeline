.PHONY: full train preprocess split export bench infer vis

full:
	python -m src.etl.preprocess
	python -m src.etl.split
	python -m src.dl.train
	python -m src.dl.export
	python -m src.dl.bench

preprocess:
	python -m src.etl.preprocess

split:
	python -m src.etl.split

train:
	python -m src.dl.train

export:
	python -m src.dl.export

bench:
	python -m src.dl.bench

infer:
	python -m src.dl.infer

vis:
	python -m src.dl.vis

tf_export:
	python -m src.dl.tf_export
