view:
	python3 mnist_view_data.py $(record)

train:
	python3 mnist_train.py

test:
	python3 mnist_test.py

infere:
	python3 mnist_infere.py $(record)