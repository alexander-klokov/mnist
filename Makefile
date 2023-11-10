SRC='src'

view:
	python3 ${SRC}/mnist_view_data.py $(record)

train:
	python3 ${SRC}/mnist_train.py

test:
	python3 ${SRC}/mnist_test.py

infere:
	python3 ${SRC}/mnist_infere.py $(record)