run-local:
	docker run -d -p 8080:8080 -v ~/Developer/Private/face-mask-recognition/jupyter:/home/jupyter gcr.io/deeplearning-platform-release/tf-cpu.1-13

