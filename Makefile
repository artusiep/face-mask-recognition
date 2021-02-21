REGION=europe-west3
PROJECT=retimo
MODEL_NAME=retimo
MODEL_VERSION=$(MODEL_NAME)_1_$(shell date +%s)

prompt:
	echo "Using model version: $(MODEL_VERSION)"

init:
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt

run-local:
	docker run -d -p 8080:8080 -v ~/Developer/Private/face-mask-recognition/jupyter:/home/jupyter gcr.io/deeplearning-platform-release/tf-cpu.1-13

run-training: prompt
	gcloud ai-platform jobs submit training "$(MODEL_VERSION)" \
	--stream-logs \
  	--module-name trainer.task \
  	--package-path trainer/ \
  	--staging-bucket "gs://$(PROJECT)-training/" \
  	--runtime-version=2.3 \
  	--python-version=3.7 \
  	--config cloudml-config.yaml \
  	-- \
  	--job-dir=gs://retimo-training