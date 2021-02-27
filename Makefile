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

run-training-on-cluster: prompt
	gcloud --configuration=$(PROJECT) ai-platform jobs submit training "$(MODEL_VERSION)" \
	--stream-logs \
  	--module-name trainer.task \
  	--package-path trainer/ \
  	--staging-bucket gs://$(PROJECT)-training/ \
  	--runtime-version=2.4 \
  	--python-version=3.7 \
  	--config cloudml-cluster-config.yaml \
  	-- \
  	--job-dir=gs://$(PROJECT)-model-output/$(MODEL_VERSION)
  	--num-epochs=20

run-training-on-single: prompt
	gcloud --configuration=$(PROJECT) ai-platform jobs submit training "$(MODEL_VERSION)" \
	--stream-logs \
  	--module-name trainer.task \
  	--package-path trainer/ \
  	--staging-bucket gs://$(PROJECT)-training/ \
  	--runtime-version=2.4 \
  	--python-version=3.7 \
  	--config cloudml-single-config.yaml \
  	-- \
  	--job-dir=gs://$(PROJECT)-model-output/$(MODEL_VERSION) \
  	--num-epochs=15

run-local-train: prompt
	gcloud ai-platform local train \
  	--module-name trainer.task \
  	--package-path trainer/ \
  	-- \
  	--job-dir=gs://retimo-model-output

