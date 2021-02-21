from urllib.parse import urlparse
from google.cloud import storage


def mark_done(gspath):
    """Uploads a file to the bucket to indicate comletion of training job.
    gspath is a path to the output directory of training such as

    gs://$PROJECT-model-output/$MODEL_NAME/$MODEL_VERSION/output
    """
    url = urlparse(gspath)
    if url.scheme != "gs":
        raise RuntimeError("not a Google Storage URL")
    bucket_name = url.netloc
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(url.path.strip("/") + "/TRAINER-DONE")
    blob.upload_from_string("done")

