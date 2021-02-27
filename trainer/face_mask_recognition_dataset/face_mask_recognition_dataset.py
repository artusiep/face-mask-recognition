"""face_mask_recognition_dataset dataset."""
import os
import re

import tensorflow_datasets as tfds

# TODO(face_mask_recognition_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(face_mask_recognition_dataset): BibTeX citation
_CITATION = """
"""

_TRAIN_URL = "https://storage.googleapis.com/face-mask-recognition-dataset/train.zip"
_TEST_URL = "https://storage.googleapis.com/face-mask-recognition-dataset/test.zip"

_IMAGE_SIZE = 256
_IMAGE_SHAPE = (_IMAGE_SIZE, _IMAGE_SIZE, 3)

_FEATURES = ['correct', 'incorrect', 'no_mask']

_NAME_RE = re.compile(
    r"^(train|test)(?:/|\\)(correct|incorrect|no_mask)(?:/|\\)[\w-]*\.(jpg|png)$"
)


class FaceMaskRecognitionDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for face_mask_recognition_dataset face_mask_recognition_dataset."""

    VERSION = tfds.core.Version('1.0.5')
    RELEASE_NOTES = {
        '1.0.1': 'Test Version Bump to check checksum update.',
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the face_mask_recognition_dataset metadata."""
        # TODO(face_mask_recognition_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(names=_FEATURES),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://pw.edu.pl/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path, test_path = dl_manager.download([_TRAIN_URL, _TEST_URL])

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(train_path),
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(test_path),
                })
        ]

    def _generate_examples(self, archive):
        """Generate rock, paper or scissors images and labels given the directory path.

        Args:
          archive: object that iterates over the zip.

        Yields:
          The image path and its corresponding label.
        """

        for file_name, file_object in archive:
            res = _NAME_RE.match(file_name)
            if not res:
                continue
            label = res.group(2).lower()
            record = {
                "image": file_object,
                "label": label,
            }
            if os.environ.get('VERBOSE') and os.environ.get('VERBOSE').lower() == 'true':
                print(f"Example:e {record}")
            yield file_name, record
