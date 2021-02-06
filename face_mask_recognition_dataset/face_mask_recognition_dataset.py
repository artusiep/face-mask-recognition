"""face_mask_recognition_dataset dataset."""
import glob

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


class FaceMaskRecognitionDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for face_mask_recognition_dataset face_mask_recognition_dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the face_mask_recognition_dataset metadata."""
        # TODO(face_mask_recognition_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your face_mask_recognition_dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(face_mask_recognition_dataset): Downloads the data and defines the splits
        path = dl_manager.download_and_extract('https://todo-data-url')

        # TODO(face_mask_recognition_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
            'test': self._generate_examples()
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(face_mask_recognition_dataset): Yields (key, example) tuples from the face_mask_recognition_dataset
        for f in path.glob('data/*/*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }


def _generate_examples(glob, path):
    """Yields examples."""
    # TODO(face_mask_recognition_dataset): Yields (key, example) tuples from the face_mask_recognition_dataset
    for f in glob.glob(f'{path}/*.jpg'):
        yield 'key', {
            'image': f,
            'label': 'yes',
        }


path = './data/00000'
for x in _generate_examples(glob, path):
    print(x)
