"""
Utility functions for the package


"""
import abc
from moviepy.editor import ImageSequenceClip
import logging


logger = logging.getLogger(__name__)


class AbstractWorkflow(abc.ABC):
	"""Describes what an analysis and reporting pipeline
	must be able to do. This needs to be implemented
	in classes in pipeline, analyse and confluence,
	such that the pipeline class can call the other's methods
	"""
	def __init__(self):
		pass

	@classmethod
	def load(self):
		pass

	@classmethod
	def identify(self):
		pass

	@classmethod
	def localize(self):
		pass

	@classmethod
	def undrift(self):
		pass


# Function to adjust contrast
def adjust_contrast(img, min_quantile, max_quantile):
    min_val = np.quantile(img, min_quantile)
    max_val = np.quantile(img, max_quantile)
    img = img.astype(np.float32) - min_val
    img = img * 255 / (max_val - min_val)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return np.rollaxis(np.array([img, img, img], dtype=np.uint8), 0, 3)


def save_movie(fname, movie, max_quantile=1, fps=1):
    # Assuming 'array_3d' is your 3D numpy array and 'contrast_factors' is a list of contrast factors for each frame
    adjusted_images = [adjust_contrast(frame, 0, max_quantile)[..., np.newaxis] for frame in movie]

    # Create movie file
    clip = ImageSequenceClip(adjusted_images, fps=fps)
    clip.write_videofile(fname, verbose=False)#, codec='mpeg4')
