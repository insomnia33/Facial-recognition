# import the necessary packages
from skimage import feature

class HOG:
	def __init__(self, orientations = 9, pixelsPerCell = (8, 8), cellsPerBlock = (3, 3), normalize = False):
		# store the number of orientations, pixels per cell,
		# cells per block, and whether or not power law
		# compression should be applied
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.normalize = "L1"
	
	def describe(self, image):
		# compute HOG for the image
		hist = feature.hog(image, orientations = self.orientations,
		pixels_per_cell = self.pixelsPerCell,
		cells_per_block = self.cellsPerBlock,
		block_norm = self.normalize)
		# return the HOG features
		return hist