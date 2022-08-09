import numpy as np

# Calculate feature values based on image shape
class calcFeatureValues:

	shape = np.empty(68)

	def __init__(self, shape):
		self.shape = shape


	# Inner eyebrow height
	def ieb_height(self):
		return (( (self.shape[21][1]+self.shape[20][1])/2 + (self.shape[22][1]+self.shape[23][1])/2 )/2)

	# Outer eyebrow height
	def oeb_height(self):
		return (( self.shape[19][1] + self.shape[24][1] )/2)

	# Eyebrow frowned
	def eb_frowned(self):
		return ((((self.shape[21][1] - self.shape[20][1])+(self.shape[19][1] - self.shape[20][1])) + ((self.shape[22][1] - self.shape[23][1])+(self.shape[24][1]-self.shape[23][1])))/2)

	# Eyebrow slanting
	def eb_slanting(self):
		return ((((self.shape[19][1] - self.shape[20][1])+(self.shape[20][1] - self.shape[21][1])) + ((self.shape[24][1] - self.shape[23][1])+(self.shape[23][1] - self.shape[22][1])))/2)

	# Eyebrow distance
	def eb_distance(self):
		return (self.shape[22][0] - self.shape[21][0])

	# Eye and Eyebrow distance 
	def eeb_distance(self):
		return (((self.shape[19][1] - self.shape[37][1]) + (self.shape[24][1] - self.shape[44][1]))/2)

	# Eye openness 
	def e_openness(self):
		return (((self.shape[37][1] - self.shape[41][1]) + (self.shape[44][1] - self.shape[46][1]))/2)

	# Eye slanting
	def e_slanting(self):
		return (((self.shape[36][1] - self.shape[39][1]) + (self.shape[45][1] - self.shape[42][1]))/2)

	# Mouth openness
	def m_openness(self):
		return ((((-(self.shape[57][1]-self.shape[51][1])-(self.shape[59][1]-self.shape[49][1]))/2) + ((-(self.shape[57][1]-self.shape[51][1])-(self.shape[55][1]-self.shape[53][1]))/2))/2)

	# Mouth upper lip curl
	def m_mos(self):
		return (((self.shape[48][1] - (self.shape[51][1]+(( self.shape[57][1]-self.shape[51][1] )/2))) + (self.shape[54][1] - (self.shape[51][1]+( (self.shape[57][1] - self.shape[51][1])/2 ))))/2)

	# Mouth width
	def m_width(self):
		return (self.shape[54][0] - self.shape[48][0])
		
	# Mouth upper lip height
	def mul_height(self):
		return ((self.shape[49][1] + self.shape[51][1] + self.shape[53][1])/3)

	# Mouth lower lip height
	def mll_height(self):
		return ((self.shape[59][1] + self.shape[57][1] + self.shape[55][1])/3)

	# Lip corner height
	def lc_height(self):
		return ((self.shape[48][1] + self.shape[54][1])/2)


	# Get all feature values and add in a dictionary
	def getAllFeatureValues(self):
		features = {
		  "ieb_height": self.ieb_height(),      "oeb_height": self.oeb_height(),
		  "eb_frowned": self.eb_frowned(),      "eb_slanting": self.eb_slanting(),
		  "eb_distance": self.eb_distance(),	"eeb_distance": self.eeb_distance(),
		  "e_openness": self.e_openness(),      "e_slanting": self.e_slanting(),
		  "m_openness": self.m_openness(),	    "m_mos": self.m_mos(),
		  "m_width": self.m_width(), 			"mul_height": self.mul_height(),
		  "mll_height": self.mll_height(),	    "lc_height": self.lc_height()
		}

		return features
		