import numpy as np
import matplotlib.pyplot as plt

plt.ion()



def show_images_quad(images, clear=False, show=True, cmap=None, Title=None, path=None, scale_up=False):
	"""Display a list of images"""
	if clear:
		plt.close()
	n_ims = images.shape[0]
	n_sqre = int(np.ceil(np.sqrt(n_ims)))
	imgs_min_val = 0
	imgs_max_val = np.max(images)
	fig = plt.figure(Title)
	plt.clf()
	n = 1
	for image in images:
		a = fig.add_subplot(n_sqre, n_sqre, n)  # Make subplot
		if image.ndim == 2 and cmap is None:  # Is image grayscale?
			cmap = 'Greys_r'
		# plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
		plt.imshow(image, interpolation='nearest', cmap=cmap)
		plt.axis('off')
		n += 1
	if scale_up:
		fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
	if Title is not None:
		fig.suptitle(Title)
	if path is not None:
		fig.savefig(path)
	if show:
		plt.show()
		plt.pause(0.0002)