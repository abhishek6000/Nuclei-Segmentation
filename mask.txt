def mask():
	masks = []
	masks_folder = '/kaggle/working/*/masks/'
	for mask_set in glob.glob(masks_folder):
    	mask_name = os.listdir(mask_set)
    	mask_tot = np.zeros((256,256))
    		for i in range(len(mask_name)):
        		mask_tot += cv2.resize(plt.imread(mask_set+mask_name[i]), (256,256))
    masks.append(mask_tot)