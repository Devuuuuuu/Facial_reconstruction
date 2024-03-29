import pydicom
import numpy as np
from mayavi import mlab

# Load the DICOM file
dicom_file = "export.dcm"
ds = pydicom.dcmread(dicom_file)

# Access pixel data
pixel_data = ds.pixel_array
print("Pixel data shape:", pixel_data.shape)

pixel_data_normalized = pixel_data.astype(np.float32) / np.max(pixel_data)

# Thresholding
initial_threshold = 0 #you might need to adjust this for getting the skull image

# Incremental step size
step_size = 0  # Adjust as needed

# Adjust the threshold incrementally
threshold = initial_threshold + step_size  # Adjust the initial threshold value

# Thresholding
binary_mask = (pixel_data_normalized > threshold).astype(np.uint8)

# Segment the skull
segmented_skull = pixel_data * binary_mask

# Create the volume visualization
color = (1, 0.75, 0.5)  # RGB color (values between 0 and 1), can be adjusted to resemble skull color
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(segmented_skull), color=color)
vol._volume_property.scalar_opacity_unit_distance = 0.1  # Adjust as needed


# Adjust the visualization as needed (e.g., camera position, colormap, opacity)
mlab.view(azimuth=45, elevation=45, distance=500)  # Adjust distance to bring the points into view

# Display the visualization
mlab.show()

