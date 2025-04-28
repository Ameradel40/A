import pydicom
import os

# Update this path to point to your actual DICOM file
dicom_file_path = r'E:\Compressed\rsna-2024-lumbar-spine-degenerative-classification\test_images\44036939'

# Check if the file exists
if os.path.exists(dicom_file_path):
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file_path)
        print("DICOM file read successfully.")
        # Optionally, you can print some information about the DICOM file
        print(ds)
    except Exception as e:
        print(f"An error occurred while reading the DICOM file: {e}")
else:
    print(f"Error: File not found at {dicom_file_path}.")
