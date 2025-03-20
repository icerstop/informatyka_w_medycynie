import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime

def save_as_dicom(image, filename, patient_info):
    """
    Save image as DICOM file with patient information.
    """
    # Create file meta dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    # Create the FileDataset instance
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add patient info
    ds.PatientName = patient_info['name']
    ds.PatientID = patient_info['id']
    ds.PatientBirthDate = patient_info['birthdate']
    
    # Add study info
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StudyDescription = patient_info['description']
    
    # Image related attributes
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = image.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = image.tobytes()
    
    # Save to disk
    ds.save_as(filename)
    
    return ds