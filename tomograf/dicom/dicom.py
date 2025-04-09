import pydicom

# Wczytaj plik
file_path = r"C:\Users\Kubol\Desktop\Studia\VI Semestr\Informatyka w Medycynie\Tomograf\informatyka_w_medycynie\tomograf\dicom\test.dcm"
ds = pydicom.dcmread(file_path)

# Wyświetl dane pacjenta
print(f"Patient Name: {ds.PatientName}")
print(f"Patient ID: {ds.PatientID}")
print(f"Birthdate: {ds.PatientBirthDate}")
print(f"Study Description: {ds.StudyDescription}")