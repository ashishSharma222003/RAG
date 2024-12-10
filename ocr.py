import os
import ocrmypdf
import shutil

source_directory="./non_ocr_pdf"
destination_directory="./data"#readable files will be present there 
# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)
languages = 'eng+hin+ben+chi_sim+chi_tra'  
# Loop through all PDF files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.pdf'):  # Check if the file is a PDF
        input_pdf = os.path.join(source_directory, filename)
        output_pdf = os.path.join(destination_directory, filename)

        try:
            print(f"Processing: {input_pdf}")

            # Perform OCR on the PDF with 'force' behavior
            # We set the 'force' option in ocrmypdf by using the --force flag internally
            ocrmypdf.ocr(input_pdf, output_pdf,language=languages, force_ocr=True)

            print(f"OCR completed for: {input_pdf} -> Saved as: {output_pdf}")

            # Delete the original PDF after processing
            os.remove(input_pdf)
            print(f"Deleted original file: {input_pdf}")

        except Exception as e:
            print(f"Error processing {input_pdf}: {e}")

print("OCR processing completed for all files.")