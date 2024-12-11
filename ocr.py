import os
import ocrmypdf
import shutil

# Source and destination directories
source_directory = "./non_ocr_pdf"
destination_directory = "./data"  # Readable files will be saved here

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Languages for OCR
languages = 'eng+hin+ben+chi_sim+chi_tra'

def process_pdf_files_in_directory(directory):
    # Loop through all files and subdirectories in the current directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # If it's a directory, recursively call the function
        if os.path.isdir(file_path):
            print(f"Entering directory: {file_path}")
            process_pdf_files_in_directory(file_path)  # Recursive call for subdirectory
        elif filename.endswith('.pdf'):  # If it's a PDF file
            output_pdf = os.path.join(destination_directory, filename)

            try:
                print(f"Processing: {file_path}")

                # Perform OCR on the PDF with 'force' behavior
                ocrmypdf.ocr(file_path, output_pdf, language=languages, force_ocr=True)

                print(f"OCR completed for: {file_path} -> Saved as: {output_pdf}")

                # Delete the original PDF after processing
                os.remove(file_path)
                print(f"Deleted original file: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        else:  # If it's not a PDF, move the file to the destination directory
            try:
                destination_file = os.path.join(destination_directory, filename)
                shutil.move(file_path, destination_file)
                print(f"Moved non-PDF file: {file_path} -> {destination_file}")
            except Exception as e:
                print(f"Error moving non-PDF file {file_path}: {e}")

# Start processing the source directory
process_pdf_files_in_directory(source_directory)

print("OCR processing completed for all files.")
