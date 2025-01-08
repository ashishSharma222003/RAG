import os
import shutil
import ocrmypdf

# Base directory for user-specific data
BASE_DIR = "./user_data"

# Languages for OCR
LANGUAGES = 'eng+hin+ben+chi_sim+chi_tra'

def process_pdf_files_in_directory(user_id: str, source_dir: str):
    """
    Process PDF files in a directory and save the OCR-processed PDFs in a user-specific directory.

    Args:
        user_id (str): Unique identifier for the user.
        source_dir (str): Directory containing the raw PDF files.

    Returns:
        str: Path to the directory where OCR-processed PDFs are saved.
    """
    # User-specific directories
    user_dir = os.path.join(BASE_DIR, user_id)
    destination_dir = os.path.join(user_dir, "ocr_processed")
    os.makedirs(destination_dir, exist_ok=True)

    # Loop through all files and subdirectories in the source directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # If it's a directory, recursively process files inside
        if os.path.isdir(file_path):
            print(f"Entering directory: {file_path}")
            process_pdf_files_in_directory(user_id, file_path)
        elif filename.endswith('.pdf'):  # If it's a PDF file
            output_pdf = os.path.join(destination_dir, filename)

            try:
                print(f"Processing: {file_path}")

                # OCR processing step (currently skipped)
                # ocrmypdf.ocr(file_path, output_pdf, language=LANGUAGES, force_ocr=True)

                # Instead of performing OCR, just copy the PDF
                shutil.copy(file_path, output_pdf)
                print(f"Copied PDF without OCR: {file_path} -> Saved as: {output_pdf}")

                # Delete the original PDF after copying
                os.remove(file_path)
                print(f"Deleted original file: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:  # If it's not a PDF, move the file to the user-specific directory
            try:
                destination_file = os.path.join(destination_dir, filename)
                shutil.move(file_path, destination_file)
                print(f"Moved non-PDF file: {file_path} -> {destination_file}")
            except Exception as e:
                print(f"Error moving non-PDF file {file_path}: {e}")

    print(f"Processing completed for user: {user_id}. OCR results saved in: {destination_dir}")

    return destination_dir
