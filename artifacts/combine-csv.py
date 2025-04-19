import os
import csv
import sys

# --- Configuration ---
TARGET_DIRECTORY = '.'  # Current directory
OUTPUT_FILENAME = 'combined_validated_data.csv' # Name for the combined output file

# --- Helper Function to Read Header ---
def read_csv_header(filepath):
    """
    Reads the header row from a CSV file.
    Returns the header as a list, or None if the file is empty.
    Raises IOError, csv.Error, or Exception on read/format issues.
    """
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            try:
                # Read the first row
                header = next(reader)
                # Simple check for completely blank header row
                if not any(field.strip() for field in header):
                     raise ValueError("Header row appears to be empty or contain only whitespace.")
                return header
            except StopIteration:
                # File is empty
                return None
            except csv.Error as e:
                # CSV format error on the first line
                raise csv.Error(f"CSV formatting error reading header line: {e}") from e
            except ValueError as ve:
                 raise ValueError(f"Invalid header content: {ve}") from ve

    except IOError as e:
        # File read error
        raise IOError(f"Could not read file: {e}") from e
    except Exception as e:
        # Catch other potential exceptions during file open/read setup
        raise Exception(f"Unexpected error reading header: {e}") from e

# --- Main Logic ---
print(f"Scanning directory: {os.path.abspath(TARGET_DIRECTORY)}")
print(f"Output file will be: {OUTPUT_FILENAME}")

try:
    # List all items
    all_items = os.listdir(TARGET_DIRECTORY)
except FileNotFoundError:
    print(f"Error: Directory not found: {os.path.abspath(TARGET_DIRECTORY)}")
    sys.exit(1)
except OSError as e:
    print(f"Error accessing directory {os.path.abspath(TARGET_DIRECTORY)}: {e}")
    sys.exit(1)

# Filter for .csv files (case-insensitive), ensure they are files, exclude output file
# Sort the list for consistent processing order (helps determine the 'first' file)
csv_files = sorted([
    f for f in all_items
    if os.path.isfile(os.path.join(TARGET_DIRECTORY, f))
       and f.lower().endswith('.csv')
       and f != OUTPUT_FILENAME
])

if not csv_files:
    print(f"No .csv files found in '{os.path.abspath(TARGET_DIRECTORY)}' (excluding '{OUTPUT_FILENAME}').")
    sys.exit(0)

print(f"\nFound {len(csv_files)} candidate CSV file(s): {', '.join(csv_files)}")

# --- Phase 1: Header Validation ---
print("\n--- Validating Headers ---")
reference_header = None
first_valid_file = None
headers_validated = True

for filename in csv_files:
    file_path = os.path.join(TARGET_DIRECTORY, filename)
    print(f"- Checking: {filename}...", end="")

    try:
        current_header = read_csv_header(file_path)

        if current_header is None:
            # File is empty - Cannot validate or use as reference
            print(" Error: File is empty. Cannot validate headers.")
            headers_validated = False
            break # Stop validation

        if reference_header is None:
            # This is the first non-empty file, use its header as reference
            reference_header = current_header
            first_valid_file = filename
            print(f" OK (Reference header with {len(reference_header)} columns established)")
        elif current_header == reference_header:
            # Header matches the reference
            print(f" OK ({len(current_header)} columns match reference)")
        else:
            # Header mismatch found!
            print("\n\n--- FATAL ERROR: HEADER MISMATCH ---")
            print(f"File '{filename}' header does not match the reference header from '{first_valid_file}'.")
            print(f"\nReference Header ({len(reference_header)} cols):")
            print(reference_header)
            print(f"\n'{filename}' Header ({len(current_header)} cols):")
            print(current_header)
            print("\nAborting combination process.")
            headers_validated = False
            break # Stop validation

    except (IOError, csv.Error, ValueError, Exception) as e:
        print(f"\n\n--- FATAL ERROR: Cannot Read/Validate Header ---")
        print(f"File: {filename}")
        print(f"Error: {e}")
        print("\nAborting combination process.")
        headers_validated = False
        break # Stop validation


# --- Check Validation Result ---
if not headers_validated:
    sys.exit(1) # Exit if any validation step failed

if reference_header is None:
    print("\nError: No valid, non-empty CSV files found to process.")
    sys.exit(1)

print("\n--- Header Validation Successful ---")


# --- Phase 2: Combine Data (only if validation passed) ---
print("\n--- Combining Data ---")
files_processed_count = 0
files_skipped_in_data_phase = 0
total_data_rows_written = 0

try:
    # Open the output file ONCE for writing
    with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)

        # Write the validated reference header
        print(f"Writing header to '{OUTPUT_FILENAME}'...")
        csv_writer.writerow(reference_header)

        # Loop through files again to append data rows
        for filename in csv_files:
            file_path = os.path.join(TARGET_DIRECTORY, filename)
            print(f"- Appending data from: {filename}...", end="")
            appended_rows = 0
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                    csv_reader = csv.reader(infile)
                    try:
                        # Skip the header row (already validated and written)
                        next(csv_reader)

                        # Write data rows
                        for row in csv_reader:
                            # Optionally skip blank rows
                            if any(field.strip() for field in row):
                                csv_writer.writerow(row)
                                appended_rows += 1
                            # else: print(" Skipped blank row.", end="")

                        print(f" Appended {appended_rows} data row(s).")
                        files_processed_count += 1
                        total_data_rows_written += appended_rows

                    except StopIteration:
                        # File only contained the header (which was validated)
                        print(" No data rows found after header.")
                        files_processed_count += 1 # Still counts as processed
                    except csv.Error as csv_e:
                        # Error reading DATA rows
                        print(f" Error: CSV formatting error while reading data near line {csv_reader.line_num}: {csv_e}. Skipping rest of this file.")
                        files_skipped_in_data_phase += 1


            except IOError as io_e:
                # Error opening/reading file during data append phase
                print(f" Error: Could not read file during data append phase: {io_e}. Skipping.")
                files_skipped_in_data_phase += 1
            except Exception as e:
                # Unexpected error during data append for this file
                print(f" Error: An unexpected error occurred appending data: {e}. Skipping.")
                files_skipped_in_data_phase += 1

except IOError as e:
    print(f"\nCRITICAL Error: Could not write to output file '{OUTPUT_FILENAME}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nCRITICAL Error: An unexpected error occurred during the writing process: {e}")
    sys.exit(1)


# --- Final Summary ---
print("\n--- Combination Summary ---")
print(f"Validated header using: '{first_valid_file}'")
print(f"Successfully processed {files_processed_count} file(s) during data appending phase.")
print(f"Combined data written to '{OUTPUT_FILENAME}'.")
print(f"Total data rows written (excluding header): {total_data_rows_written}")
if files_skipped_in_data_phase > 0:
     print(f"Warning: {files_skipped_in_data_phase} file(s) were skipped during data appending due to errors (check logs above).")
