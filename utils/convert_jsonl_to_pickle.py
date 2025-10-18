import json
import pickle
from pathlib import Path


def convert_jsonl_to_pickle(constants_dir: str = "constants", output_file: str = "constants_cache.pkl"):
    """Convert JSONL constant files to pickle format."""

    constants = {}
    precision_map = {}

    constants_path = Path(constants_dir)

    # Debug: Check if directory exists
    if not constants_path.exists():
        print(f"Error: Directory '{constants_dir}' does not exist!")
        return None

    # Debug: List all files in directory
    print(f"Looking in directory: {constants_path.absolute()}")
    all_files = list(constants_path.iterdir())
    print(f"Files found: {[f.name for f in all_files]}")

    # Find all JSONL files (including subdirectories)
    jsonl_files = list(constants_path.glob("**/*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {constants_dir}")
        # Try looking for .json files too
        jsonl_files = list(constants_path.glob("**/*.json"))
        if jsonl_files:
            print(f"Found .json files instead: {[f.name for f in jsonl_files]}")

    # Process each JSONL file
    for jsonl_file in jsonl_files:
        const_name = jsonl_file.stem  # 'pi', 'e', etc.

        print(f"Processing {jsonl_file}...")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)

                        # Your format: {"name": "constant", "precision": p, "value": "n", ...}
                        name = data.get('name', const_name)
                        precision = data.get('precision')
                        value = data.get('value')

                        if value and precision:
                            # Store full precision version
                            key = f"{name}_{precision}"
                            constants[key] = value

                            # Create truncated versions for common precision levels
                            for p in [100, 1000, 10000, 50000, 100000]:
                                if p <= precision:
                                    truncated_key = f"{name}_{p}"
                                    if '.' in value:
                                        parts = value.split('.')
                                        # Ensure we don't exceed available digits
                                        available_digits = len(parts[1]) if len(parts) > 1 else 0
                                        digits_to_take = min(p, available_digits)
                                        if available_digits > 0:
                                            constants[truncated_key] = f"{parts[0]}.{parts[1][:digits_to_take]}"
                                    else:
                                        constants[truncated_key] = value

                            # Track precision levels
                            if name not in precision_map:
                                precision_map[name] = []
                            precision_map[name].append(precision)

                            print(f"  Loaded {name} with {precision} decimal places")
                        else:
                            print(f"  Skipping line {line_num}: missing value or precision")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {jsonl_file}: {e}")
                    except Exception as e:
                        print(f"Unexpected error in {jsonl_file} line {line_num}: {e}")

    # Remove duplicates and sort precision map
    for const in precision_map:
        precision_map[const] = sorted(list(set(precision_map[const])))

    # Create the pickle data structure
    pickle_data = {
        'constants': constants,
        'precision': precision_map
    }

    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nConverted {len(constants)} constant entries to {output_file}")
    print(f"Constants available: {list(precision_map.keys())}")
    for const, precisions in precision_map.items():
        max_precision = max(precisions) if precisions else 0
        print(f"  {const}: max precision {max_precision} dps")

    return pickle_data


# Run the conversion
if __name__ == "__main__":
    # Try different possible locations
    result = convert_jsonl_to_pickle("constants")
    if not result or not result['constants']:
        print("\nTrying parent directory...")
        result = convert_jsonl_to_pickle("../constants")


