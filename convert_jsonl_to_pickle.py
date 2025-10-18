import json
import pickle
from pathlib import Path


def convert_jsonl_to_pickle(constants_dir: str = "constants", output_file: str = "constants_cache.pkl"):
    """Convert JSONL constant files to pickle format."""

    constants = {}
    precision_map = {}

    constants_path = Path(constants_dir)

    # Process each JSONL file
    for jsonl_file in constants_path.glob("*.jsonl"):
        const_name = jsonl_file.stem  # 'pi' or 'e'

        print(f"Processing {jsonl_file}...")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)

                        # Your format: {"name": "constant", "value": "n"}
                        name = data.get('name', const_name)
                        value = data.get('value')

                        if value:
                            # Count actual decimal places
                            if '.' in value:
                                decimal_places = len(value.split('.')[1])
                            else:
                                decimal_places = 0

                            # Store with actual precision
                            precision = decimal_places if decimal_places > 0 else 100000

                            # Store full precision version
                            key = f"{const_name}_{precision}"
                            constants[key] = value

                            # Also store common precision levels for faster lookup
                            for p in [100, 1000, 10000, 50000, 100000]:
                                if p <= decimal_places:
                                    truncated_key = f"{const_name}_{p}"
                                    if '.' in value:
                                        parts = value.split('.')
                                        constants[truncated_key] = f"{parts[0]}.{parts[1][:p]}"

                            # Track precision levels
                            if const_name not in precision_map:
                                precision_map[const_name] = []
                            precision_map[const_name].extend([100, 1000, 10000, 50000, precision])

                            print(f"  Loaded {const_name} with {decimal_places} decimal places")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {jsonl_file}: {e}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")

    # Remove duplicates from precision map
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
        print(f"  {const}: precisions {precisions}")

    return pickle_data


# Run the conversion
if __name__ == "__main__":
    convert_jsonl_to_pickle()


