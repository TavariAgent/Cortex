import pickle
from pathlib import Path


def inspect_pickle(pickle_file: str = "constants_cache.pkl"):
    """Inspect contents of the pickle file."""

    pickle_path = Path(pickle_file)

    if not pickle_path.exists():
        print(f"File {pickle_file} not found!")
        return

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Successfully loaded {pickle_file}")
        print(f"File size: {pickle_path.stat().st_size} bytes\n")

        # Check structure
        print("Data structure keys:", list(data.keys()))

        # Check constants
        if 'constants' in data:
            constants = data['constants']
            print(f"\nTotal constants stored: {len(constants)}")

            # Show first few entries
            print("\nSample entries:")
            for i, (key, value) in enumerate(list(constants.items())[:5]):
                # Truncate long values for display
                display_value = value[:50] + "..." if len(value) > 50 else value
                print(f"  {key}: {display_value}")

        # Check precision map
        if 'precision' in data:
            precision_map = data['precision']
            print(f"\nConstants available: {list(precision_map.keys())}")
            for const, precisions in precision_map.items():
                print(f"  {const}: {len(precisions)} precision levels")
                print(f"    Range: {min(precisions)} to {max(precisions)}")

        return data

    except pickle.UnpicklingError as e:
        print(f"❌ Error unpickling file: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return None


# Run inspection
if __name__ == "__main__":
    inspect_pickle()