import torch
import pickle
import sys
from pathlib import Path

def convert_pickle_to_pt(pickle_path, output_path=None):
    """
    Convert a pickle file to a PyTorch .pt file with weights_only=True compatibility.
    """
    pickle_path = Path(pickle_path)
    
    if output_path is None:
        output_path = pickle_path.with_suffix('.pt')
    else:
        output_path = Path(output_path)
    
    print(f"Loading pickle file: {pickle_path}")
    
    # Load the pickle file
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle file. Type: {type(data)}")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False
    
    # Analyze the data structure
    print("Analyzing data structure...")
    if isinstance(data, dict):
        print(f"Dictionary with {len(data)} keys:")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
    else:
        print(f"Data type: {type(data)}")
    
    # Convert to PyTorch-compatible format
    converted_data = convert_data_structure(data)
    
    # Save as .pt file
    try:
        torch.save(converted_data, output_path)
        print(f"Successfully saved to: {output_path}")
        
        # Verify it can be loaded with weights_only=True
        test_load = torch.load(output_path, weights_only=True)
        print("✓ Verification: File can be loaded with weights_only=True")
        return True
        
    except Exception as e:
        print(f"Error saving or verifying file: {e}")
        return False

def convert_data_structure(data):
    """
    Recursively convert data structure to be compatible with weights_only=True.
    """
    if torch.is_tensor(data):
        return data
    elif isinstance(data, (int, float, bool, str)):
        return data
    elif isinstance(data, (list, tuple)):
        return [convert_data_structure(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_data_structure(value) for key, value in data.items()}
    elif hasattr(data, '__dict__'):
        # Convert custom objects to dictionaries
        print(f"Converting custom object {type(data)} to dict")
        return convert_data_structure(data.__dict__)
    else:
        # Try to convert to tensor if it's array-like
        try:
            if hasattr(data, '__array__') or hasattr(data, 'tolist'):
                return torch.tensor(data)
            else:
                print(f"Warning: Cannot convert {type(data)}, storing as string representation")
                return str(data)
        except:
            print(f"Warning: Cannot convert {type(data)}, storing as string representation")
            return str(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_pickle_to_pt.py <pickle_file> [output_file]")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_pickle_to_pt(pickle_file, output_file)
    sys.exit(0 if success else 1)