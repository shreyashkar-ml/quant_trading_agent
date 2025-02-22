import pandas as pd
import lzma
import dill as pickle
from typing import List, Tuple, Dict

def load_pickle(path: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    try:
        with lzma.open(path, "rb") as fp:
            return pickle.load(fp)
    except FileNotFoundError:
        print(f"Pickle file {path} not found. Fetching fresh data.")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None
    
def save_pickle(path: str, obj: Tuple[List[str], Dict[str, pd.DataFrame]]) -> None:
    try:
        with lzma.open(path, "wb") as fp:
            pickle.dump(obj, fp)
        print(f"Saved data to {path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

