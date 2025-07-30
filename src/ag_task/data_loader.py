import pandas as pd
from typing import List, Dict, Any

def load_ag_topics(topics_path: str) -> List[Dict[str, Any]]:
    """
    Loads and parses the Answer Generation (AG) topics file.

    Args:
        topics_path (str): The path to the CSV topics file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a question with its metadata.
    """
    try:
        df = pd.read_csv(topics_path)
        # Strip any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        required_columns = ["Q_ID", "Video_ID", "Question"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                "Topics file must contain Q_ID, Video_ID, and Question columns."
            )
            
        return df.to_dict('records')
    except FileNotFoundError:
        print(f"Error: Topics file not found at {topics_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the topics file: {e}")
        return [] 