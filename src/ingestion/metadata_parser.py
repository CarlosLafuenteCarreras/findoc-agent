from pathlib import Path

def parse_financial_metadata(filename: str) -> dict:
    """
    Parse the financial metadata from a JSON file.

    Args:
        filename (str): The name of the financial metadata JSON file.
    Returns:
        dict: A dictionary containing the financial metadata.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    return {
        "source":filename,
        "company": parts[0],
        "collection": "financial",
        "type": parts[1],
        "year": parts[2],
        "language": parts[3]
        }

def parse_regulatory_metadata(filename: str) -> dict:
    """
    Parse the regulatory metadata from a JSON file.

    Args:
        filename (str): The name of the regulatory metadata JSON file.
    Returns:
        dict: A dictionary containing the regulatory metadata.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    return {
        "source":filename,
        "regulation": parts[0],
        "collection": "regulatory",
        "year": parts[1],
        "subtype": parts[2] if parts[2] not in ["1"] else None,
        "language": parts[-1]
        }
    