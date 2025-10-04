from pathlib import Path

def explore_hknews_structure():
    """Explore and print the HKNews repository structure"""
    # Assuming the HKNews folder is in the same directory as our script
    base_dir = Path("HKNews")
    
    # Check if directory exists
    if not base_dir.exists():
        print(f"Error: HKNews directory not found at {base_dir.absolute()}")
        return
    
    print(f"Found HKNews directory at: {base_dir.absolute()}")
    
    # Count language directories
    lang_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(lang_dirs)} language directories: {[d.name for d in lang_dirs]}")
    
    # Sample a few JSON files to understand structure
    json_files = []
    for lang_dir in lang_dirs:
        for year_dir in [d for d in lang_dir.iterdir() if d.is_dir()]:
            for month_dir in [d for d in year_dir.iterdir() if d.is_dir()]:
                for json_file in [f for f in month_dir.iterdir() if f.suffix.lower() == '.json']:
                    json_files.append(json_file)
                    if len(json_files) >= 3:
                        break
                if len(json_files) >= 3:
                    break
            if len(json_files) >= 3:
                break
        if len(json_files) >= 3:
            break
    
    print(f"Sample JSON files:")
    for file in json_files:
        print(f"  - {file}")

if __name__ == "__main__":
    explore_hknews_structure()