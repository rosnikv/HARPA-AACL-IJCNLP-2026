# utils/llm_logger.py

import os
import csv
from threading import Lock
from datetime import datetime

lock = Lock()
LOG_FILE = None  # Will be set dynamically

PRICING = {
    "gpt-4o": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "claude-3.5-sonnet": {"prompt": 3.0, "completion": 15.0},
    "claude-4-sonnet": {"prompt": 3.0, "completion": 15.0},
    "o3-mini": {"prompt": 1.10, "completion": 4.40},
    "deepseek/deepseek-reasoner": {"prompt": 0.55, "completion": 2.20},
    "default": {"prompt": 5.0, "completion": 20.0}
}

def set_log_file_path(output_dir: str):
    global LOG_FILE
    os.makedirs(output_dir, exist_ok=True)
    LOG_FILE = os.path.join(output_dir, "llm_usage_log.csv")
    print(f"Log file will be saved to: {LOG_FILE}")
    
    # Test write permissions
    try:
        with open(LOG_FILE, 'a') as f:
            pass
        print(f"Log file will be saved to: {LOG_FILE}")
    except PermissionError:
        print(f"No write permission for: {LOG_FILE}")
        raise

def _has_header(file_path):
    """Check if CSV file has proper header."""
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            # Check if first line is exactly the header we expect
            expected_header = "timestamp,model,prompt_tokens,completion_tokens,total_tokens,cost_usd"
            return first_line == expected_header
    except:
        return False

def log_llm_usage(model: str, prompt_tokens: int, completion_tokens: int):
    print(f"[LOGGING] model={model}, prompt={prompt_tokens}, completion={completion_tokens}")
    
    if LOG_FILE is None:
        raise ValueError("LOG_FILE path not set. Call set_log_file_path(output_dir) first.")
    
    try:
        pricing = PRICING.get(model, PRICING["default"])
        cost = (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) / 1_000_000

        with lock:
            # Check if we need to write header
            needs_header = not _has_header(LOG_FILE)
            
            with open(LOG_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                if needs_header:
                    print("  Writing CSV header...")
                    writer.writerow([
                        "timestamp", "model", "prompt_tokens",
                        "completion_tokens", "total_tokens", "cost_usd"
                    ])
                writer.writerow([
                    datetime.now().isoformat(), model, prompt_tokens,
                    completion_tokens, prompt_tokens + completion_tokens,
                    round(cost, 6)
                ])
        print(f"Logged to: {LOG_FILE}")
        
    except Exception as e:
        print(f"Error logging LLM usage: {e}")
        print(f"LOG_FILE: {LOG_FILE}")
        raise

def summarize_cost_log():
    if LOG_FILE is None or not os.path.exists(LOG_FILE):
        print("No LLM usage log found.")
        return
    
    try:
        import pandas as pd
        # Read CSV and ensure cost_usd is treated as numeric
        df = pd.read_csv(LOG_FILE)
        
        # Debug: show what columns we found
        print(f"CSV columns detected: {list(df.columns)}")
        
        # Check if cost_usd column exists
        if 'cost_usd' not in df.columns:
            print("Missing 'cost_usd' column. Attempting to fix...")
            _fix_csv_header()
            # Try again after fix
            df = pd.read_csv(LOG_FILE)
            
        if 'cost_usd' in df.columns:
            # Convert cost_usd to numeric (handles string values)
            df['cost_usd'] = pd.to_numeric(df['cost_usd'], errors='coerce')
            
            # Remove any rows where cost_usd couldn't be converted (NaN)
            df = df.dropna(subset=['cost_usd'])
            
            total = df["cost_usd"].sum()
            count = len(df)
            print(f"\n=== LLM Usage Summary ===")
            print(f"Total Requests: {count}")
            print(f"Total Cost: ${total:.6f}")
            
            # Show breakdown by model
            if count > 0:
                model_summary = df.groupby('model')['cost_usd'].agg(['sum', 'count']).round(6)
                model_summary.columns = ['total_cost', 'requests']
                print(f"\nBy Model:")
                for model, row in model_summary.iterrows():
                    print(f"  {model}: ${row['total_cost']:.6f} ({int(row['requests'])} requests)")
            
            print("=" * 25)
        else:
            print("Still cannot find 'cost_usd' column after fix attempt.")
            
    except ImportError:
        print("Pandas not available. Install with: pip install pandas")
    except Exception as e:
        print(f"Error reading log: {e}")
        # Show first few lines of file for debugging
        try:
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()[:5]
                print("First 5 lines of log file:")
                for i, line in enumerate(lines, 1):
                    print(f"  {i}: {line.strip()}")
        except:
            pass

def _fix_csv_header():
    """Fix CSV file that's missing headers."""
    if not os.path.exists(LOG_FILE):
        return
    
    print("Attempting to fix CSV headers...")
    
    # Read all existing lines
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        return
    
    # Check if first line looks like data (not headers)
    first_line = lines[0].strip()
    if not ('timestamp' in first_line and 'model' in first_line):
        print("  Adding missing headers...")
        # Backup original
        backup_file = LOG_FILE + '.backup'
        with open(backup_file, 'w') as f:
            f.writelines(lines)
        print(f"Backup created: {backup_file}")
        
        # Write new file with headers
        with open(LOG_FILE, 'w') as f:
            f.write("timestamp,model,prompt_tokens,completion_tokens,total_tokens,cost_usd\n")
            f.writelines(lines)
        print("Headers added successfully!")
    else:
        print("Headers already present")

def clear_log():
    """Clear the log file (useful for testing)."""
    if LOG_FILE and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Cleared log file: {LOG_FILE}")