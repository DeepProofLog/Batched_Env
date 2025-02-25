import os

def filter_queries():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    seen = set()
    input_path = os.path.join(base_dir, "test_label.txt")
    output_path = os.path.join(base_dir, "test_label_.txt")
    
    # Count total number of rows
    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)
    print(f"Total number of lines: {total_lines}")
    current_line = 0
    last_percent = -1
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            current_line += 1
            percent = int((current_line / total_lines) * 100)
            if percent != last_percent:
                print(f"Progress: {percent}%   ", end="\r", flush=True)
                last_percent = percent
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                outfile.write(line)
    print("Progress: 100%")

if __name__ == "__main__":
    filter_queries()
