import re
from collections import defaultdict

def organize_clauses(input_file, output_file):
    # Dictionary to store clauses grouped by predicate
    clauses_by_predicate = defaultdict(list)
    
    # Regular expression to match clauses and extract predicate names
    clause_pattern = re.compile(r"^(\w+)\(.+\)\.$")
    
    with open(input_file, "r") as file:
        for line in file:
            line = line.strip()
            match = clause_pattern.match(line)
            if match:
                # Extract predicate name (e.g., "locatedInCR" from "locatedInCR(country, region).")
                predicate_name = match.group(1)
                # Append the line to the list of its respective predicate in the dictionary
                clauses_by_predicate[predicate_name].append(line)

    # Write the organized content to the output file
    with open(output_file, "w") as file:
        for predicate_name, clauses in clauses_by_predicate.items():
            for clause in clauses:
                file.write(clause + "\n")

    print(f"Organized clauses saved to {output_file}")

# Run the function
# organize_clauses("./data/countries_s1/train.txt", "./data/countries_s1/countries_s1_train_organized.txt")
organize_clauses("./data/s2_designed/train.txt", "./data/s2_designed/train_organized.txt")
organize_clauses("./data/s2_designed/test.txt", "./data/s2_designed/test_organized.txt")
