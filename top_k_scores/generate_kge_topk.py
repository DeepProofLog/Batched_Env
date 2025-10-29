import argparse
import heapq
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

# Add the parent directory to sys.path to allow imports from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kge_inference import KGEInference


DEFAULT_RUN_SIGNATURES: Dict[str, str] = {
    "family": "kinship_family-backward_0_1-no_reasoner-rotate-True-256-256-4-rules.txt",
    "wn18rr": "wn18rr-backward_0_1-no_reasoner-rotate-True-256-256-1-rules.txt",
    "countries_s3": "countries_s3-backward_0_1-no_reasoner-rotate-True-256-256-128-rules.txt",
}

T = TypeVar("T")

ProgressState = Dict[str, Any]


def progress_file_path(output_path: str) -> str:
    return f"{output_path}.progress.json"


def load_progress(path: str) -> Optional[ProgressState]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="ascii") as progress_file:
            return json.load(progress_file)
    except (json.JSONDecodeError, OSError):
        return None


def save_progress(path: str, state: ProgressState) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="ascii") as temp_file:
        json.dump(state, temp_file)
    os.replace(temp_path, path)


def infer_run_signature(dataset_name: str, override: Optional[str]) -> str:
    if override:
        return override
    if dataset_name in DEFAULT_RUN_SIGNATURES:
        return DEFAULT_RUN_SIGNATURES[dataset_name]
    raise ValueError(
        "No default run signature for dataset '%s'. Please provide --run-signature." % dataset_name
    )


def resolve_output_path(path: Optional[str], dataset_name: str, k: int) -> str:
    if path:
        print(f"Using provided output path: {path}")
        return path
    filename = f"kge_top{k}_{dataset_name}_facts.txt"
    path = os.path.join("./top_k_scores/", filename)
    print(f"No output path provided. Using default: {path}")
    return path


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def topk_scored_atoms(
    engine: KGEInference,
    predicate_name: str,
    anchor_constant: str,
    other_constants: Iterable[str],
    role_name: str,
    top_k: int,
    batch_size: int = 256,
    existing_facts: Optional[set] = None,
) -> List[Tuple[Tuple[str, str, str], float]]:
    if top_k <= 0:
        return []

    heap: List[Tuple[float, Tuple[str, str, str]]] = []
    for constants_batch in batched(other_constants, batch_size):
        if role_name == "head":
            atoms = [(predicate_name, anchor_constant, other) for other in constants_batch if other != anchor_constant]
        else:
            atoms = [(predicate_name, other, anchor_constant) for other in constants_batch if other != anchor_constant]

        if not atoms:
            continue

        # For locatedInCR predicate with role "head", filter out candidates 
        # where a fact with the same predicate and first constant (head) already exists
        # When role_name == "head", atoms are (predicate, anchor_constant, other)
        # where anchor_constant is the country (head/first arg) and we're varying continents (tail/second arg)
        if predicate_name == "locatedInCR" and role_name == "head" and existing_facts is not None:
            filtered_atoms = []
            for atom in atoms:
                # atom is (predicate, head, tail) where head = anchor_constant (the country)
                # Check if there's any existing fact with same predicate and head (first constant)
                head_constant = atom[1]
                # Look for any fact matching the pattern: locatedInCR(head_constant, *)
                has_existing_fact = any(
                    fact.startswith(f"{predicate_name}({head_constant},") 
                    for fact in existing_facts
                )
                if not has_existing_fact:
                    filtered_atoms.append(atom)
            atoms = filtered_atoms

        if not atoms:
            continue

        scores = engine.predict_batch(atoms)
        for atom, score in zip(atoms, scores):
            entry = (score, atom)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    heap.sort(key=lambda item: item[0], reverse=True)
    return [(atom, score) for score, atom in heap]


def load_existing_facts(dataset_name: str, data_path: str) -> set:
    """Load training, validation, and test facts to filter them out from predictions."""
    existing_facts = set()
    
    # Load training facts
    train_file = os.path.join(data_path, dataset_name, "train.txt")
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fact = line.rstrip(".")
                existing_facts.add(fact)
        print(f"Loaded {len(existing_facts)} training facts from {train_file}")
    else:
        print(f"Warning: Training file not found at {train_file}")
    
    # Load validation facts
    val_file = os.path.join(data_path, dataset_name, "valid.txt")
    initial_count = len(existing_facts)
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fact = line.rstrip(".")
                existing_facts.add(fact)
        val_count = len(existing_facts) - initial_count
        print(f"Loaded {val_count} validation facts from {val_file}")
    else:
        print(f"Warning: Validation file not found at {val_file}")
    
    # Load test facts
    test_file = os.path.join(data_path, dataset_name, "test.txt")
    initial_count = len(existing_facts)
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fact = line.rstrip(".")
                existing_facts.add(fact)
        test_count = len(existing_facts) - initial_count
        print(f"Loaded {test_count} test facts from {test_file}")
    else:
        print(f"Warning: Test file not found at {test_file}")
    
    print(f"Total facts to filter out: {len(existing_facts)}")
    return existing_facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-k KGE facts for each constant and role.")
    parser.add_argument("--dataset", default="countries_s3", help="Dataset name (e.g., family, wn18rr, countries_s3).")
    parser.add_argument("--data-path", default="./data", help="Base path that holds dataset folders.")
    parser.add_argument(
        "--checkpoint-dir",
        default="./../../checkpoints/",
        help="Directory containing trained KGE checkpoints.",
    )
    parser.add_argument("--run-signature", default=None, help="Override run signature for the KGE checkpoint.")
    parser.add_argument("--output", default=None, help="Output txt file. Defaults to runs/top{k}_<dataset>_facts.txt")
    parser.add_argument("--k", type=int, default=10, help="Number of top predictions to keep per constant and role.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the KGE inference engine.")
    parser.add_argument(
        "--runtime-cache-size",
        type=int,
        default=0,
        help=(
            "Maximum number of runtime KGE scores to retain in memory (set 0 to disable, "
            "negative for unlimited). This is to avoid calculating the same score multiple "
            "times in a batch (in one call)."
        ),
    )
    parser.add_argument(
        "--persist-runtime-scores",
        default=False,
        help="Persist newly scored atoms in memory for reuse across runs (may increase RAM).",
    )

    args = parser.parse_args()

    run_signature = infer_run_signature(args.dataset, args.run_signature)
    output_path = resolve_output_path(args.output, args.dataset, args.k)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load training and test facts to filter them out
    existing_facts = load_existing_facts(args.dataset, args.data_path)

    engine = KGEInference(
        dataset_name=args.dataset,
        base_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        run_signature=run_signature,
        seed=args.seed,
        runtime_cache_max_entries=args.runtime_cache_size,
        persist_runtime_scores=args.persist_runtime_scores,
    )

    data_handler = engine.data_handler

    progress_path = progress_file_path(output_path)
    progress_state = load_progress(progress_path)

    resume_active = False
    resume_predicate_idx = 0
    resume_role_name = ""
    resume_constant_index = -1
    resume_offset = 0

    if progress_state and not progress_state.get("finished", False):
        if os.path.exists(output_path):
            resume_active = True
            resume_predicate_idx = int(progress_state.get("predicate_index", 0) or 0)
            resume_role_name = progress_state.get("role_name", "") or ""
            resume_constant_index = int(progress_state.get("constant_index", -1) or -1)
            resume_offset = int(progress_state.get("file_offset", 0) or 0)
            print(
                "Resuming from predicate=%s, role=%s, constant_index=%d"
                % (
                    progress_state.get("predicate_name"),
                    resume_role_name or "<unknown>",
                    resume_constant_index,
                )
            )
        else:
            print("Progress file found but output file missing. Starting a fresh run.")
            progress_state = None
    elif progress_state and progress_state.get("finished", False):
        print("Previous run marked as finished. Starting a fresh run and overwriting results.")
        progress_state = None

    file_mode = "r+" if resume_active else "w"
    with open(output_path, file_mode, encoding="ascii") as f:
        if resume_active:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            safe_offset = min(resume_offset, file_size)
            f.seek(safe_offset)
            f.truncate()
            f.seek(safe_offset)

        total_predicates = len(data_handler.predicates)
        resuming = resume_active
        last_state: Optional[ProgressState] = progress_state

        for predicate_index, predicate in enumerate(data_handler.predicates):
            if predicate.arity != 2:
                continue

            domains = predicate.domains
            if len(domains) != 2:
                continue

            role_domains = {
                "head": domains[0].constants,
                "tail": domains[1].constants,
            }
            role_items = list(role_domains.items())

            if resuming and predicate_index < resume_predicate_idx:
                continue

            if resuming and predicate_index == resume_predicate_idx:
                resume_role_index = next(
                    (idx for idx, (name, _) in enumerate(role_items) if name == resume_role_name),
                    None,
                )
                if resume_role_index is None:
                    print(
                        f"Saved role '{resume_role_name}' not found for predicate '{predicate.name}'. "
                        "Restarting from current predicate."
                    )
                    resuming = False
            else:
                resume_role_index = None

            for role_idx, (role_name, constants) in enumerate(role_items):
                other_constants = role_domains["tail" if role_name == "head" else "head"]
                if not constants or not other_constants:
                    continue

                # Skip relations locatedInCS and locatedInSR entirely
                if predicate.name in ["locatedInCS", "locatedInSR"]:
                    continue

                # For locatedInCR, only process when role is "head" (varying continent for fixed country)
                # Skip when role is "tail" (varying country for fixed continent)
                if predicate.name == "locatedInCR" and role_name == "tail":
                    continue

                # For locatedInCR, use top_k=1, otherwise use args.k
                current_k = 1 if predicate.name == "locatedInCR" else args.k

                start_constant_index = 0
                if resuming:
                    if predicate_index < resume_predicate_idx:
                        continue
                    if predicate_index == resume_predicate_idx:
                        if resume_role_index is None:
                            resuming = False
                        elif role_idx < resume_role_index:
                            continue
                        elif role_idx > resume_role_index:
                            resuming = False
                        else:
                            start_constant_index = resume_constant_index + 1
                            if start_constant_index >= len(constants):
                                resuming = False
                                continue
                            resuming = False
                    else:
                        resuming = False

                print(
                    f"# predicate={predicate.name} ({predicate_index}/{total_predicates}). role={role_name}.\n"
                )

                for j, constant in enumerate(constants[start_constant_index:], start=start_constant_index):
                    print(f"Scoring constant {j}/{len(constants)}: {constant}", end= "\r")
                    scored_pairs = topk_scored_atoms(
                        engine=engine,
                        predicate_name=predicate.name,
                        anchor_constant=constant,
                        other_constants=other_constants,
                        role_name=role_name,
                        top_k=current_k,
                        existing_facts=existing_facts,
                    )
                    for rank, (atom_tuple, score) in enumerate(scored_pairs, start=1):
                        # print(f"    {atom_tuple} -> {score:.6f}")
                        fact_str = f"{atom_tuple[0]}({atom_tuple[1]},{atom_tuple[2]})"
                        # Filter out facts that are already in training or test data
                        if fact_str not in existing_facts:
                            f.write(f"{fact_str} {score:.6f} {rank}\n")
                    f.flush()
                    current_state: ProgressState = {
                        "predicate_index": predicate_index,
                        "predicate_name": predicate.name,
                        "role_name": role_name,
                        "constant_index": j,
                        "file_offset": f.tell(),
                        "finished": False,
                    }
                    save_progress(progress_path, current_state)
                    last_state = current_state

        f.flush()
        final_state: ProgressState
        if last_state:
            final_state = dict(last_state)
        else:
            final_state = {
                "predicate_index": -1,
                "predicate_name": None,
                "role_name": None,
                "constant_index": -1,
            }
        final_state["file_offset"] = f.tell()
        final_state["finished"] = True
        save_progress(progress_path, final_state)

    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
