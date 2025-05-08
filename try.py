import torch
# Assuming IndexManagerIdx is imported correctly, e.g.:
# from .index_manager_idx import IndexManagerIdx
# from .dataset_idx import LogicProgramTensor # Or wherever it's defined

# Dummy IndexManagerIdx for the example to run. Replace with your actual import.
class IndexManagerIdx:
    def __init__(self):
        self.ent_to_idx = {}
        self.idx_to_ent = {}
        self.pred_to_idx = {}
        self.idx_to_pred = {}
        self.TRUE_IDX = -1 # Placeholder
        self.FALSE_IDX = -2 # Placeholder

    def add_token(self, token, token_type):
        if token_type == "pred":
            if token not in self.pred_to_idx:
                idx = len(self.pred_to_idx)
                self.pred_to_idx[token] = idx
                self.idx_to_pred[idx] = token
            return self.pred_to_idx[token]
        elif token_type == "ent":
            if token not in self.ent_to_idx:
                idx = len(self.ent_to_idx)
                self.ent_to_idx[token] = idx
                self.idx_to_ent[idx] = token
            return self.ent_to_idx[token]
        return -1 # Should not happen

    def is_variable(self, token_idx: int) -> bool:
        if token_idx in self.idx_to_ent:
            ent_name = self.idx_to_ent[token_idx]
            return ent_name[0].isupper()
        return False

    def tensor_to_string(self, tensor_atom, is_goal_list=False):
        # Simplified for brevity; adapt to your actual implementation
        if tensor_atom is None:
            return "None"
        if not isinstance(tensor_atom, torch.Tensor):
             return str(tensor_atom) # Or handle as appropriate

        if tensor_atom.numel() == 1 and tensor_atom.item() == self.TRUE_IDX:
            return "[TRUE]"
        if tensor_atom.numel() == 1 and tensor_atom.item() == self.FALSE_IDX:
            return "[FALSE]"

        items = tensor_atom.tolist()
        if not items:
            return "[]"

        pred_idx = items[0]
        pred_name = self.idx_to_pred.get(pred_idx, f"P?{pred_idx}")
        args_str = []
        for arg_idx in items[1:]:
            args_str.append(self.idx_to_ent.get(arg_idx, f"E?{arg_idx}"))
        return f"{pred_name}({', '.join(args_str)})"

# Dummy LogicProgramTensor for the example to run
class LogicProgramTensor:
    def __init__(self, facts, rules, queries, labels, im, device):
        self.facts = facts
        self.rules = rules
        self.queries = queries
        self.labels = labels
        self.im = im
        self.device = device
        self.num_eval_queries = len(queries)


class MockDataHandler:
    def __init__(self, device, im: IndexManagerIdx):
        self.device = device
        self.im = im # Use the passed IndexManagerIdx instance

        # Define predicates
        preds = ['grandparent', 'parent', 'q', 'p', 'r']
        for pred in preds:
            self.im.add_token(pred, "pred")

        # Define entities and variables
        # Ensure 'X', 'Y', 'Z' are treated as distinct variables
        # Their indices will depend on the order of addition if not already present in im
        required_ents = ['john', 'mary', 'peter', 'a', 'b', 'c', 'X', 'Y', 'Z']
        for ent in required_ents:
            self.im.add_token(ent, "ent") # This will populate ent_to_idx and idx_to_ent

        # Set TRUE_IDX and FALSE_IDX after all entities and predicates are added
        # to avoid index collision.
        # This assumes TRUE_ATOM and FALSE_ATOM are special strings not used elsewhere.
        true_atom_str = "TRUE_ATOM"
        false_atom_str = "FALSE_ATOM"
        self.im.TRUE_IDX = self.im.add_token(true_atom_str, "ent")
        self.im.FALSE_IDX = self.im.add_token(false_atom_str, "ent")


        # Tensorized facts for evaluation
        self.tensorized_facts_eval = [
            torch.tensor([self.im.pred_to_idx['parent'], self.im.ent_to_idx['john'], self.im.ent_to_idx['mary']], device=device).unsqueeze(0),
            torch.tensor([self.im.pred_to_idx['parent'], self.im.ent_to_idx['mary'], self.im.ent_to_idx['peter']], device=device).unsqueeze(0),
            torch.tensor([self.im.pred_to_idx['p'], self.im.ent_to_idx['a']], device=device).unsqueeze(0),
            torch.tensor([self.im.pred_to_idx['r'], self.im.ent_to_idx['a']], device=device).unsqueeze(0),
        ]

        # Tensorized rules for evaluation
        # Rule 1: grandparent(X,Y) :- parent(X,Z), parent(Z,Y)
        # ************ THIS IS THE KEY CORRECTION ************
        # The head should be grandparent(X, Y)
        # The body uses X, Y, and Z.
        rule1_head = torch.tensor([
            self.im.pred_to_idx['grandparent'],
            self.im.ent_to_idx['X'],  # Variable X
            self.im.ent_to_idx['Y']   # Variable Y
        ], device=device).unsqueeze(0)
        rule1_body = [
            torch.tensor([self.im.pred_to_idx['parent'], self.im.ent_to_idx['X'], self.im.ent_to_idx['Z']], device=device).unsqueeze(0), # parent(X,Z)
            torch.tensor([self.im.pred_to_idx['parent'], self.im.ent_to_idx['Z'], self.im.ent_to_idx['Y']], device=device).unsqueeze(0)  # parent(Z,Y)
        ]

        # Rule 2: q(X,Y) :- p(X), r(Y)  (Assuming this was the intended second rule, or adjust as needed)
        # For demonstration, let's assume it was q(A,B) :- p(A), r(B) to use different variable names
        # If you use X, Y again, they are the same X, Y as in the grandparent rule *within this rule's scope*
        # It's generally good practice to use distinct variable names if they are meant to be distinct,
        # or understand that X in one rule is independent of X in another.
        # Let's use A, B for clarity here, assuming A, B are added to required_ents if not already.
        # If 'A' and 'B' are intended as constants, they should not be uppercase.
        # Assuming 'A', 'B' are variables for this example:
        if 'A' not in self.im.ent_to_idx: self.im.add_token('A', "ent")
        if 'B' not in self.im.ent_to_idx: self.im.add_token('B', "ent")

        rule2_head = torch.tensor([
            self.im.pred_to_idx['q'],
            self.im.ent_to_idx['X'], # Using X
            self.im.ent_to_idx['Y']  # Using Y
        ], device=device).unsqueeze(0)
        rule2_body = [
            torch.tensor([self.im.pred_to_idx['p'], self.im.ent_to_idx['X']], device=device).unsqueeze(0), # p(X)
            torch.tensor([self.im.pred_to_idx['r'], self.im.ent_to_idx['Y']], device=device).unsqueeze(0)  # r(Y) (Corrected from your log's Z)
        ]


        self.tensorized_rules_eval = [
            (rule1_head, rule1_body),
            (rule2_head, rule2_body) # Add more rules as needed
        ]

        # Tensorized queries for evaluation
        self.tensorized_queries_eval = [
            torch.tensor([self.im.pred_to_idx['grandparent'], self.im.ent_to_idx['john'], self.im.ent_to_idx['peter']], device=device),
            torch.tensor([self.im.pred_to_idx['q'], self.im.ent_to_idx['a'], self.im.ent_to_idx['b']], device=device), # q(a,b)
            torch.tensor([self.im.pred_to_idx['parent'], self.im.ent_to_idx['a'], self.im.ent_to_idx['b']], device=device), # parent(a,b) - likely false
        ]
        self.tensorized_labels_eval = [1, 0, 0] # 1 for true, 0 for false

        # Create the LogicProgramTensor for evaluation
        self.eval_program = LogicProgramTensor(
            facts=self.tensorized_facts_eval,
            rules=self.tensorized_rules_eval,
            queries=self.tensorized_queries_eval,
            labels=self.tensorized_labels_eval,
            im=self.im,
            device=self.device
        )

    def get_eval_program(self) -> LogicProgramTensor:
        return self.eval_program

# Example Usage (simplified, adapt to your test setup)
if __name__ == '__main__':
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {test_device}")

    index_manager = IndexManagerIdx()
    # Initialize TRUE_IDX and FALSE_IDX if they are determined outside MockDataHandler
    # For this example, MockDataHandler does it.

    mock_data_handler = MockDataHandler(device=test_device, im=index_manager)
    eval_program = mock_data_handler.get_eval_program()

    print("\n--- Index Manager State ---")
    print("Entities to Index:", eval_program.im.ent_to_idx)
    print("Index to Entities:", eval_program.im.idx_to_ent)
    print("Predicates to Index:", eval_program.im.pred_to_idx)
    print("Index to Predicates:", eval_program.im.idx_to_pred)
    print(f"TRUE_IDX: {eval_program.im.TRUE_IDX}, FALSE_IDX: {eval_program.im.FALSE_IDX}")


    print("\n--- Tensorized Rules (Eval) ---")
    for i, (head, body) in enumerate(eval_program.rules):
        print(f"Rule {i+1}:")
        print(f"  Head: {eval_program.im.tensor_to_string(head.squeeze(0))}")
        body_str = [eval_program.im.tensor_to_string(b.squeeze(0)) for b in body]
        print(f"  Body: {', '.join(body_str)}")

    print("\n--- Tensorized Queries (Eval) ---")
    for i, query_tensor in enumerate(eval_program.queries):
        label = eval_program.labels[i]
        print(f"Query {i+1}: {eval_program.im.tensor_to_string(query_tensor)} (Label: {label})")

    print("\n--- Tensorized Facts (Eval) ---")
    for i, fact_tensor in enumerate(eval_program.facts):
        print(f"Fact {i+1}: {eval_program.im.tensor_to_string(fact_tensor.squeeze(0))}")

    # Test the specific case from the log
    # Rule head: grandparent(X, Y)
    # Query: grandparent(john, peter)

    # Manually get indices for demonstration
    idx_grandparent = eval_program.im.pred_to_idx['grandparent']
    idx_X = eval_program.im.ent_to_idx['X']
    idx_Y = eval_program.im.ent_to_idx['Y'] # Ensure this is different from Z's index
    idx_Z = eval_program.im.ent_to_idx['Z']

    idx_john = eval_program.im.ent_to_idx['john']
    idx_peter = eval_program.im.ent_to_idx['peter']

    print(f"\n--- Checking specific indices ---")
    print(f"grandparent: {idx_grandparent}, X: {idx_X}, Y: {idx_Y}, Z: {idx_Z}, john: {idx_john}, peter: {idx_peter}")

    # This should now use the corrected rule head:
    # eval_program.rules[0][0] is grandparent(X,Y)
    # tensor([idx_grandparent, idx_X, idx_Y])
    # eval_program.queries[0] is grandparent(john,peter)
    # tensor([idx_grandparent, idx_john, idx_peter])

    # Your unification engine would then try to unify:
    # eval_program.rules[0][0] with eval_program.queries[0]
    # Example:
    # unification_engine = PythonUnificationIdx(idx_manager=eval_program.im)
    # substitution = unification_engine.unify_atoms_tensor(eval_program.queries[0], eval_program.rules[0][0])
    # print(f"\nUnification of {eval_program.im.tensor_to_string(eval_program.queries[0])} and {eval_program.im.tensor_to_string(eval_program.rules[0][0].squeeze(0))}:")
    # if substitution is not None:
    #     print("  Success! Substitution:")
    #     for var_idx, term_idx in substitution.items():
    #         print(f"    {eval_program.im.idx_to_ent.get(var_idx, '?')}: {eval_program.im.idx_to_ent.get(term_idx, '?')}")
    # else:
    #     print("  Failed.")

