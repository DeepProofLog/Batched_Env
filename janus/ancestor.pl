parent(bob,alice).
parent(charlie, bob).
parent(charlie, mary).

ancestor(X,Y) :- parent(X,Y).
ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).

proof_first([H | T], T) :- call(H).

prove_one_step(Goal, RuleHead :- RuleBody, RuleBody) :-
    clause(Goal, RuleBody),
    RuleHead = Goal.

prove(Query, AppliedRule, NextSubgoals) :-
    prove_one_step(Query, AppliedRule, NextSubgoals).
