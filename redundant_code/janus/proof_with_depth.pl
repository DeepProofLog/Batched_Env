:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.
:- table locatedInCR/2.

% Base facts
locatedInCR(france, europe).
neighborOf(italy, france).
neighborOf(france, italy).

% Rule with Depth Tracking and Depth Limit
locatedInCR_with_depth(X, Z, Recursion, Depth, Proof) :-
    Recursion =< 5,
    locatedInCR(X, Z),
    Depth = 1, % Base facts have depth 1
    Proof = [locatedInCR(X, Z)]. % Record the base fact as proof

locatedInCR_with_depth(X, Z, Recursion, Depth, Proof) :-
    Recursion < 5,
    neighborOf(X, Y),
    Next_Recursion is Recursion + 1,
    locatedInCR_with_depth(Y, Z, Next_Recursion, SubDepth, SubProof),
    Depth is SubDepth + 1,
    Proof = [neighborOf(X, Y) | SubProof].

% Query to Return Depth and Proofs
query_with_depth(X, Z, Depths, Proofs) :-
    findall([Depth, Proof],
            locatedInCR_with_depth(X, Z, 0, Depth, Proof),
            Results),
    % Separate depths and proofs into two lists
    findall(D, member([D, _], Results), Depths),
    findall(P, member([_, P], Results), Proofs).