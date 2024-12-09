:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.
:- table locatedInCR/2.

% Base facts
locatedInCR(spain, europe).
locatedInCR(france, europe).
neighborOf(italy, spain).
neighborOf(italy, france).

% Rule with Depth Tracking
locatedInCR_with_depth(X, Z, Depth, Proof) :-
    locatedInCR(X, Z),
    Depth = 1, % Base facts have depth 1
    Proof = [locatedInCR(X, Z)]. % Record the base fact as proof

locatedInCR_with_depth(X, Z, Depth, Proof) :-
    neighborOf(X, Y),
    locatedInCR_with_depth(Y, Z, SubDepth, SubProof),
    Depth is SubDepth + 1, % Add 1 for each recursive step
    Proof = [neighborOf(X, Y) | SubProof]. % Record proof steps

% Query to Return Depth and Proofs
query_with_depth(X, Z, Depths, Proofs) :-
    findall([Depth, Proof],
            locatedInCR_with_depth(X, Z, Depth, Proof),
            Results),
    % Separate depths and proofs into two lists
    findall(D, member([D, _], Results), Depths),
    findall(P, member([_, P], Results), Proofs).