neighborOf(italy,france).
neighborOf(france,italy).
%locatedInCR(france,europe).

locatedInCR(X,Z) :- neighborOf(X,Y), locatedInCR(Y,Z).
proof_first([H | T], T) :- call(H).
