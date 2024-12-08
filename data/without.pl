:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.
:- table locatedInCR/2.
neighborOf(sint_maarten,saint_martin).
neighborOf(saint_martin,sint_maarten).
locatedInCR(X,Z) :- neighborOf(X,Y), locatedInCR(Y,Z).
