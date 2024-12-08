:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.
:- table locatedInCR/2.
locatedInCR(saint_martin,americas).
neighborOf(sint_maarten,saint_martin).
neighborOf(saint_martin,sint_maarten).
locatedInCR(X,Z) :- neighborOf(X,Y), locatedInCR(Y,Z).
