:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.
:- dynamic locatedInCR/2.
:- dynamic neighborOf/2.
locatedInCR(saint_martin,americas).
locatedInCR(sint_maarten,americas).
%neighborOf(sint_maarten,saint_martin).
neighborOf(saint_martin,sint_maarten).
locatedInCR(X,Z) :- neighborOf(X,Y), locatedInCR(Y,Z).
