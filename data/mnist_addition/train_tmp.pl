:- discontiguous addition/3.
:- discontiguous digit/2.
addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.
proof_first([H | T], T) :- call(H).