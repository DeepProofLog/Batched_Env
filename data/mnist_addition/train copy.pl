:- discontiguous addition/3
:- discontiguous digit/2
% nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9])  digit(X,Y).
% multiplication(X,Y,Z) :- digit(X,X2), digit(Y,Y2), digit(Z,Z2), Z2 is X2*Y2.
% addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2),digit(Z,Z2), Z2 is X2+Y2.
addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.
proof_first([H | T], T) :- call(H).