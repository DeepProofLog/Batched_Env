digit(0). digit(1). digit(2). digit(3). digit(4).
digit(5). digit(6). digit(7). digit(8). digit(9).

is_addition(1, 0, 1).
is_addition(1, 1, 0).

addition(Z) :- is_addition(Z, X, Y), digit(X), digit(Y).
proof_first([H | T], T) :- call(H).