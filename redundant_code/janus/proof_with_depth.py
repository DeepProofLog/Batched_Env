import janus_swi as janus

janus.consult("proof_with_depth.pl")
res = janus.query_once(f'query_with_depth(italy, europe, _Depths, _Proofs), term_string(_Depths, Depths), term_string(_Proofs, Proofs)')
print(res)