import janus_swi as janus

janus.consult("dynamic_with.pl")
res = janus.query_once("locatedInCR(sint_maarten,americas).")
print(res)
janus.query_once("retract(locatedInCR(sint_maarten,americas)).")
# res = janus.query_once("call_with_depth_limit(locatedInCR(sint_maarten,americas), 3, Res).")
res = janus.query_once("locatedInCR(sint_maarten,americas).")
print(res)
janus.query_once("asserta(locatedInCR(sint_maarten,americas)).")
res = janus.query_once("locatedInCR(sint_maarten,americas).")
print(res)