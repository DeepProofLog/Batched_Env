import janus_swi as janus

janus.consult("ancestor.pl")

print(janus.query("ancestor(charlie, alice)"))