% --- MINIMAL KNOWLEDGE BASE V3 ---

% Add directives to allow clauses for the same predicate to be non-contiguous
:- discontiguous locatedInCR/2.
:- discontiguous neighborOf/2.

% Facts (minimal set for testing)
locatedInCR(albania, europe).
neighborOf(albania, greece).
locatedInCR(greece, europe). % This fact should now load fine after neighborOf

% --- Helper: Convert comma-separated body to list ---
comma_list((A,B), [A|List]) :- !, comma_list(B, List).
comma_list(A, [A]).

% --- Single Step Resolution Logic ---
% Attempts one resolution step on the first goal in the list.
% Returns the *list* of resulting goals (terms).
% Fails if the goal cannot be resolved. Succeeds otherwise.

% Base case: Empty list means success (represented by ['true'])
resolve_step([], ['true']).

% Optimization: If first goal is 'true', skip it and resolve the rest.
resolve_step([true|Rest], Result) :-
    !,
    resolve_step(Rest, Result).

% Failure case: If first goal is 'false', the state is ['false'].
resolve_step([false|_Rest], ['false']) :- !.

% Main resolution case:
resolve_step([Goal|Rest], NextState) :-
    Goal \= true, Goal \= false, % Ensure not already handled
    clause(Goal, Body), % Find a clause matching the Goal (can backtrack)
    ( Body == true -> % Clause is a fact
        NextState = Rest % Next state is simply the rest of the goals
    ; % Clause is a rule
      comma_list(Body, BodyList), % Convert body to list
      append(BodyList, Rest, NextState) % Prepend body, this is the next state
    ).

% --- String Conversion Helper ---
terms_to_strings([], []).
terms_to_strings([Term|T1], [String|T2]) :-
    term_string(Term, String), % Convert one term to string
    terms_to_strings(T1, T2). % Recurse

% --- findall Wrapper (SIMPLIFIED) ---
% Gets ALL possible next states from one resolve_step, converts terms to strings.
% Returns [['false']] if resolve_step fails for all clauses.
find_next_states(CurrentState, AllNextStatesAsStrings) :-
    % Use findall to get all possible outcomes of ONE resolution step
    findall(
        NextStateTermList, % Variable to collect the resulting list of terms
        resolve_step(CurrentState, NextStateTermList), % Call the single-step resolver
        AllTermStates      % Collect all possible resulting lists (can include duplicates)
    ),
    % Check if findall found any solutions
    ( AllTermStates == [] ->
        % If findall is empty, it means resolve_step failed for all possibilities
        % (e.g., the first Goal had no matching clauses)
        AllNextStatesAsStrings = [['false']]
    ;
        % If solutions were found, convert each list of terms to a list of strings
        % No sorting or filtering ['false'] here - let Python handle it.
        maplist(terms_to_strings, AllTermStates, AllNextStatesAsStrings)
    ).

% --- Recursive rule definition (KEEP COMMENTED OUT for now) ---
% locatedInCR(X,Z) :- neighborOf(X,Y), locatedInCR(Y,Z).

% --- END OF MINIMAL KB V3 ---