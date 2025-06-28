from collections import deque

class NFAState:
    def __init__(self, id):
        self.id = id
        self.transitions = {}
    
    def add_transition(self, symbol, state):
        if symbol not in self.transitions:
            self.transitions[symbol] = set()
        self.transitions[symbol].add(state)
    
    def __repr__(self):
        return f"State({self.id})"

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

OPERATORS = {'(', ')', '|', '*', '+', '?', '.'}

def lex(regex):
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c == '\\':
            if i + 1 < len(regex):
                tokens.append(regex[i+1])
                i += 2
            else:
                raise ValueError("Invalid escape at end of regex")
        elif c in OPERATORS:
            tokens.append(c)
            i += 1
        else:
            tokens.append(c)
            i += 1
    return tokens

def insert_concat(tokens):
    if len(tokens) <= 1:
        return tokens
    new_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        curr = tokens[i]
        if (prev in [')','*','+','?'] or prev not in OPERATORS) and \
           (curr in ['('] or curr not in OPERATORS):
            new_tokens.append('.')
        new_tokens.append(curr)
    return new_tokens

def to_rpn(tokens):
    output = []
    stack = []
    precedence = {'*':5, '+':5, '?':5, '.':4, '|':3}
    for token in tokens:
        if token not in OPERATORS:
            output.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses")
            stack.pop()  # Remove '('
        else:
            while stack and stack[-1] != '(' and precedence.get(stack[-1],0) >= precedence.get(token,0):
                output.append(stack.pop())
            stack.append(token)
    while stack:
        if stack[-1] == '(':
            raise ValueError("Mismatched parentheses")
        output.append(stack.pop())
    return output

def basic_nfa(symbol):
    global state_counter
    s0 = NFAState(state_counter)
    state_counter += 1
    s1 = NFAState(state_counter)
    state_counter += 1
    s0.add_transition(symbol, s1)
    return NFA(s0, s1)

def empty_nfa():
    global state_counter
    s0 = NFAState(state_counter)
    state_counter += 1
    return NFA(s0, s0)

def concat_nfa(nfa1, nfa2):
    nfa1.accept.add_transition(None, nfa2.start)
    return NFA(nfa1.start, nfa2.accept)

def union_nfa(nfa1, nfa2):
    global state_counter
    s0 = NFAState(state_counter)
    state_counter += 1
    s1 = NFAState(state_counter)
    state_counter += 1
    s0.add_transition(None, nfa1.start)
    s0.add_transition(None, nfa2.start)
    nfa1.accept.add_transition(None, s1)
    nfa2.accept.add_transition(None, s1)
    return NFA(s0, s1)

def star_nfa(nfa):
    global state_counter
    s0 = NFAState(state_counter)
    state_counter += 1
    s1 = NFAState(state_counter)
    state_counter += 1
    s0.add_transition(None, nfa.start)
    s0.add_transition(None, s1)
    nfa.accept.add_transition(None, nfa.start)
    nfa.accept.add_transition(None, s1)
    return NFA(s0, s1)

def plus_nfa(nfa):
    nfa_star = star_nfa(nfa)
    return concat_nfa(nfa, nfa_star)

def optional_nfa(nfa):
    global state_counter
    s0 = NFAState(state_counter)
    state_counter += 1
    s1 = NFAState(state_counter)
    state_counter += 1
    s0.add_transition(None, nfa.start)
    s0.add_transition(None, s1)
    nfa.accept.add_transition(None, s1)
    return NFA(s0, s1)

def build_nfa(rpn):
    stack = []
    for token in rpn:
        if token == '*':
            stack.append(star_nfa(stack.pop()))
        elif token == '+':
            stack.append(plus_nfa(stack.pop()))
        elif token == '?':
            stack.append(optional_nfa(stack.pop()))
        elif token == '|':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            stack.append(union_nfa(nfa1, nfa2))
        elif token == '.':
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            stack.append(concat_nfa(nfa1, nfa2))
        else:
            stack.append(basic_nfa(token))
    if len(stack) != 1:
        raise ValueError("Invalid RPN expression")
    return stack[0]

def epsilon_closure(state):
    closure = {state}
    stack = [state]
    while stack:
        s = stack.pop()
        for ns in s.transitions.get(None, set()):
            if ns not in closure:
                closure.add(ns)
                stack.append(ns)
    return closure

def move(states, symbol):
    next_states = set()
    for s in states:
        next_states.update(s.transitions.get(symbol, set()))
    return next_states

def gather_nfa_states(start):
    visited = set()
    stack = [start]
    while stack:
        state = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        for symbol, next_states in state.transitions.items():
            for ns in next_states:
                if ns not in visited:
                    stack.append(ns)
    return visited

def build_dfa(nfa):
    alphabet = set()
    for state in gather_nfa_states(nfa.start):
        for symbol in state.transitions:
            if symbol is not None:
                alphabet.add(symbol)
    start_closure = frozenset(epsilon_closure(nfa.start))
    dfa_states = {start_closure: True}
    dfa_transitions = {}
    dfa_accept = set()
    queue = deque([start_closure])
    if nfa.accept in start_closure:
        dfa_accept.add(start_closure)
    while queue:
        current = queue.popleft()
        for a in alphabet:
            next_states = move(current, a)
            closure = set()
            for s in next_states:
                closure.update(epsilon_closure(s))
            closure = frozenset(closure)
            if not closure:
                continue
            if current not in dfa_transitions:
                dfa_transitions[current] = {}
            dfa_transitions[current][a] = closure
            if closure not in dfa_states:
                dfa_states[closure] = True
                queue.append(closure)
                if nfa.accept in closure:
                    dfa_accept.add(closure)
    return {
        'start': start_closure,
        'transitions': dfa_transitions,
        'accept': dfa_accept
    }

def simulate_dfa(dfa, string):
    current = dfa['start']
    for c in string:
        if current in dfa['transitions'] and c in dfa['transitions'][current]:
            current = dfa['transitions'][current][c]
        else:
            return False
    return current in dfa['accept']

def main():
    global state_counter
    print("Enter a regex (e.g., 'a(bc)+'): ")
    regex = input().strip()
    print("Enter a string to test (e.g., 'abcbc'): ")
    string = input().strip()
    state_counter = 0
    if not regex:
        nfa = empty_nfa()
    else:
        tokens = lex(regex)
        tokens = insert_concat(tokens)
        rpn = to_rpn(tokens)
        nfa = build_nfa(rpn)
    dfa = build_dfa(nfa)
    result = simulate_dfa(dfa, string)
    print(f"Result: {result}")

if __name__ == "__main__":
    state_counter = 0
    main()