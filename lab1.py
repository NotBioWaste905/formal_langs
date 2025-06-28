from collections import deque
from typing import Set, Dict, List, Optional, Any

class NFAState:
    def __init__(self, id: int):
        self.id = id
        self.transitions: Dict[Optional[str], Set['NFAState']] = {}
    
    def add_transition(self, symbol: Optional[str], state: 'NFAState') -> None:
        if symbol not in self.transitions:
            self.transitions[symbol] = set()
        self.transitions[symbol].add(state)
    
    def __repr__(self) -> str:
        return f"State({self.id})"

class NFA:
    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept
    
    def get_all_states(self) -> Set[NFAState]:
        """Gather all states reachable from start state"""
        visited = set()
        stack = [self.start]
        while stack:
            state = stack.pop()
            if state in visited:
                continue
            visited.add(state)
            for next_states in state.transitions.values():
                for ns in next_states:
                    if ns not in visited:
                        stack.append(ns)
        return visited
    
    def get_alphabet(self) -> Set[str]:
        """Extract alphabet from all transitions"""
        alphabet = set()
        for state in self.get_all_states():
            for symbol in state.transitions:
                if symbol is not None:
                    alphabet.add(symbol)
        return alphabet

class DFAState:
    def __init__(self, nfa_states: frozenset):
        self.nfa_states = nfa_states
        self.transitions: Dict[str, 'DFAState'] = {}
        self.is_accept = False
    
    def add_transition(self, symbol: str, state: 'DFAState') -> None:
        self.transitions[symbol] = state
    
    def __repr__(self) -> str:
        return f"DFAState({len(self.nfa_states)} states)"

class DFA:
    def __init__(self, start: DFAState):
        self.start = start
        self.states: Set[DFAState] = {start}
        self.accept_states: Set[DFAState] = set()
    
    def add_state(self, state: DFAState) -> None:
        self.states.add(state)
        if state.is_accept:
            self.accept_states.add(state)
    
    def simulate(self, string: str) -> bool:
        """Simulate DFA on input string"""
        current = self.start
        for c in string:
            if c in current.transitions:
                current = current.transitions[c]
            else:
                return False
        return current in self.accept_states

class RegexParser:
    """Handles regex parsing and tokenization"""
    
    OPERATORS = {'(', ')', '|', '*', '+', '?', '.'}
    PRECEDENCE = {'*': 5, '+': 5, '?': 5, '.': 4, '|': 3}
    
    @staticmethod
    def lex(regex: str) -> List[str]:
        """Tokenize regex string"""
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
            elif c in RegexParser.OPERATORS:
                tokens.append(c)
                i += 1
            else:
                tokens.append(c)
                i += 1
        return tokens
    
    @staticmethod
    def insert_concat(tokens: List[str]) -> List[str]:
        """Insert concatenation operators where needed"""
        if len(tokens) <= 1:
            return tokens
        new_tokens = [tokens[0]]
        for i in range(1, len(tokens)):
            prev = tokens[i-1]
            curr = tokens[i]
            if (prev in [')','*','+','?'] or prev not in RegexParser.OPERATORS) and \
               (curr in ['('] or curr not in RegexParser.OPERATORS):
                new_tokens.append('.')
            new_tokens.append(curr)
        return new_tokens
    
    @staticmethod
    def to_rpn(tokens: List[str]) -> List[str]:
        """Convert infix tokens to reverse polish notation"""
        output = []
        stack = []
        for token in tokens:
            if token not in RegexParser.OPERATORS:
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
                while stack and stack[-1] != '(' and \
                      RegexParser.PRECEDENCE.get(stack[-1], 0) >= RegexParser.PRECEDENCE.get(token, 0):
                    output.append(stack.pop())
                stack.append(token)
        while stack:
            if stack[-1] == '(':
                raise ValueError("Mismatched parentheses")
            output.append(stack.pop())
        return output

class NFABuilder:
    """Handles NFA construction from regex"""
    
    def __init__(self):
        self.state_counter = 0
    
    def _new_state(self) -> NFAState:
        """Create a new NFA state with unique ID"""
        state = NFAState(self.state_counter)
        self.state_counter += 1
        return state
    
    def basic_nfa(self, symbol: str) -> NFA:
        """Create NFA for a single symbol"""
        s0 = self._new_state()
        s1 = self._new_state()
        s0.add_transition(symbol, s1)
        return NFA(s0, s1)
    
    def empty_nfa(self) -> NFA:
        """Create NFA for empty string"""
        s0 = self._new_state()
        return NFA(s0, s0)
    
    def concat_nfa(self, nfa1: NFA, nfa2: NFA) -> NFA:
        """Concatenate two NFAs"""
        nfa1.accept.add_transition(None, nfa2.start)
        return NFA(nfa1.start, nfa2.accept)
    
    def union_nfa(self, nfa1: NFA, nfa2: NFA) -> NFA:
        """Create union of two NFAs"""
        s0 = self._new_state()
        s1 = self._new_state()
        s0.add_transition(None, nfa1.start)
        s0.add_transition(None, nfa2.start)
        nfa1.accept.add_transition(None, s1)
        nfa2.accept.add_transition(None, s1)
        return NFA(s0, s1)
    
    def star_nfa(self, nfa: NFA) -> NFA:
        """Create Kleene star of NFA"""
        s0 = self._new_state()
        s1 = self._new_state()
        s0.add_transition(None, nfa.start)
        s0.add_transition(None, s1)
        nfa.accept.add_transition(None, nfa.start)
        nfa.accept.add_transition(None, s1)
        return NFA(s0, s1)
    
    def plus_nfa(self, nfa: NFA) -> NFA:
        """Create plus operator of NFA"""
        nfa_star = self.star_nfa(nfa)
        return self.concat_nfa(nfa, nfa_star)
    
    def optional_nfa(self, nfa: NFA) -> NFA:
        """Create optional operator of NFA"""
        s0 = self._new_state()
        s1 = self._new_state()
        s0.add_transition(None, nfa.start)
        s0.add_transition(None, s1)
        nfa.accept.add_transition(None, s1)
        return NFA(s0, s1)
    
    def build_nfa(self, rpn: List[str]) -> NFA:
        """Build NFA from reverse polish notation"""
        stack = []
        for token in rpn:
            if token == '*':
                stack.append(self.star_nfa(stack.pop()))
            elif token == '+':
                stack.append(self.plus_nfa(stack.pop()))
            elif token == '?':
                stack.append(self.optional_nfa(stack.pop()))
            elif token == '|':
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                stack.append(self.union_nfa(nfa1, nfa2))
            elif token == '.':
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                stack.append(self.concat_nfa(nfa1, nfa2))
            else:
                stack.append(self.basic_nfa(token))
        if len(stack) != 1:
            raise ValueError("Invalid RPN expression")
        return stack[0]

class NFAToDFAConverter:
    """Converts NFA to DFA using subset construction"""
    
    @staticmethod
    def epsilon_closure(state: NFAState) -> Set[NFAState]:
        """Compute epsilon closure of a state"""
        closure = {state}
        stack = [state]
        while stack:
            s = stack.pop()
            for ns in s.transitions.get(None, set()):
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        return closure
    
    @staticmethod
    def epsilon_closure_set(states: Set[NFAState]) -> Set[NFAState]:
        """Compute epsilon closure of a set of states"""
        closure = set()
        for state in states:
            closure.update(NFAToDFAConverter.epsilon_closure(state))
        return closure
    
    @staticmethod
    def move(states: Set[NFAState], symbol: str) -> Set[NFAState]:
        """Move from set of states on symbol"""
        next_states = set()
        for s in states:
            next_states.update(s.transitions.get(symbol, set()))
        return next_states
    
    def convert(self, nfa: NFA) -> DFA:
        """Convert NFA to DFA using subset construction"""
        alphabet = nfa.get_alphabet()
        start_closure = frozenset(self.epsilon_closure(nfa.start))
        
        # Create DFA start state
        dfa_start = DFAState(start_closure)
        dfa_start.is_accept = nfa.accept in start_closure
        dfa = DFA(dfa_start)
        
        # Map from NFA state sets to DFA states
        state_map = {start_closure: dfa_start}
        queue = deque([start_closure])
        
        while queue:
            current_nfa_states = queue.popleft()
            current_dfa_state = state_map[current_nfa_states]
            
            for symbol in alphabet:
                # Move on symbol and compute epsilon closure
                next_nfa_states = self.move(current_nfa_states, symbol)
                if not next_nfa_states:
                    continue
                    
                next_nfa_states = frozenset(self.epsilon_closure_set(next_nfa_states))
                
                # Create new DFA state if needed
                if next_nfa_states not in state_map:
                    next_dfa_state = DFAState(next_nfa_states)
                    next_dfa_state.is_accept = nfa.accept in next_nfa_states
                    state_map[next_nfa_states] = next_dfa_state
                    dfa.add_state(next_dfa_state)
                    queue.append(next_nfa_states)
                else:
                    next_dfa_state = state_map[next_nfa_states]
                
                # Add transition
                current_dfa_state.add_transition(symbol, next_dfa_state)
        
        return dfa

class RegexEngine:
    """Main regex engine that orchestrates the entire process"""
    
    def __init__(self):
        self.parser = RegexParser()
        self.nfa_builder = NFABuilder()
        self.converter = NFAToDFAConverter()
    
    def compile(self, regex: str) -> DFA:
        """Compile regex to DFA"""
        if not regex:
            nfa = self.nfa_builder.empty_nfa()
        else:
            tokens = self.parser.lex(regex)
            tokens = self.parser.insert_concat(tokens)
            rpn = self.parser.to_rpn(tokens)
            nfa = self.nfa_builder.build_nfa(rpn)
        
        return self.converter.convert(nfa)
    
    def match(self, regex: str, string: str) -> bool:
        """Match string against regex"""
        dfa = self.compile(regex)
        return dfa.simulate(string)

def main():
    engine = RegexEngine()
    print("Enter a regex (e.g., 'a(bc)+'): ")
    regex = input().strip()
    print("Enter a string to test (e.g., 'abcbc'): ")
    string = input().strip()
    
    result = engine.match(regex, string)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()