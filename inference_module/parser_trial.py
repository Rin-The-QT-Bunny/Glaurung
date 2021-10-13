def to_list(string):
    list_str = []
    for tok in string:
        list_str.append(tok)
    return list_str
def to_string(list):
    string_list = ""
    for item in list:
        string_list += item
    return string_list

def fst_op(program):
    raw = ""
    for item in program:
        if item == "(":
            break
        else:
            raw += item
    return raw

def break_down(ops):
    raw = [""]
    quote_mode = 0
    ind = 0
    for i in range(len(ops)):
        # Read the quote
        if ops[i] == "(" :
            quote_mode += 1
        if ops[i] == ")" :
            quote_mode -= 1
        # Decompose
        if ops[i] == "[":
            quote_mode += 1
        if ops[i] == "]":
            quote_mode -= 1

        if ops[i] == "," and quote_mode == 0:
            ind += 1
            raw.append("")
        else:
            raw[ind] += ops[i]
    return raw

def de_layer(p):
    p = to_list(p)
    for i in range(len(p)):
        
        if p[i] == "(":
            p[i] = ""
            p[-1] = ""
            break
        else:
            p[i] = ""
    return to_string(p)

def Decompose(p):
    sequence = []
    p = p
    def decompose(p):
        ops  = break_down(p)

        for op in ops:
            sequence.append(fst_op(op))
            # Check if it can be decompose again

            if "(" in op:
                delayered = de_layer(op)
                arguments = break_down(delayered)
                
                for arg in arguments:
                    decompose(arg)
    decompose(p)
    return sequence