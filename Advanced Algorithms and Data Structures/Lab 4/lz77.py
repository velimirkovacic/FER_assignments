def LZ77_encode(input:str, dict_len:int=4, la_len:int=4) -> list:
    """
    Function implements the LZ77 encoding algorithm.
    Args:
        input (str): Input string to the algorithm.
        dict_len (int): Lenght of the dictionary.
        la_len (int): Lenght of the lookahead.
    Returns:
        list: List where the first element is the first character and subsequent elements are tuple (i,j,k)
    """
    enc = [input[0]]

    dict_la = "".join([input[0]] * dict_len) + input[:la_len]

    input = input[la_len + 0:]


    while len(dict_la) - dict_len > 0:
        max_ind = 0
        max_plen = 0
        
        for i in range(dict_len):
            if dict_la[i] == dict_la[dict_len]:
                p_len = 1
                while p_len < len(dict_la) - dict_len and dict_la[i + p_len] == dict_la[dict_len + p_len]:
                    p_len += 1
                if p_len >= max_plen:
                    max_plen = p_len
                    max_ind = i
        if max_plen == len(dict_la) - dict_len:
            enc += [(dict_len - max_ind - 1, max_plen - 1, dict_la[-1])]
        elif max_plen == 0:
            enc += [(0, 0, dict_la[dict_len])]
            max_plen = 1
        else:
            enc += [(dict_len - max_ind - 1, max_plen, dict_la[dict_len + max_plen])]
            max_plen += 1
        

        dict_la = dict_la[max_plen:]
        dict_la += input[:max_plen]
        input = input[max_plen:]

    return enc

