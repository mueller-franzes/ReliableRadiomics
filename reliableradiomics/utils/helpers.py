



def dict_insert(input_dict, data, *keys):
    if len(keys) == 1:
        input_dict[keys[0]] = data

    for key in keys:
        if key in input_dict:
            return dict_insert(input_dict[key], data, *keys[1:])
        else:
            input_dict[key] = {} 
            return dict_insert(input_dict[key], data, *keys[1:])

def dict_append(input_dict, data, *keys):
    if len(keys) == 1:
        if keys[0] in input_dict:
            input_dict[keys[0]].append(data) 
        else:
            input_dict[keys[0]] = [data,] 

    for key in keys:
        if key in input_dict:
            return dict_append(input_dict[key], data, *keys[1:])
        else:
            input_dict[key] = {} 
            return dict_append(input_dict[key], data, *keys[1:])