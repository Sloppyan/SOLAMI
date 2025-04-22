


def func_validate(func, gpt_response, example): ############
    try: 
        func(gpt_response, example)
        return True
    except:
        return False 
    
def compare_types(obj1, obj2):
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, (list, tuple)):
        if len(obj1) == 0 or len(obj2) == 0:
            return False
        for item in obj1:
            if not compare_types(item, obj2[0]):
                return False
    elif isinstance(obj1, dict):
        if len(obj1) != len(obj2):
            return False
        for key1, key2 in zip(sorted(obj1.keys()), sorted(obj2.keys())):
            # TODO check to make dict keys are the same
            if key1 != key2:
                return False
            if not compare_types(key1, key2) or not compare_types(obj1[key1], obj2[key2]):
                return False
                
    return True

