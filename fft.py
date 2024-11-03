import sys

def set_arguments(args):
    parameters = {
        'mode': 1,
        'image' : 'moonlanding.png'
    }

    i = 1

    while i < len(args):

        if args[i] == '-m':
            if i+1 < len(args) and args[i+1].isdigit():
                parameters['mode'] = int(args[i+1])
                i += 2
            else:
                print(f"ERROR\tIncorrect input syntax: expected integer after argument {args[i]}")
                return None
        
        elif args[i] == '-i':
            if i+1 < len(args):
                parameters['image'] = args[i+1]
                i += 2
            else:
                print(f"ERROR\tIncorrect input syntax: expected image after argument {args[i]}")
                return None
             
    return parameters

if __name__ == "__main__":
    
    parameters = set_arguments(sys.argv)
