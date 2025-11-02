def parse_args(argv):
        message = None
        top_k = None
        max_distance = None

        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg == '-k' and i + 1 < len(argv):
                try:
                    top_k = int(argv[i+1])
                except ValueError:
                    top_k = None
                i += 2
            elif arg == '-d' and i + 1 < len(argv):
                try:
                    max_distance = float(argv[i+1])
                except ValueError:
                    max_distance = None
                i += 2
            elif not arg.startswith('-') and message is None:
                message = arg
                i += 1
            else:
                i += 1

        return message, top_k, max_distance
    
print(parse_args(['มรรค', '-k', '5', '-d', '0.7']))