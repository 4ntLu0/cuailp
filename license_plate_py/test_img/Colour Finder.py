for i in range(127):
    COLOUR = '\033[' + str(i) + 'm'
    print(COLOUR + 'this is', i )