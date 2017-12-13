# isalnum
assert_eq(''.isalnum(), False)
assert_eq('a0 33'.isalnum(), False)
assert_eq('1'.isalnum(), True)
assert_eq('a033'.isalnum(), True)

# isdigit
assert_eq(''.isdigit(), False)
assert_eq(' '.isdigit(), False)
assert_eq('a'.isdigit(), False)
assert_eq('0234325.33'.isdigit(), False)
assert_eq('1'.isdigit(), True)
assert_eq('033'.isdigit(), True)

# isspace
assert_eq(''.isspace(), False)
assert_eq('a'.isspace(), False)
assert_eq('1'.isspace(), False)
assert_eq('\ta\n'.isspace(), False)
assert_eq(' '.isspace(), True)
assert_eq('\t\n'.isspace(), True)

# islower
assert_eq(''.islower(), False)
assert_eq(' '.islower(), False)
assert_eq('1'.islower(), False)
assert_eq('Almost'.islower(), False)
assert_eq('abc'.islower(), True)
assert_eq(' \nabc'.islower(), True)
assert_eq('abc def\n'.islower(), True)
assert_eq('\ta\n'.islower(), True)

# isupper
assert_eq(''.isupper(), False)
assert_eq(' '.isupper(), False)
assert_eq('1'.isupper(), False)
assert_eq('aLMOST'.isupper(), False)
assert_eq('ABC'.isupper(), True)
assert_eq(' \nABC'.isupper(), True)
assert_eq('ABC DEF\n'.isupper(), True)
assert_eq('\tA\n'.isupper(), True)

# istitle
assert_eq(''.istitle(), False)
assert_eq(' '.istitle(), False)
assert_eq('134'.istitle(), False)
assert_eq('almost Correct'.istitle(), False)
assert_eq('1nope Nope Nope'.istitle(), False)
assert_eq('NO Way'.istitle(), False)
assert_eq('T'.istitle(), True)
assert_eq('Correct'.istitle(), True)
assert_eq('Very Correct! Yes\nIndeed1X'.istitle(), True)
assert_eq('1234Ab Ab'.istitle(), True)
assert_eq('\tA\n'.istitle(), True)
