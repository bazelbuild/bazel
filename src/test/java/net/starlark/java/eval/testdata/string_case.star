# str.lower / str.upper

assert_eq("HELLO".lower(), "hello")
assert_eq("hello".lower(), "hello")
assert_eq("HeLlO".lower(), "hello")
assert_eq("".lower(), "")
assert_eq("Hello World!".lower(), "hello world!")

assert_eq("hello".upper(), "HELLO")
assert_eq("HELLO".upper(), "HELLO")
assert_eq("HeLlO".upper(), "HELLO")
assert_eq("".upper(), "")
assert_eq("Hello World!".upper(), "HELLO WORLD!")

assert_eq("123 ABC abc!?".lower(), "123 abc abc!?")
assert_eq("123 ABC abc!?".upper(), "123 ABC ABC!?")
assert_eq("  ".lower(), "  ")
assert_eq("\t\n".upper(), "\t\n")

# ASCII-only: non-ASCII letters pass through unchanged.
assert_eq("café".upper(), "CAFé")
assert_eq("CAFÉ".lower(), "cafÉ")
assert_eq("naïve".upper(), "NAïVE")
assert_eq("ÄÖÜß".lower(), "ÄÖÜß")
assert_eq("äöüß".upper(), "äöüß")

assert_eq(type("Hi".lower()), "string")
assert_eq(type("Hi".upper()), "string")

assert_eq("abcxyz".upper().lower(), "abcxyz")
assert_eq("ABCXYZ".lower().upper(), "ABCXYZ")

assert_fails(lambda: "abc".lower("x"), "lower\\(\\) got unexpected positional argument")
assert_fails(lambda: "abc".upper("x"), "upper\\(\\) got unexpected positional argument")
