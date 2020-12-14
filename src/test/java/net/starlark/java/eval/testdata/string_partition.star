assert_eq("lawl".partition("a"), ("l", "a", "wl"))
assert_eq("lawl".rpartition("a"), ("l", "a", "wl"))
assert_eq("google".partition("o"), ("g", "o", "ogle"))
assert_eq("google".rpartition("o"), ("go", "o", "gle"))
assert_eq("xxx".partition("x"), ("", "x", "xx"))
assert_eq("xxx".rpartition("x"), ("xx", "x", ""))
assert_eq("".partition("a"), ("", "", ""))
assert_eq("".rpartition("a"), ("", "", ""))

# no match
assert_eq("google".partition("x"), ("google", "", ""))
assert_eq("google".rpartition("x"), ("", "", "google"))

# at word boundaries
assert_eq("goog".partition("g"), ("", "g", "oog"))
assert_eq("goog".rpartition("g"), ("goo", "g", ""))
assert_eq("plex".partition("p"), ("", "p", "lex"))
assert_eq("plex".rpartition("p"), ("", "p", "lex"))
assert_eq("plex".partition("x"), ("ple", "x", ""))
assert_eq("plex".rpartition("x"), ("ple", "x", ""))

assert_eq("google".partition("oog"), ("g", "oog", "le"))
assert_eq("google".rpartition("oog"), ("g", "oog", "le"))
assert_eq("lolgooglolgooglolgooglol".partition("goog"), ("lol", "goog", "lolgooglolgooglol"))
assert_eq("lolgooglolgooglolgooglol".rpartition("goog"), ("lolgooglolgooglol", "goog", "lol"))

# full string
assert_eq("google".partition("google"), ("", "google", ""))
assert_eq("google".rpartition("google"), ("", "google", ""))

assert_fails(lambda: "google".partition(1), "got value of type 'int', want 'string'")
assert_fails(lambda: "google".rpartition(1), "got value of type 'int', want 'string'")
assert_fails(lambda: "google".partition(""), "empty separator")
assert_fails(lambda: "google".rpartition(""), "empty separator")
assert_fails(lambda: "google".partition(), "missing 1 required positional argument")
assert_fails(lambda: "google".rpartition(), "missing 1 required positional argument")
