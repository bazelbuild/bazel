# split
assert_eq("h i".split(" "), ["h", "i"])
assert_eq("h i p".split(" "), ["h", "i", "p"])
assert_eq("a,e,i,o,u".split(",", 2), ["a", "e", "i,o,u"])
assert_eq("  1  2  3  ".split(" "), ["", "", "1", "", "2", "", "3", "", ""])
assert_eq("a.b.c".split(sep=".", maxsplit=1), ["a", "b.c"])

# rsplit
assert_eq("abcdabef".rsplit("ab"), ["", "cd", "ef"])
assert_eq("google_or_gogol".rsplit("go"), ["", "ogle_or_", "", "l"])
assert_eq("a.b.c".rsplit(sep=".", maxsplit=1), ["a.b", "c"])

assert_fails(lambda: "abc".rsplit(""), "Empty separator")

# rsplit regex
assert_eq("foo/bar.lisp".rsplit("."), ["foo/bar", "lisp"])
assert_eq("foo/bar.?lisp".rsplit(".?"), ["foo/bar", "lisp"])
assert_eq("fwe$foo".rsplit("$"), ["fwe", "foo"])
assert_eq("windows".rsplit("\\w"), ["windows"])

# rsplit no match
assert_eq("".rsplit("o"), [""])
assert_eq("google".rsplit("x"), ["google"])

# rsplit separator
assert_eq("xxxxxx".rsplit("x"), ["", "", "", "", "", "", ""])
assert_eq("xxxxxx".rsplit("x", 1), ["xxxxx", ""])
assert_eq("xxxxxx".rsplit("x", 2), ["xxxx", "", ""])
assert_eq("xxxxxx".rsplit("x", 3), ["xxx", "", "", ""])
assert_eq("xxxxxx".rsplit("x", 4), ["xx", "", "", "", ""])
assert_eq("xxxxxx".rsplit("x", 5), ["x", "", "", "", "", ""])
assert_eq("xxxxxx".rsplit("x", 6), ["", "", "", "", "", "", ""])
assert_eq("xxxxxx".rsplit("x", 7), ["", "", "", "", "", "", ""])

# split max split
assert_eq("google".rsplit("o"), ["g", "", "gle"])
assert_eq("google".rsplit("o"), ["g", "", "gle"])
assert_eq("google".rsplit("o", 1), ["go", "gle"])
assert_eq("google".rsplit("o", 2), ["g", "", "gle"])
assert_eq("google".rsplit("o", 3), ["g", "", "gle"])
assert_eq("ogooglo".rsplit("o"), ["", "g", "", "gl", ""])
assert_eq("ogooglo".rsplit("o", 1), ["ogoogl", ""])
assert_eq("ogooglo".rsplit("o", 2), ["ogo", "gl", ""])
assert_eq("ogooglo".rsplit("o", 3), ["og", "", "gl", ""])
assert_eq("ogooglo".rsplit("o", 4), ["", "g", "", "gl", ""])
assert_eq("ogooglo".rsplit("o", 5), ["", "g", "", "gl", ""])
assert_eq("google".rsplit("google"), ["", ""])
assert_eq("google".rsplit("google", 1), ["", ""])
assert_eq("google".rsplit("google", 2), ["", ""])

# zero/negative split
assert_eq("a-b-c".split("-", 0), ["a-b-c"])
assert_eq("a-b-c".rsplit("-", 0), ["a-b-c"])
assert_eq("a-b-c".split("-", -1), ["a", "b", "c"])
assert_eq("a-b-c".rsplit("-", -2), ["a", "b", "c"])
