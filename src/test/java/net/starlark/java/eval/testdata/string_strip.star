# strip / lstrip / rstrip

assert_eq("  abc  ".strip(), "abc")
assert_eq("  abc  ".lstrip(), "abc  ")
assert_eq("  abc  ".rstrip(), "  abc")

assert_eq(" \t\n\rabc\t\n\r ".strip(), "abc")
assert_eq("\013\014abc\013\014".strip(), "abc")
assert_eq("\034\035\036\037abc\034\035\036\037".strip(), "abc")

assert_eq("abc".strip(), "abc")
assert_eq("abc".lstrip(), "abc")
assert_eq("abc".rstrip(), "abc")

assert_eq("   ".strip(), "")
assert_eq("   ".lstrip(), "")
assert_eq("   ".rstrip(), "")
assert_eq(" \t\n\r ".strip(), "")

assert_eq("".strip(), "")
assert_eq("".lstrip(), "")
assert_eq("".rstrip(), "")

assert_eq("  a b  c  ".strip(), "a b  c")
assert_eq("  a b  c  ".lstrip(), "a b  c  ")
assert_eq("  a b  c  ".rstrip(), "  a b  c")

assert_eq("   abc".strip(), "abc")
assert_eq("   abc".lstrip(), "abc")
assert_eq("   abc".rstrip(), "   abc")

assert_eq("abc   ".strip(), "abc")
assert_eq("abc   ".lstrip(), "abc   ")
assert_eq("abc   ".rstrip(), "abc")

assert_eq("  abc  ".strip(None), "abc")
assert_eq("  abc  ".lstrip(None), "abc  ")
assert_eq("  abc  ".rstrip(None), "  abc")

# chars: any combination removed, NOT a prefix/suffix
assert_eq("abcba".lstrip("ba"), "cba")
assert_eq("abcbaa".rstrip("ab"), "abc")
assert_eq("aabcbcbaa".strip("ab"), "cbc")

assert_eq("xyzfoozyx".strip("xyz"), "foo")
assert_eq("xyzfoozyx".strip("zyx"), "foo")
assert_eq("xyzfoozyx".strip("yxz"), "foo")

assert_eq("aaaaaXaaaa".strip("a"), "X")
assert_eq("...path...".strip("."), "path")

assert_eq("  abc  ".strip(""), "  abc  ")
assert_eq("aaa".lstrip(""), "aaa")
assert_eq("aaa".rstrip(""), "aaa")

assert_eq("abcba".strip("abc"), "")
assert_eq("abc".lstrip("abc"), "")
assert_eq("abc".rstrip("abc"), "")

assert_eq("aXbXa".strip("a"), "XbX")
assert_eq("abXba".strip("ab"), "X")

assert_eq("  a b c  ".strip(" "), "a b c")

assert_fails(lambda: "abc".strip(1), "got value of type 'int', want 'string or NoneType'")
assert_fails(lambda: "abc".lstrip(1), "got value of type 'int', want 'string or NoneType'")
assert_fails(lambda: "abc".rstrip(1), "got value of type 'int', want 'string or NoneType'")
assert_fails(lambda: "abc".strip(["a"]), "got value of type 'list', want 'string or NoneType'")
assert_fails(lambda: "abc".strip("a", "b"), "accepts no more than 1 positional argument")
