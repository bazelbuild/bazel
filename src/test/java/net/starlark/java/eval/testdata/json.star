# tests of JSON encoding/decoding

assert_eq(dir(json), ["decode", "encode", "encode_indent", "indent"])

# Some of these cases were inspired by github.com/nst/JSONTestSuite.

## json.encode

assert_eq(json.encode(None), "null")
assert_eq(json.encode(True), "true")
assert_eq(json.encode(False), "false")
assert_eq(json.encode(-123), "-123")
assert_eq(json.encode(12345 * 12345 * 12345 * 12345 * 12345 * 12345), "3539537889086624823140625")
assert_eq(json.encode(float(12345*12345*12345*12345*12345*12345)), "3.539537889086625e+24")
assert_eq(json.encode(12.345e67), "1.2345e+68")

assert_eq(json.encode("hello"), '"hello"')
# TODO(adonovan): test more control codes when Starlark/Java has string escapes
assert_eq(json.encode("\t"), r'"\t"')
assert_eq(json.encode("\r"), r'"\r"')
assert_eq(json.encode("\n"), r'"\n"')
assert_eq(json.encode("'"), '"\'"')
assert_eq(json.encode("\""), r'"\""')
assert_eq(json.encode("/"), '"/"')
assert_eq(json.encode("\\"), r'"\\"')
assert_eq(json.encode(""), '""')
assert_eq(json.encode("😹"[:1]), '"�"') # invalid UTF-16 -> replacement char U+FFFD

assert_eq(json.encode([1, 2, 3]), "[1,2,3]")
assert_eq(json.encode((1, 2, 3)), "[1,2,3]")
assert_eq(json.encode(range(3)), "[0,1,2]")  # a built-in iterable
assert_eq(json.encode(dict(x = 1, y = "two")), '{"x":1,"y":"two"}')
assert_eq(json.encode(dict(y = "two", x = 1)), '{"x":1,"y":"two"}')  # key, not insertion, order
assert_eq(json.encode(struct(x = 1, y = "two")), '{"x":1,"y":"two"}')  # a value with fields
assert_eq(json.encode(struct(y = "two", x = 1, )), '{"x":1,"y":"two"}')  # field name order
assert_eq(json.encode(struct(**{'\t': 0})), '{"\\t":0}') # struct keys are escaped too

json.encode(float("NaN")) ### cannot encode non-finite float nan
---
json.encode({1: "two"})  ### dict has int key, want string
---
json.encode(len)  ### cannot encode builtin_function_or_method as JSON
---
json.encode(struct(x = [1, len]))  ### in struct field .x: at list index 1: cannot encode builtin_function_or_method as JSON
---
json.encode(struct(x = [1, {"x": len}]))  ### in struct field .x: at list index 1: in dict key "x": cannot encode builtin_function_or_method as JSON
---
def f(deep):
  for x in range(10000):
    deep = [deep]
  json.encode(deep)  ### nesting depth limit exceeded
f(None)
---
## json.decode

assert_eq(json.decode("null"), None)
assert_eq(json.decode("true"), True)
assert_eq(json.decode("false"), False)
assert_eq(json.decode("-123"), -123)
assert_eq(json.decode("-0"), 0)
assert_eq(json.decode("3539537889086624823140625"), 3539537889086624823140625)
assert_eq(json.decode("3539537889086624823140625.0"), float(3539537889086624823140625))
assert_eq(json.decode("3.539537889086625e+24"), 3.539537889086625e+24)
assert_eq(json.decode("0e+1"), 0)
assert_eq(json.decode("-0.0"), -0.0)
assert_eq(json.decode(
    "-0.000000000000000000000000000000000000000000000000000000000000000000000000000001"),
    -0.000000000000000000000000000000000000000000000000000000000000000000000000000001)
assert_eq(json.decode("1e999"), float("+Inf"))
assert_eq(json.decode("-1e999"), float("-Inf"))
assert_eq(json.decode("5e-1"), 0.5)
assert_eq(json.decode("5e1"), 50.0)
assert_eq(json.decode("5.0e1"), 50.0)
assert_eq(json.decode('[]'), [])
assert_eq(json.decode('[1]'), [1])
assert_eq(json.decode('[1,2,3]'), [1, 2, 3])
assert_eq(json.decode('{"one": 1, "two": 2}'), dict(one=1, two=2))
assert_eq(json.decode('{"foo\\u0000bar": 42}'), {"foo\0bar": 42})
assert_eq(json.decode('"\\ud83d\\ude39\\ud83d\\udc8d"'), "😹💍")
assert_eq(json.decode('"\\u0123"'), 'ģ')
#assert_eq(json.decode('"\x7f"'), "\x7f")
assert_eq(json.decode('\t[\t1,\r2,\n3]\n'), [1, 2, 3]) # whitespace other than ' '
assert_eq(json.decode('\n{\t"a":\r1\t}\n'), {'a': 1}) # same, with dict
assert_eq(json.decode(r'"\\\/\"\n\r\t"'), "\\/\"\n\r\t") # TODO(adonovan): test \b\f when Starlark/Java supports them

# We accept UTF-16 strings that have been arbitrarily truncated,
# as many Java and JavaScript programs emit them.
assert_eq(json.decode('"<' + "😹"[:1] + '>"'), '<' + "😹"[:1] + '>')

# Lists and dicts are mutable.
mutable = json.decode('[{}]')
mutable.append(3)
mutable[0][1] = 2
assert_eq(str(mutable), "[{1: 2}, 3]")

json.decode('truefalse') ### at offset 4, unexpected character "f" after value
---
json.decode('"abc') ### unclosed string literal
---
json.decode('"ab\\gc"') ### invalid escape '\\g'
---
json.decode("'abc'") ### unexpected character "'"
---
json.decode("1.2.3") ### invalid number: 1.2.3
---
json.decode(".5e1") ### unexpected character "\."
---
json.decode("+1") ### unexpected character "\+"
---
json.decode("-abc") ### invalid number: -
---
json.decode("-") ### invalid number: -
---
json.decode("-00") ### invalid number: -00
---
json.decode("00") ### invalid number: 00
---
json.decode("--1") ### invalid number: --1
---
json.decode("-+1") ### invalid number: -\+1
---
json.decode("1e1e1") ### invalid number: 1e1e1
---
json.decode("5.") ### invalid number: 5.
---
json.decode(".5") ### unexpected character "."
---
json.decode("5.e1") ### invalid number: 5.e1
---
json.decode("5e") ### invalid number: 5e
---
json.decode("5ee1") ### invalid number
---
json.decode("0123") ### invalid number: 0123
---
json.decode("000.123") ### invalid number: 000.123
---
json.decode("-0123") ### invalid number: -0123
---
json.decode("-000.123") ### invalid number: -000.123
---
json.decode("0x123") ### unexpected character "x" after value
---
json.decode('[1, 2 ') ### unexpected end of file
---
json.decode('[1, 2, ') ### unexpected end of file
---
json.decode('[1, 2, ]') ### unexpected character "]"
---
json.decode('[1, 2, }') ### unexpected character "}"
---
json.decode('[1, 2}') ### got "}", want ',' or ']'
---
json.decode('{"one": 1') ### unexpected end of file
---
json.decode('{"one" 1') ### after object key, got "1", want ':'
---
json.decode('{"one": 1 "two": 2') ### in object, got "\\"", want ',' or '}'
---
json.decode('{"x": 1, "x": 2}') ### object has duplicate key: "x"
---
json.decode('{1:2}') ### got int for object key, want string
---
json.decode('{"one": 1,') ### unexpected end of file
---
json.decode('{"one": 1, }') ###  unexpected character "}"
---
json.decode('{"one": 1]') ### in object, got "]", want ',' or '}'
---
json.decode('[' * 10000) ### nesting depth limit exceeded
---
# Unescaped control codes (even tabs) are forbidden in strings.
json.decode('"\t"') ### invalid character '\\x09' in string literal
---
json.decode('"\\u123"') ### incomplete \\uXXXX escape
---
json.decode('"\\u123') ### incomplete \\uXXXX escape
---
json.decode('"\\u1') ### incomplete \\uXXXX escape
---

def codec(x):
    return json.decode(json.encode(x))

# string round-tripping
strings = [
    "\t",
    "'",
    "\"",
    "/",
    "\\",
    "",
    "😿", # U+1F63F CRYING_CAT_FACE
    "🐱‍👤", # CAT FACE + ZERO WIDTH JOINER + BUST IN SILHOUETTE
]
assert_eq(codec(strings), strings)

# # codepoints is a string with every valid non-surrogate 16-bit code point.
# TODO(adonovan): enable once %c is supported.
# codepoints = ''.join(['%c' % c for c in range(65536) if c < 0xD800 or d > 0xDFFF])
# assert_eq(codec(codepoints), codepoints)

# number round-tripping
numbers = [
     0, 1, -1, +1,
     0.0, -0.0, 1.0, -1.0, +1.0, 1e6, -1e6, 1.23e45, -1.23e-45,
     3539537889086624823140625,
     float(3539537889086624823140625),
]
assert_eq(codec(numbers), numbers)

## json.indent

assert_eq(json.indent('  1  '), '1')
assert_eq(json.indent(' \n \r \t [ 1 , \r null \n , true , false , [ ], { }, "foo\\u1234" ]'), """\
[
	1,
	null,
	true,
	false,
	[],
	{},
	"foo\\u1234"
]""")

s = json.encode(dict(x = 1, y = ["one", "two", [], {}]))

assert_eq(json.indent(s), '''\
{
	"x": 1,
	"y": [
		"one",
		"two",
		[],
		{}
	]
}''')

assert_eq(json.encode_indent(dict(x = 1, y = ["one", "two", [], {}])), '''\
{
	"x": 1,
	"y": [
		"one",
		"two",
		[],
		{}
	]
}''')

assert_eq(json.encode_indent(dict(x = 1, y = ["one", "two", [], {}]), prefix='¶', indent='–––'), '''\
{
¶–––"x": 1,
¶–––"y": [
¶––––––"one",
¶––––––"two",
¶––––––[],
¶––––––{}
¶–––]
¶}''')

assert_eq(json.decode(json.indent(s)), {"x": 1, "y": ["one", "two", [], {}]})

assert_eq(json.indent(s, prefix='¶', indent='–––'), '''\
{
¶–––"x": 1,
¶–––"y": [
¶––––––"one",
¶––––––"two",
¶––––––[],
¶––––––{}
¶–––]
¶}''')

# The current implementation may produce garbage output for garbage input.
# TODO(adonovan): fix that.
assert_eq(json.indent('[n!@#,t$%^,f&*(),-e1+9,"\\k\\upqrs"]'),
          """[
	n!@#,
	t$%^,
	f&*(),
	-e1+9,
	"\\k\\upqrs\"
]""")

json.indent('[' * 10000 + ']' * 10000) # no depth error: indentation is nonrecursive

json.indent('[') ### unexpected end of file
---
json.indent('"') ### input is not valid JSON
---

# assert.fails(lambda: json.indent("!@#$%^& this is not json"), 'invalid character')
