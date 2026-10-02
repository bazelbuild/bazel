"""Tests of TOML encoding and decoding."""

assert_eq(dir(toml), ["decode", "encode"])

def assert_round_trip(value):
    assert_eq(toml.decode(toml.encode(value)), value)

## toml.decode

assert_eq(toml.decode('title = "TOML Example"'), {"title": "TOML Example"})
assert_eq(toml.decode("bool = true"), {"bool": True})
assert_eq(toml.decode("date = 1979-05-27T07:32:00Z"), {"date": "1979-05-27T07:32:00Z"})
assert_eq(toml.decode("dates = [1979-05-27, 1980-06-28]"), {"dates": ["1979-05-27", "1980-06-28"]})
assert_eq(toml.decode('mixed = ["a", 1979-05-27, "b"]'), {"mixed": ["a", "1979-05-27", "b"]})
assert_eq(toml.decode('[event]\nname = "meeting"\nwhen = 2024-01-15T10:00:00Z'), {"event": {"name": "meeting", "when": "2024-01-15T10:00:00Z"}})
assert_eq(toml.decode("float = 42.42"), {"float": 42.42})
assert_eq(toml.decode("number = 42"), {"number": 42})
assert_eq(toml.decode("temp_targets = { cpu = 79.5, case = 72.0 }"), {"temp_targets": {"cpu": 79.5, "case": 72.0}})
assert_eq(toml.decode("nested_arrays_of_ints = [ [ 1, 2 ], [3, 4, 5] ]"), {"nested_arrays_of_ints": [[1, 2], [3, 4, 5]]})
assert_eq(toml.decode("title =", default = True), True)
assert_eq(toml.decode(""), {})

dict_example = """
[servers]

# first one
[servers.alpha]
ip = "10.0.0.1"
role = "frontend"

# second one
[servers.beta]
ip = "10.0.0.2"
role = "backend"
"""
assert_eq(toml.decode(dict_example), {"servers": {"alpha": {"ip": "10.0.0.1", "role": "frontend"}, "beta": {"ip": "10.0.0.2", "role": "backend"}}})

assert_fails(lambda: toml.decode("title = "), "Unexpected end of input")
assert_fails(lambda: toml.decode("title = ["), "Unexpected end of input")
assert_fails(lambda: toml.decode("["), "Unexpected end of input")
assert_fails(lambda: toml.decode("{"), "Unexpected")

nested_array_of_tables = """
[[fruits]]
name = "apple"

[[fruits.varieties]]
name = "red delicious"

[[fruits.varieties]]
name = "granny smith"

[[fruits]]
name = "banana"

[[fruits.varieties]]
name = "plantain"
"""
assert_eq(toml.decode(nested_array_of_tables), {
    "fruits": [
        {
            "name": "apple",
            "varieties": [
                {"name": "red delicious"},
                {"name": "granny smith"},
            ],
        },
        {
            "name": "banana",
            "varieties": [
                {"name": "plantain"},
            ],
        },
    ],
})

## toml.encode

assert_eq(toml.encode({"x": 1, "y": "two"}), """\
x = 1
y = 'two'
""")
assert_eq(toml.encode({"title": "TOML Example"}), """\
title = 'TOML Example'
""")
assert_eq(toml.encode({"bool": True, "number": 42}), """\
bool = true
number = 42
""")
assert_eq(toml.encode({"count": -123}), """\
count = -123
""")
assert_eq(toml.encode({"ratio": 12.345}), """\
ratio = 12.345
""")
assert_eq(toml.encode({"name": "hello"}), """\
name = 'hello'
""")
assert_eq(toml.encode({"empty": ""}), """\
empty = ''
""")

# Lists and tuples become TOML arrays
assert_eq(toml.encode({"numbers": [1, 2, 3]}), """\
numbers = [
  1,
  2,
  3,
]
""")
assert_eq(toml.encode({"tuple_val": (1, 2, 3)}), """\
tuple_val = [
  1,
  2,
  3,
]
""")
assert_eq(toml.encode({"strings": ["a", "b", "c"]}), """\
strings = [
  'a',
  'b',
  'c',
]
""")
assert_eq(toml.encode({"empty_array": []}), """\
empty_array = []
""")

# Sets retain their deterministic iteration order.
assert_eq(toml.encode({"items": set([3, 1, 2])}), """\
items = [
  3,
  1,
  2,
]
""")

# Nested structures
assert_eq(toml.encode({"nested": {"x": 1, "y": 2}}), """\
[nested]
x = 1
y = 2
""")

# Scalars must come before tables to avoid being associated with the wrong table
assert_eq(toml.encode({"file": {"path": "a.txt"}, "other": True}), """\
other = true

[file]
path = 'a.txt'
""")
assert_eq(toml.encode({"nested1": {"nested2": {"x": 1, "y": 2}}}), """\
[nested1.nested2]
x = 1
y = 2
""")
assert_eq(toml.encode({"arrays": [[1, 2], [3, 4, [5, 6]]]}), """\
arrays = [
  [
    1,
    2,
  ],
  [
    3,
    4,
    [
      5,
      6,
    ],
  ],
]
""")

# Array with nested tables uses [[array]] syntax
assert_eq(toml.encode({"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}), """\
[[points]]
x = 1
y = 2

[[points]]
x = 3
y = 4
""")
assert_eq(toml.encode({"points": [{"x": {"y": 1, "z": 2}}]}), """\
[[points]]

[points.x]
y = 1
z = 2
""")

assert_eq(toml.encode({"foo": {"bar": {"baz": {"qux": [{"x": 1}]}}}}), """\
[[foo.bar.baz.qux]]
x = 1
""")

# Array of tables with nested arrays of primitives
assert_eq(toml.encode({"items": [{"flags": ["a", "b", "c"]}]}), """\
[[items]]
flags = [
  'a',
  'b',
  'c',
]
""")

# Array of tables with nested tables
assert_eq(toml.encode({"data": [{"outer": {"inner": [1, 2, 3]}}]}), """\
[[data]]

[data.outer]
inner = [
  1,
  2,
  3,
]
""")

# Nested arrays of tables
assert_eq(toml.encode({"config": [{"groups": [{"items": ["x", "y"]}]}]}), """\
[[config]]

[[config.groups]]
items = [
  'x',
  'y',
]
""")

# Structs
assert_eq(toml.encode(struct(x = 1, y = "two")), """\
x = 1
y = 'two'
""")
assert_eq(toml.encode(struct(title = "Example", count = 5)), """\
count = 5
title = 'Example'
""")
assert_eq(toml.encode({"foo": struct(title = "Example", count = 5)}), """\
[foo]
count = 5
title = 'Example'
""")

# Empty dict
assert_eq(toml.encode({}), "")
assert_eq(toml.encode({"foo": {}}), """\
[foo]
""")

# Mixed nested types
assert_eq(
    toml.encode({"database": {"host": "localhost", "port": 5432, "enabled": True}}),
    """\
[database]
host = 'localhost'
port = 5432
enabled = true
""",
)

# None-valued table fields are omitted, but None in arrays is rejected.
assert_fails(lambda: toml.encode({"value": (None, None)}), "at tuple index 0: cannot encode NoneType")
assert_eq(toml.encode({"val": None}), "")
assert_eq(toml.encode({"a": 1, "b": None, "c": 3}), """\
a = 1
c = 3
""")
assert_fails(lambda: toml.encode({"items": [1, None, 3]}), "at list index 1: cannot encode NoneType")
assert_eq(toml.encode({"nested": {"x": 1, "y": None}}), """\
[nested]
x = 1
""")

# TOML supports non-finite floats.
assert_eq(toml.encode({"x": float("NaN")}), "x = nan\n")
assert_eq(toml.encode({"x": float("+Inf")}), "x = +inf\n")
assert_eq(toml.encode({"x": float("-Inf")}), "x = -inf\n")

# Error cases - non-dict top-level values
assert_fails(lambda: toml.encode([1, 2, 3]), "TOML encode requires a dict, struct, or native provider at the top level")
assert_fails(lambda: toml.encode((1, 2, 3)), "TOML encode requires a dict, struct, or native provider at the top level")
assert_fails(lambda: toml.encode("hello"), "TOML encode requires a dict, struct, or native provider at the top level")
assert_fails(lambda: toml.encode(42), "TOML encode requires a dict, struct, or native provider at the top level")

# Error cases - non-string dict keys
assert_fails(lambda: toml.encode({1: "two"}), "dict has int key, want string")

# Error cases - non-serializable types
assert_fails(lambda: toml.encode({"fn": len}), "cannot encode builtin_function_or_method as TOML")
assert_fails(
    lambda: toml.encode(struct(x = [1, len])),
    "in struct field .x: at list index 1: cannot encode builtin_function_or_method as TOML",
)
assert_fails(
    lambda: toml.encode(struct(x = [1, {"y": len}])),
    "in struct field .x: at list index 1: in dict key \"y\": cannot encode builtin_function_or_method as TOML",
)

# Nesting depth limit
def f(deep):
    for _ in range(100000):
        deep = [deep]
    toml.encode({"data": deep})

assert_fails(lambda: f(1), "nesting depth limit exceeded")

# Round-trip test: encode then decode should give back original value
assert_round_trip({
    "title": "Config",
    "count": 42,
    "enabled": True,
    "ratio": 3.14,
    "items": ["a", "b", "c"],
})

## String escaping in encode

assert_eq(toml.encode({"msg": "it's fine"}), """\
msg = "it's fine"
""")
assert_eq(toml.encode({"msg": 'say "hello"'}), """\
msg = 'say "hello"'
""")
assert_eq(toml.encode({"msg": "it's \"complex\""}), """\
msg = "it's \\"complex\\""
""")
assert_eq(toml.encode({"path": "C:\\Users\\name"}), """\
path = "C:\\\\Users\\\\name"
""")
assert_eq(toml.encode({"text": "line1\nline2"}), """\
text = "line1\\nline2"
""")
assert_eq(toml.encode({"text": "col1\tcol2"}), """\
text = "col1\\tcol2"
""")
assert_eq(toml.encode({"text": "line1\rline2"}), """\
text = "line1\\rline2"
""")

assert_eq(toml.encode({"text": "café"}), """\
text = 'café'
""")
assert_eq(toml.encode({"emoji": "hello 🎉"}), """\
emoji = 'hello 🎉'
""")

# Array of tables with empty tables uses [[array]] syntax
assert_eq(toml.encode({"items": [{}, {"x": 1}, {}]}), """\
[[items]]

[[items]]
x = 1

[[items]]
""")

assert_eq(toml.encode({"a": {"b": {"c": {"d": {"e": 1}}}}}), """\
[a.b.c.d]
e = 1
""")

# Mixed arrays and tables at same level
assert_eq(toml.encode({"config": {"items": [1, 2], "name": "test"}}), """\
[config]
items = [
  1,
  2,
]
name = 'test'
""")

assert_eq(toml.encode(struct()), "")

assert_eq(toml.encode({"outer": struct()}), """\
[outer]
""")

assert_eq(toml.encode({"z": 1, "a": 2, "m": 3}), """\
z = 1
a = 2
m = 3
""")

assert_eq(toml.encode({"foo": [
    {"name": "bar", "id": 1},
    {"name": "baz", "id": 2},
]}), """\
[[foo]]
name = 'bar'
id = 1

[[foo]]
name = 'baz'
id = 2
""")

## Keys with special characters need quoting

# Keys with spaces
assert_eq(toml.encode({"my key": "value"}), """\
'my key' = 'value'
""")

# Keys with dots (would be confused with nested tables)
assert_eq(toml.encode({"my.key": "value"}), """\
'my.key' = 'value'
""")

# Keys with special characters
assert_eq(toml.encode({"key@domain": "value"}), """\
'key@domain' = 'value'
""")

# Keys with quotes need escaping
assert_eq(toml.encode({"it's a key": "value"}), """\
"it's a key" = 'value'
""")

# Empty key
assert_eq(toml.encode({"": "value"}), """\
'' = 'value'
""")

# Bare keys with allowed characters (underscore and dash)
assert_eq(toml.encode({"MY_key-name": "value"}), """\
MY_key-name = 'value'
""")

# Nested table with special key
assert_eq(toml.encode({"my table": {"x": 1}}), """\
['my table']
x = 1
""")

# Deeply nested with special keys
assert_eq(toml.encode({"a.b": {"c d": {"e": 1}}}), """\
['a.b'.'c d']
e = 1
""")

# Array of tables with special key
assert_eq(toml.encode({"my items": [{"x": 1}, {"x": 2}]}), """\
[['my items']]
x = 1

[['my items']]
x = 2
""")

# Inline table with special keys
assert_eq(toml.encode({"items": [{"my key": 1, "other key": 2}]}), """\
[[items]]
'my key' = 1
'other key' = 2
""")

# Round-trip with special keys
assert_round_trip({
    "my key": "value1",
    "key.with.dots": "value2",
})

# All date/time types are strings unless a callback transforms them.
assert_eq(toml.decode("date = 1979-05-27"), {"date": "1979-05-27"})
assert_eq(toml.decode("time = 07:32:01.123"), {"time": "07:32:01.123"})
assert_eq(toml.decode("local = 1979-05-27T07:32:01"), {"local": "1979-05-27T07:32:01"})
assert_eq(toml.decode("offset = 1979-05-27T07:32:01-07:00"), {"offset": "1979-05-27T07:32:01-07:00"})

# Zero seconds are kept and +00:00 becomes Z, so decoded dates and times are spelled as TOML
# spells them and can be re-parsed as dates.
assert_eq(
    toml.decode("t = 07:32:00\nd = 1979-05-27T07:32:00\no = 1979-05-27T07:32:00+00:00\nf = 07:32:00.5"),
    {"t": "07:32:00", "d": "1979-05-27T07:32:00", "o": "1979-05-27T07:32:00Z", "f": "07:32:00.5"},
)

def test_dates_reparse():
    for date in ["1979-05-27T07:32:00Z", "1979-05-27T07:32:00.5-07:00", "1979-05-27T07:32:00", "1979-05-27", "07:32:00"]:
        assert_eq(toml.decode("x = " + toml.decode("x = " + date)["x"]), {"x": date})

test_dates_reparse()

assert_eq(toml.decode("date = 1979-05-27", decode_date = None), {"date": "1979-05-27"})
assert_eq(toml.decode("date = 1979-05-27", decode_date = lambda date: date.split("-")), {"date": ["1979", "05", "27"]})
custom_date = toml.decode("date = 1979-05-27", decode_date = lambda date: struct(custom_date_type = date))["date"]
assert_eq(type(custom_date), "struct")
assert_eq(custom_date.custom_date_type, "1979-05-27")
assert_eq(toml.decode("dates = [1979-05-27, 1980-06-28]", decode_date = lambda date: date[:4]), {"dates": ["1979", "1980"]})
assert_eq(toml.decode("[event]\ndate = 1979-05-27", decode_date = lambda date: {"iso": date}), {"event": {"date": {"iso": "1979-05-27"}}})
assert_eq(toml.decode('values = [1, 1979-05-27, "last"]', decode_date = lambda date: None), {"values": [1, None, "last"]})
assert_eq(toml.decode("date = 1979-05-27", decode_date = lambda date: None), {"date": None})
assert_eq(toml.decode('date = "1979-05-27"', decode_date = lambda date: fail("unexpected callback")), {"date": "1979-05-27"})
assert_fails(lambda: toml.decode("date = 1979-05-27", decode_date = lambda date: fail("dates disabled")), "dates disabled")
assert_fails(lambda: toml.decode("date = 1979-05-27", default = {}, decode_date = lambda date: fail("dates disabled")), "dates disabled")
assert_fails(lambda: toml.decode("", decode_date = 1), "decode_date.*got value of type 'int'")
assert_eq(toml.decode("broken =", default = None), None)
assert_eq(toml.decode("broken =", default = 42, decode_date = lambda date: fail("unexpected callback")), 42)

# Input that cannot be decoded for reasons other than a syntax error also yields the default.
assert_fails(lambda: toml.decode("a = 1979-05-27T07:32:00-07'00"), "TOML decode error")
assert_eq(toml.decode("a = 1979-05-27T07:32:00-07'00", default = None), None)
assert_fails(lambda: toml.decode("x = " + "[" * 200000), "nesting depth limit exceeded")
assert_eq(toml.decode("x = " + "[" * 200000, default = "deep"), "deep")
assert_fails(lambda: toml.decode("x = 9223372036854775808"), "Integer is too large")
assert_eq(toml.decode("x = 9223372036854775808", default = None), None)

def test_invalid_utf8():
    # In UTF-8 byte-string mode, "\377" is a lone byte that is not valid UTF-8.
    if _utf8_byte_strings:
        assert_fails(lambda: toml.decode("x = '\377'"), "TOML decode error: input is not valid UTF-8")
        assert_eq(toml.decode("x = '\377'", default = None), None)
    else:
        assert_round_trip({"x": "\377"})

test_invalid_utf8()

def test_unpaired_surrogate():
    # In UTF-16 mode, indexing splits a surrogate pair; TOML cannot represent the halves.
    if not _utf8_byte_strings:
        assert_eq(toml.encode({"x": "🎉"[0], "y": "🎉"}), "x = \"�\"\ny = '🎉'\n")

test_unpaired_surrogate()

# Every decoded container belongs to the caller's mutability.
mutable = toml.decode("[table]\nitems = [{x = 1}]")
mutable["extra"] = True
mutable["table"]["extra"] = True
mutable["table"]["items"].append(2)
mutable["table"]["items"][0]["x"] = 3
assert_eq(mutable, {"extra": True, "table": {"extra": True, "items": [{"x": 3}, 2]}})

# Unicode escapes and raw Unicode agree in both Starlark string modes.
assert_eq(toml.decode('"\\u00E9" = "\\U0001F389"'), {"é": "🎉"})
assert_eq(toml.decode('"é" = "🎉"'), {"é": "🎉"})
assert_round_trip({"é": "🎉"})
assert_eq(toml.encode({"delete": "\177"}), 'delete = "\\u007F"\n')
assert_round_trip({"\177": "\177"})

# All TOML spellings of non-finite numbers decode; encoding preserves their value.
def test_nonfinite():
    for literal in ["inf", "+inf", "-inf", "nan", "+nan", "-nan"]:
        value = toml.decode("x = " + literal)["x"]
        assert_eq(str(toml.decode(toml.encode({"x": value}))["x"]), str(value))

test_nonfinite()

# Sets do not need comparable elements, and retain insertion order.
assert_eq(toml.decode(toml.encode({"values": set(["last", 2, True])})), {"values": ["last", 2, True]})
assert_eq(toml.encode({"values": set(["z", "a"])}), "values = [\n  'z',\n  'a',\n]\n")

# Scalars precede all tables, while tables and arrays of tables retain their relative order.
interleaved = {"a": {"b": {"x": 1}, "c": [{"y": 2}], "d": {"z": 3}, "scalar": 4}}
assert_eq(toml.encode(interleaved), """\
[a]
scalar = 4

[a.b]
x = 1

[[a.c]]
y = 2

[a.d]
z = 3
""")
assert_round_trip(interleaved)

# Empty tables, including those left after omitting None fields, need headers.
assert_eq(toml.encode({"a": {"b": {}}}), "[a.b]\n")
assert_eq(toml.encode({"a": {"b": None}}), "[a]\n")
assert_eq(toml.encode(struct(x = None)), "")
assert_round_trip({"items": [{}, {"x": {}}, {}]})
assert_round_trip({"a": {"b": {}}, "c": {}})

# Inline tables must stay on one line, including their nested arrays and tables.
mixed = {"items": [1, {"nested": {"values": [2, 3]}, "absent": None}]}
assert_eq(toml.encode(mixed), """\
items = [
  1,
  { nested = { values = [2, 3] } },
]
""")
assert_eq(toml.decode(toml.encode(mixed)), {"items": [1, {"nested": {"values": [2, 3]}}]})
assert_round_trip({"items": [[{"x": [1, 2]}, {}]]})
assert_eq(toml.decode(toml.encode({"items": [struct(x = 1), struct(x = 2)]})), {"items": [{"x": 1}, {"x": 2}]})
assert_eq(toml.decode(toml.encode({"items": [1, struct(x = 2)]})), {"items": [1, {"x": 2}]})

# A reused table path cannot leak into siblings or later elements of an array of tables.
paths = {"a.b": [{"c d": {"x": 1}}, {"c d": {"y": 2}}], "sibling": {"z": 3}}
assert_round_trip(paths)
assert_round_trip(toml.decode(nested_array_of_tables))

# None in an array cannot silently shift element positions.
assert_fails(lambda: toml.encode({"items": [{"x": 1}, None]}), "at list index 1: cannot encode NoneType")
assert_fails(lambda: toml.encode({"items": [1, {"x": [None]}]}), 'in dict key "items": at list index 1: in dict key "x": at list index 0: cannot encode NoneType')
assert_fails(lambda: toml.encode(None), "TOML encode requires a dict, struct, or native provider")
assert_fails(lambda: toml.encode({"a": [{"x": len}]}), 'in dict key "a": at list index 0: in dict key "x": cannot encode builtin_function_or_method')
assert_fails(lambda: toml.encode({"a": {1: "bad"}}), 'in dict key "a": dict has int key, want string')

cycle = []
cycle.append(cycle)
assert_fails(lambda: toml.encode({"cycle": cycle}), "nesting depth limit exceeded")
dict_cycle = {}
dict_cycle["self"] = dict_cycle
assert_fails(lambda: toml.encode(dict_cycle), "nesting depth limit exceeded")
