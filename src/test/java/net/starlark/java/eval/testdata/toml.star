# tests of TOML encoding/decoding

assert_eq(dir(toml), ["decode", "encode"])

## toml.decode

assert_eq(toml.decode('title = "TOML Example"'), {"title": "TOML Example"})
assert_eq(toml.decode('bool = true'), {"bool": True})
assert_eq(toml.decode('date = 1979-05-27T07:32:00Z'), {"date": "1979-05-27T07:32Z"})
assert_eq(toml.decode('float = 42.42'), {"float": 42.42})
assert_eq(toml.decode('number = 42'), {"number": 42})
assert_eq(toml.decode('temp_targets = { cpu = 79.5, case = 72.0 }', ), {"temp_targets": {"cpu": 79.5, "case": 72.0}})
assert_eq(toml.decode("nested_arrays_of_ints = [ [ 1, 2 ], [3, 4, 5] ]"), {"nested_arrays_of_ints": [[1, 2], [3, 4, 5]]})
assert_eq(toml.decode('title =', default = True), True)
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

assert_fails(lambda: toml.decode('title = '), "Unexpected end of input")
assert_fails(lambda: toml.decode('title = ['), "Unexpected end of input")
assert_fails(lambda: toml.decode('['), "Unexpected end of input")
assert_fails(lambda: toml.decode('{'), "Unexpected")

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
    ]
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

# Sets become TOML arrays (order is deterministic after sorting)
assert_eq(toml.encode({"items": set([3, 1, 2])}), """\
items = [
  1,
  2,
  3,
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
[nested1]
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
[foo]
[foo.bar]
[foo.bar.baz]
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
enabled = true
host = 'localhost'
port = 5432
"""
)

# None values are omitted from output
assert_eq(toml.encode({"value": (None, None)}), "value = []\n")
assert_eq(toml.encode({"val": None}), "")
assert_eq(toml.encode({"a": 1, "b": None, "c": 3}), """\
a = 1
c = 3
""")
assert_eq(toml.encode({"items": [1, None, 3]}), """\
items = [
  1,
  3,
]
""")
assert_eq(toml.encode({"nested": {"x": 1, "y": None}}), """\
[nested]
x = 1
""")

# Error cases - non-finite floats
assert_fails(lambda: toml.encode({"x": float("NaN")}), "cannot encode non-finite")
assert_fails(lambda: toml.encode({"x": float("+Inf")}), "cannot encode non-finite")
assert_fails(lambda: toml.encode({"x": float("-Inf")}), "cannot encode non-finite")

# Error cases - non-dict top-level values
assert_fails(lambda: toml.encode([1, 2, 3]), "TOML encode requires a dict at the top level")
assert_fails(lambda: toml.encode((1, 2, 3)), "TOML encode requires a dict at the top level")
assert_fails(lambda: toml.encode("hello"), "TOML encode requires a dict at the top level")
assert_fails(lambda: toml.encode(42), "TOML encode requires a dict at the top level")

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
    for x in range(100000):
        deep = [deep]
    toml.encode({"data": deep})

assert_fails(lambda: f(1), "nesting depth limit exceeded")

# Round-trip test: encode then decode should give back original value
original = {
    "title": "Config",
    "count": 42,
    "enabled": True,
    "ratio": 3.14,
    "items": ["a", "b", "c"],
}
encoded = toml.encode(original)
decoded = toml.decode(encoded)
# Arrays will decode as expected
assert_eq(decoded, original)

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
[a]
[a.b]
[a.b.c]
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
a = 2
m = 3
z = 1
""")

assert_eq(toml.encode({"foo": [
    {"name": "bar", "id": 1},
    {"name": "baz", "id": 2},
]}), """\
[[foo]]
id = 1
name = 'bar'

[[foo]]
id = 2
name = 'baz'
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
['a.b']
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
special_keys_original = {
    "my key": "value1",
    "key.with.dots": "value2",
}
special_keys_encoded = toml.encode(special_keys_original)
special_keys_decoded = toml.decode(special_keys_encoded)
assert_eq(special_keys_decoded, special_keys_original)
