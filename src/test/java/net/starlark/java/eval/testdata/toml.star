# tests of TOML encoding/decoding

assert_eq(dir(toml), ["decode", "encode"])

## toml.decode

assert_eq(toml.decode('title = "TOML Example"'), {"title": "TOML Example"})
assert_eq(toml.decode('bool = true'), {"bool": True})
assert_eq(toml.decode('date = 1979-05-27T07:32:00Z'), {"date": "1979-05-27T07:32:00Z"})
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

assert_fails(lambda: toml.decode('title = '), "Premature end of file")
assert_fails(lambda: toml.decode('title = ['), "Premature end of file")
assert_fails(lambda: toml.decode('['), "Premature end of file")
assert_fails(lambda: toml.decode('{'), "Unknown token")

## toml.encode

# Basic dict with different value types
assert_eq(toml.encode({"x": 1, "y": "two"}), "x = 1\ny = 'two'\n")
assert_eq(toml.encode({"title": "TOML Example"}), "title = 'TOML Example'\n")
assert_eq(toml.encode({"bool": True, "number": 42}), "bool = true\nnumber = 42\n")
assert_eq(toml.encode({"count": -123}), "count = -123\n")
assert_eq(toml.encode({"bignum": 12345 * 12345 * 12345}), "bignum = 1881365963625\n")
assert_eq(toml.encode({"ratio": 12.345}), "ratio = 12.345\n")
assert_eq(toml.encode({"name": "hello"}), "name = 'hello'\n")
assert_eq(toml.encode({"empty": ""}), "empty = ''\n")

# Lists and tuples become TOML arrays
assert_eq(toml.encode({"numbers": [1, 2, 3]}), "numbers = [1, 2, 3]\n")
assert_eq(toml.encode({"tuple_val": (1, 2, 3)}), "tuple_val = [1, 2, 3]\n")
assert_eq(toml.encode({"strings": ["a", "b", "c"]}), "strings = ['a', 'b', 'c']\n")
assert_eq(toml.encode({"empty_array": []}), "empty_array = []\n")

# Sets become TOML arrays (order is deterministic after sorting)
assert_eq(toml.encode({"items": set([3, 1, 2])}), "items = [1, 2, 3]\n")
assert_eq(toml.encode({"words": set(["z", "a", "m"])}), "words = ['a', 'm', 'z']\n")

# Nested structures
assert_eq(toml.encode({"nested": {"x": 1, "y": 2}}), "nested.x = 1\nnested.y = 2\n")
assert_eq(toml.encode({"arrays": [[1, 2], [3, 4]]}), "arrays = [[1, 2], [3, 4]]\n")

# Structs
assert_eq(toml.encode(struct(x = 1, y = "two")), "x = 1\ny = 'two'\n")
assert_eq(toml.encode(struct(title = "Example", count = 5)), "count = 5\ntitle = 'Example'\n")
assert_eq(toml.encode({"foo": struct(title = "Example", count = 5)}), "foo.count = 5\nfoo.title = 'Example'\n")

# Empty dict
assert_eq(toml.encode({}), "")
assert_eq(toml.encode({"foo": {}}), "foo = {}\n")

# Mixed nested types
assert_eq(
    toml.encode({"database": {"host": "localhost", "port": 5432, "enabled": True}}),
    "database.enabled = true\ndatabase.host = 'localhost'\ndatabase.port = 5432\n"
)

# Error cases - None cannot be encoded
assert_fails(lambda: toml.encode({"val": None}), "cannot encode None as TOML")

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
assert_eq(decoded, original)
