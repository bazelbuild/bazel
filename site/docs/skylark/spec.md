---
layout: documentation
title: Skylark Language Specification
---

# Skylark Language Specification

Skylark is a dialect of Python intended for use as a configuration
language.  A Skylark interpreter is typically embedded within a larger
application, and this application may define additional
domain-specific functions and data types beyond those provided by the
core language.  For example, Skylark is embedded within (and was
originally developed for) the [Bazel build tool](https://bazel.build),
and [Bazel's build language](https://docs.bazel.build/versions/master/skylark/language.html) is based on Skylark.
Another implementation in Go can be found here: https://github.com/google/skylark/

This document was derived from the [description of the Go
implementation](https://github.com/google/skylark/blob/master/doc/spec.md)
of Skylark.
It was influenced by the Python specification,
Copyright 1990&ndash;2017, Python Software Foundation,
and the Go specification, Copyright 2009&ndash;2017, The Go Authors. It is
now maintained by the Bazel team.

The name "Skylark" is a code name of the Bazel project.
We plan to rename the language before the end of 2017 to reflect its
applicability to projects unrelated to Bazel.

<!-- [TOC] -->

## Overview

Skylark is an untyped dynamic language with high-level data types,
first-class functions with lexical scope, and automatic memory
management or _garbage collection_.

Skylark is strongly influenced by Python, and is almost a subset of
that language.  In particular, its data types and syntax for
statements and expressions will be very familiar to any Python
programmer.
However, Skylark is intended not for writing applications but for
expressing configuration: its programs are short-lived and have no
external side effects and their main result is structured data or side
effects on the host application.

Skylark is intended to be simple. There are no user-defined types, no
inheritance, no reflection, no exceptions, no explicit memory management.
Execution is finite. The language does not allow recursion or unbounded loops.

Skylark is suitable for use in highly parallel applications.  An application may
invoke the Skylark interpreter concurrently from many threads, without the
possibility of a data race, because shared data structures become immutable due
to _freezing_.

The language is deterministic and hermetic. Executing the same file with the
same interpreter leads to the same result. By default, user code cannot
interact with the environment.


## Lexical elements

A Skylark program consists of one or more modules. Each module is defined by a
single UTF-8-encoded text file.

Skylark grammar is introduced gradually throughout this document as shown below,
and a [complete Skylark grammar reference](#grammar) is provided at the end.

Grammar notation:

```text
- lowercase and 'quoted' items are lexical tokens.
- Capitalized names denote grammar productions.
- (...) implies grouping.
- x | y means either x or y.
- [x] means x is optional.
- {x} means x is repeated zero or more times.
- The end of each declaration is marked with a period.
```

The contents of a Skylark file are broken into a sequence of tokens of
five kinds: white space, punctuation, keywords, identifiers, and literals.
Each token is formed from the longest sequence of characters that
would form a valid token of each kind.

```text
File = {Statement | newline} eof .
```

*White space* consists of spaces (U+0020), tabs (U+0009), carriage
returns (U+000D), and newlines (U+000A).  Within a line, white space
has no effect other than to delimit the previous token, but newlines,
and spaces at the start of a line, are significant tokens.

*Comments*: A hash character (`#`) appearing outside of a string
literal marks the start of a comment; the comment extends to the end
of the line, not including the newline character.
Comments are treated like other white space.

*Punctuation*: The following punctuation characters or sequences of
characters are tokens:

```text
+    -    *    //   %    **
.    ,    =    ;    :
(    )    [    ]    {    }
<    >    >=   <=   ==   !=
+=   -=   *=   //=  %=
```

*Keywords*: The following tokens are keywords and may not be used as
identifiers:

```text
and            else           load
break          for            not
continue       if             or
def            in             pass
elif           return
```

The tokens below also may not be used as identifiers although they do not
appear in the grammar; they are reserved as possible future keywords:

<!-- and to remain a syntactic subset of Python -->

```text
as             is
assert         lambda
class          nonlocal
del            raise
except         try
finally        while
from           with
global         yield
import
```

*Identifiers*: an identifier is a sequence of Unicode letters, decimal
 digits, and underscores (`_`), not starting with a digit.
Identifiers are used as names for values.

Examples:

```text
None    True    len
x       index   starts_with     arg0
```

*Literals*: literals are tokens that denote specific values.  Skylark
has string and integer literals.

```text
0                               # int
123                             # decimal int
0x7f                            # hexadecimal int
0o755                           # octal int

"hello"      'hello'            # string
'''hello'''  """hello"""        # triple-quoted string
r'hello'     r"hello"           # raw string literal
```

Integer literal tokens are defined by the following grammar:

```text
int         = decimal_lit | octal_lit | hex_lit | 0 .
decimal_lit = ('1' … '9') {decimal_digit} .
octal_lit   = '0' ('o' | 'O') octal_digit {octal_digit} .
hex_lit     = '0' ('x' | 'X') hex_digit {hex_digit} .

decimal_digit = '0' … '9' .
octal_digit   = '0' … '7' .
hex_digit     = '0' … '9' | 'A' … 'F' | 'a' … 'f' .
```

TODO: define string_lit, indent, outdent, semicolon, newline, eof

## Data types

These are the main data types built in to the interpreter:

```text
NoneType                     # the type of None
bool                         # True or False
int                          # a signed integer
string                       # a byte string
list                         # a fixed-length sequence of values
tuple                        # a fixed-length sequence of values, unmodifiable
dict                         # a mapping from values to values
function                     # a function
```

Some functions, such as the `range` function, return instances of
special-purpose types that don't appear in this list.
Additional data types may be defined by the host application into
which the interpreter is embedded, and those data types may
participate in basic operations of the language such as arithmetic,
comparison, indexing, and function calls.

<!-- We needn't mention the stringIterable type here. -->

Some operations can be applied to any Skylark value.  For example,
every value has a type string that can be obtained with the expression
`type(x)`, and any value may be converted to a string using the
expression `str(x)`, or to a Boolean truth value using the expression
`bool(x)`.  Other operations apply only to certain types.  For
example, the indexing operation `a[i]` works only with strings, lists,
and tuples, and any application-defined types that are _indexable_.
The [_value concepts_](#value-concepts) section explains the groupings of
types by the operators they support.


### None

`None` is a distinguished value used to indicate the absence of any other value.
For example, the result of a call to a function that contains no return statement is `None`.

`None` is equal only to itself.  Its [type](#type) is `"NoneType"`.
The truth value of `None` is `False`.


### Booleans

There are two Boolean values, `True` and `False`, representing the
truth or falsehood of a predicate.  The [type](#type) of a Boolean is `"bool"`.

Boolean values are typically used as conditions in `if`-statements,
although any Skylark value used as a condition is implicitly
interpreted as a Boolean.
For example, the values `None`, `0`, and the empty sequences
`""`, `()`, `[]`, and `{}` have a truth value of `False`, whereas non-zero
numbers and non-empty sequences have a truth value of `True`.
Application-defined types determine their own truth value.
Any value may be explicitly converted to a Boolean using the built-in `bool`
function.

```python
1 + 1 == 2                              # True
2 + 2 == 5                              # False

if 1 + 1:
        print("True")
else:
        print("False")
```

### Integers

The Skylark integer type represents integers.  Its [type](#type) is `"int"`.

Integers may be positive or negative. The precision is implementation-dependent.
It is a dynamic error if a result is outside the supported range.
Integers are totally ordered; comparisons follow mathematical
tradition.

The `+` and `-` operators perform addition and subtraction, respectively.
The `*` operator performs multiplication.

The `//` and `%` operations on integers compute floored division and
remainder of floored division, respectively.
If the signs of the operands differ, the sign of the remainder `x % y`
matches that of the dividend, `x`.
For all finite x and y (y ≠ 0), `(x // y) * y + (x % y) == x`.

Any bool, number, or string may be interpreted as an integer by using
the `int` built-in function.

An integer used in a Boolean context is considered true if it is
non-zero.

```python
100 // 5 * 9 + 32               # 212
3 // 2                          # 1
111111111 * 111111111           # 12345678987654321
int("0xffff", 16)               # 65535
```


### Strings

A string represents an immutable sequence of bytes.
The [type](#type) of a string is `"string"`.

Strings can represent arbitrary binary data, including zero bytes, but
most strings contain text, encoded by convention using UTF-8.

The built-in `len` function returns the number of bytes in a string.

Strings may be concatenated with the `+` operator.

The substring expression `s[i:j]` returns the substring of `s` from
index `i` up to index `j`.  The index expression `s[i]` returns the
1-byte substring `s[i:i+1]`.

Strings are hashable, and thus may be used as keys in a dictionary.

Strings are totally ordered lexicographically, so strings may be
compared using operators such as `==` and `<`.

Strings are _not_ iterable sequences, so they cannot be used as the operand of
a `for`-loop, list comprehension, or any other operation than requires
an iterable sequence.

Any value may formatted as a string using the `str` or `repr` built-in
functions, the `str % tuple` operator, or the `str.format` method.

A string used in a Boolean context is considered true if it is
non-empty.

Strings have several built-in methods:

* [`capitalize`](#string·capitalize)
* [`count`](#string·count)
* [`endswith`](#string·endswith)
* [`find`](#string·find)
* [`format`](#string·format)
* [`index`](#string·index)
* [`isalnum`](#string·isalnum)
* [`isalpha`](#string·isalpha)
* [`isdigit`](#string·isdigit)
* [`islower`](#string·islower)
* [`isspace`](#string·isspace)
* [`istitle`](#string·istitle)
* [`isupper`](#string·isupper)
* [`join`](#string·join)
* [`lower`](#string·lower)
* [`lstrip`](#string·lstrip)
* [`partition`](#string·partition)
* [`replace`](#string·replace)
* [`rfind`](#string·rfind)
* [`rindex`](#string·rindex)
* [`rpartition`](#string·rpartition)
* [`rsplit`](#string·rsplit)
* [`rstrip`](#string·rstrip)
* [`split`](#string·split)
* [`splitlines`](#string·splitlines)
* [`startswith`](#string·startswith)
* [`strip`](#string·strip)
* [`title`](#string·title)
* [`upper`](#string·upper)


### Lists

A list is a mutable sequence of values.
The [type](#type) of a list is `"list"`.

Lists are indexable sequences: the elements of a list may be iterated
over by `for`-loops, list comprehensions, and various built-in
functions.

List may be constructed using bracketed list notation:

```python
[]              # an empty list
[1]             # a 1-element list
[1, 2]          # a 2-element list
```

Lists can also be constructed from any iterable sequence by using the
built-in `list` function.

The built-in `len` function applied to a list returns the number of elements.
The index expression `list[i]` returns the element at index i,
and the slice expression `list[i:j]` returns a new list consisting of
the elements at indices from i to j.

List elements may be added using the `append` or `extend` methods,
removed using the `remove` method, or reordered by assignments such as
`list[i] = list[j]`.

The concatenation operation `x + y` yields a new list containing all
the elements of the two lists x and y.

For most types, `x += y` is equivalent to `x = x + y`, except that it
evaluates `x` only once, that is, it allocates a new list to hold
the concatenation of `x` and `y`.
However, if `x` refers to a list, the statement does not allocate a
new list but instead mutates the original list in place, similar to
`x.extend(y)`.

Lists are not hashable, so may not be used in the keys of a dictionary.

A list used in a Boolean context is considered true if it is
non-empty.

A [_list comprehension_](#comprehensions) creates a new list whose elements are the
result of some expression applied to each element of another sequence.

```python
[x*x for x in [1, 2, 3, 4]]      # [1, 4, 9, 16]
```

A list value has these methods:

* [`append`](#list·append)
* [`clear`](#list·clear)
* [`extend`](#list·extend)
* [`index`](#list·index)
* [`insert`](#list·insert)
* [`pop`](#list·pop)
* [`remove`](#list·remove)

### Tuples

A tuple is an immutable sequence of values.
The [type](#type) of a tuple is `"tuple"`.

Tuples are constructed using parenthesized list notation:

```python
()                      # the empty tuple
(1,)                    # a 1-tuple
(1, 2)                  # a 2-tuple ("pair")
(1, 2, 3)               # a 3-tuple
```

Observe that for the 1-tuple, the trailing comma is necessary to
distinguish it from the parenthesized expression `(1)`.
1-tuples are seldom used.

Skylark, unlike Python, does not permit a trailing comma to appear in
an unparenthesized tuple expression:

```python
for k, v, in dict.items(): pass                 # syntax error at 'in'
_ = [(v, k) for k, v, in dict.items()]          # syntax error at 'in'

sorted(3, 1, 4, 1,)                             # ok
[1, 2, 3, ]                                     # ok
{1: 2, 3:4, }                                   # ok
```

Any iterable sequence may be converted to a tuple by using the
built-in `tuple` function.

Like lists, tuples are indexed sequences, so they may be indexed and
sliced.  The index expression `tuple[i]` returns the tuple element at
index i, and the slice expression `tuple[i:j]` returns a subsequence
of a tuple.

Tuples are iterable sequences, so they may be used as the operand of a
`for`-loop, a list comprehension, or various built-in functions.

Unlike lists, tuples cannot be modified.
However, the mutable elements of a tuple may be modified.

Tuples are hashable (assuming their elements are hashable),
so they may be used as keys of a dictionary.

Tuples may be concatenated using the `+` operator.

A tuple used in a Boolean context is considered true if it is
non-empty.


### Dictionaries

A dictionary is a mutable mapping from keys to values.
The [type](#type) of a dictionary is `"dict"`.

Dictionaries provide constant-time operations to insert an element, to
look up the value for a key, or to remove an element.  Dictionaries
are implemented using hash tables, so keys must be hashable.  Hashable
values include `None`, Booleans, numbers, and strings, and tuples
composed from hashable values.  Most mutable values, such as lists,
and dictionaries, are not hashable, unless they are frozen.
Attempting to use a non-hashable value as a key in a dictionary
results in a dynamic error, as does passing one to the built-in
`hash` function.

A [dictionary expression](#dictionary-expressions) specifies a
dictionary as a set of key/value pairs enclosed in braces:

```python
coins = {
  "penny": 1,
  "nickel": 5,
  "dime": 10,
  "quarter": 25,
}
```

The expression `d[k]`, where `d` is a dictionary and `k` is a key,
retrieves the value associated with the key.  If the dictionary
contains no such item, the operation fails:

```python
coins["penny"]          # 1
coins["dime"]           # 10
coins["silver dollar"]  # error: key not found
```

The number of items in a dictionary `d` is given by `len(d)`.
A key/value item may be added to a dictionary, or updated if the key
is already present, by using `d[k]` on the left side of an assignment:

```python
len(coins)				# 4
coins["shilling"] = 20
len(coins)				# 5, item was inserted
coins["shilling"] = 5
len(coins)				# 5, existing item was updated
```

A dictionary can also be constructed using a [dictionary
comprehension](#comprehension), which evaluates a pair of expressions,
the _key_ and the _value_, for every element of another iterable such
as a list.  This example builds a mapping from each word to its length
in bytes:

```python
words = ["able", "baker", "charlie"]
{x: len(x) for x in words}	# {"charlie": 7, "baker": 5, "able": 4}
```

Dictionaries are iterable sequences, so they may be used as the
operand of a `for`-loop, a list comprehension, or various built-in
functions.
Iteration yields the dictionary's keys in the order in which they were
inserted; updating the value associated with an existing key does not
affect the iteration order.

```python
x = dict([("a", 1), ("b", 2)])          # {"a": 1, "b": 2}
x.update([("a", 3), ("c", 4)])          # {"a": 3, "b": 2, "c": 4}
```

```python
for name in coins:
  print(name, coins[name])	# prints "quarter 25", "dime 10", ...
```

Like all mutable values in Skylark, a dictionary can be frozen, and
once frozen, all subsequent operations that attempt to update it will
fail.

A dictionary used in a Boolean context is considered true if it is
non-empty.

The binary `+` operation may be applied to two dictionaries.  It
yields a new dictionary whose elements are the union of the two
operands.  If a key is present in both operands, the result contains
the value from the right operand.
<b>Note:</b> this feature is deprecated.  Use the
`dict.update` method instead.

Dictionaries may be compared for equality using `==` and `!=`.  Two
dictionaries compare equal if they contain the same number of items
and each key/value item (k, v) found in one dictionary is also present
in the other.  Dictionaries are not ordered; it is an error to compare
two dictionaries with `<`.


A dictionary value has these methods:

* [`clear`](#dict·clear)
* [`get`](#dict·get)
* [`items`](#dict·items)
* [`keys`](#dict·keys)
* [`pop`](#dict·pop)
* [`popitem`](#dict·popitem)
* [`setdefault`](#dict·setdefault)
* [`update`](#dict·update)
* [`values`](#dict·values)


### Functions

A function value represents a function defined in Skylark.
Its [type](#type) is `"function"`.
A function value used in a Boolean context is always considered true.

Function definitions may not be nested.

A function definition defines zero or more named parameters.
Skylark has a rich mechanism for passing arguments to functions.

<!-- TODO break up this explanation into caller-side and callee-side
     parts, and put the former under function calls and the latter
     under function definitions. Also try to convey that the Callable
     interface sees the flattened-out args and kwargs and that's what
     built-ins get.
-->

The example below shows a definition and call of a function of two
required parameters, `x` and `y`.

```python
def idiv(x, y):
  return x // y

idiv(6, 3)		# 2
```

A call may provide arguments to function parameters either by
position, as in the example above, or by name, as in first two calls
below, or by a mixture of the two forms, as in the third call below.
All the positional arguments must precede all the named arguments.
Named arguments may improve clarity, especially in functions of
several parameters.

```python
idiv(x=6, y=3)		# 2
idiv(y=3, x=6)		# 2

idiv(6, y=3)		# 2
```

<b>Optional parameters:</b> A parameter declaration may specify a
default value using `name=value` syntax; such a parameter is
_optional_.  The default value expression is evaluated during
execution of the `def` statement, and the default value forms part of the function value.
All optional parameters must follow all non-optional parameters.
A function call may omit arguments for any suffix of the optional
parameters; the effective values of those arguments are supplied by
the function's parameter defaults.

```python
def f(x, y=3):
  return x, y

f(1, 2)	# (1, 2)
f(1)	# (1, 3)
```

If a function parameter's default value is a mutable expression,
modifications to the value during one call may be observed by
subsequent calls.
Beware of this when using lists or dicts as default values.
If the function becomes frozen, its parameters' default values become
frozen too.

```python
# module a.sky
def f(x, list=[]):
  list.append(x)
  return list

f(4, [1,2,3])           # [1, 2, 3, 4]
f(1)                    # [1]
f(2)                    # [1, 2], not [2]!

# module b.sky
load("a.sky", "f")
f(3)                    # error: cannot append to frozen list
```

<b>Variadic functions:</b> Some functions allow callers to provide an
arbitrary number of arguments.
After all required and optional parameters, a function definition may
specify a _variadic arguments_ or _varargs_ parameter, indicated by a
star preceding the parameter name: `*args`.
Any surplus positional arguments provided by the caller are formed
into a tuple and assigned to the `args` parameter.

```python
def f(x, y, *args):
  return x, y, args

f(1, 2)                 # (1, 2, ())
f(1, 2, 3, 4)           # (1, 2, (3, 4))
```

<b>Keyword-variadic functions:</b> Some functions allow callers to
provide an arbitrary sequence of `name=value` keyword arguments.
A function definition may include a final _keyworded arguments_ or
_kwargs_ parameter, indicated by a double-star preceding the parameter
name: `**kwargs`.
Any surplus named arguments that do not correspond to named parameters
are collected in a new dictionary and assigned to the `kwargs` parameter:

```python
def f(x, y, **kwargs):
  return x, y, kwargs

f(1, 2)                 # (1, 2, {})
f(x=2, y=1)             # (2, 1, {})
f(x=2, y=1, z=3)        # (2, 1, {"z": 3})
```

It is a static error if any two parameters of a function have the same name.

Just as a function definition may accept an arbitrary number of
positional or keyworded arguments, a function call may provide an
arbitrary number of positional or keyworded arguments supplied by a
list or dictionary:

```python
def f(a, b, c=5):
  return a * b + c

f(*[2, 3])              # 11
f(*[2, 3, 7])           # 13
f(*[2])                 # error: f takes at least 2 arguments (1 given)

f(**dict(b=3, a=2))             # 11
f(**dict(c=7, a=2, b=3))        # 13
f(**dict(a=2))                  # error: f takes at least 2 arguments (1 given)
f(**dict(d=4))                  # error: f got unexpected keyword argument "d"
```

Once the parameters have been successfully bound to the arguments
supplied by the call, the sequence of statements that comprise the
function body is executed.

A function call completes normally after the execution of either a
`return` statement, or of the last statement in the function body.
The result of the function call is the value of the return statement's
operand, or `None` if the return statement had no operand or if the
function completeted without executing a return statement.

```python
def f(x):
  if x == 0:
    return
  if x < 0:
    return -x
  print(x)

f(1)            # returns None after printing "1"
f(0)            # returns None without printing
f(-1)           # returns 1 without printing
```


It is a dynamic error for a function to call itself or another
function value with the same declaration.

```python
def fib(x):
  if x < 2:
    return x
  return fib(x-2) + fib(x-1)	# dynamic error: function fib called recursively

fib(5)
```

This rule, combined with the invariant that all loops are iterations
over finite sequences, implies that Skylark programs are not Turing-complete.

<!-- This rule is supposed to deter people from abusing Skylark for
     inappropriate uses, especially in the build system.
     It may work for that purpose, but it doesn't stop Skylark programs
     from consuming too much time or space.  Perhaps it should be a
     dialect option.
-->



### Built-in functions

A built-in function is a function or method implemented by the interpreter
or the application into which the interpreter is embedded.

A built-in function value used in a Boolean context is always considered true.

Many built-in functions are defined in the "universe" block of the environment
(see [Name Resolution](#name-resolution)), and are thus available to
all Skylark programs.

Except where noted, built-in functions accept only positional arguments.

## Name binding and variables

After a Skylark file is parsed, but before its execution begins, the
Skylark interpreter checks statically that the program is well formed.
For example, `break` and `continue` statements may appear only within
a loop; `if`, `for`, and `return` statements may appear only within a
function; and `load` statements may appear only outside any function.

_Name resolution_ is the static checking process that
resolves names to variable bindings.
During execution, names refer to variables.  Statically, names denote
places in the code where variables are created; these places are
called _bindings_.  A name may denote different bindings at different
places in the program.  The region of text in which a particular name
refers to the same binding is called that binding's _scope_.

Four Skylark constructs bind names, as illustrated in the example below:
`load` statements (`a` and `b`),
`def` statements (`c`),
function parameters (`d`),
and assignments (`e`, `h`, including the augmented assignment `e += h`).
Variables may be assigned or re-assigned explicitly (`e`, `h`), or implicitly, as
in a `for`-loop (`f`) or comprehension (`g`, `i`).

```python
load("lib.sky", "a", b="B")

def c(d):
  e = 0
  for f in d:
     print([True for g in f])
     e += 1

h = [2*i for i in a]
```

The environment of a Skylark program is structured as a tree of
_lexical blocks_, each of which may contain name bindings.
The tree of blocks is parallel to the syntax tree.
Blocks are of four kinds.

<!-- Avoid the term "built-in block" since that's also a type. -->
At the root of the tree is the _universe_ block, which binds constant
values such as `None`, `True`, and `False`, and built-in functions
such as `len`, `list`, and so on.
Skylark programs cannot change the set of universe bindings.
Because the universe block is shared by all files (modules),
all values bound in it must be immutable and stateless
from the perspective of the Skylark program.

Nested beneath the universe block is the _module_ block, which
contains the bindings of the current file.
Bindings in the module block (such as `a`, `b`, `c`, and `h` in the
example) are called _global_.
The module block is typically empty at the start of the file
and is populated by top-level binding statements,
but an application may pre-bind one or more global names,
to provide domain-specific functions to that file, for example.

A module block contains a _function_ block for each top-level
function, and a _comprehension_ block for each top-level
comprehension.
Bindings inside either of these kinds of block are called _local_.
Additional functions and comprehensions, and their blocks, may be
nested in any order, to any depth.

If name is bound anywhere within a block, all uses of the name within
the block are treated as references to that binding, even uses that
appear before the binding.
The binding of `y` on the last line of the example below makes `y`
local to the function `hello`, so the use of `y` in the print
statement also refers to the local `y`, even though it appears
earlier.

```python
y = "goodbye"

def hello():
  for x in (1, 2):
    if x == 2:
      print(y) # prints "hello"
    if x == 1:
      y = "hello"
```

It is a dynamic error to evaluate a reference to a local variable
before it has been bound:

```python
def f():
  print(x)              # dynamic error: local variable x referenced before assignment
  x = "hello"
```

The same is true for global variables:

```python
print(x)                # dynamic error: global variable x referenced before assignment
x = "hello"
```

It is a static error to bind a global variable already explicitly bound in the file:

```python
x = 1
x = 2                   # static error: cannot reassign global x declared on line 1
```

If a name was pre-bound by the application, the Skylark program may
explicitly bind it, but only once.

An augmented assignment statement such as `x += 1` is considered a
binding of `x`. It is therefore a static error to use it on a global variable.

A name appearing after a dot, such as `split` in
`get_filename().split('/')`, is not resolved statically.
The [dot expression](#dot-expressions) `.split` is a dynamic operation
on the value returned by `get_filename()`.


## Value concepts {#value-concepts}

Skylark has eleven core [data types](#data-types).  An application
that embeds the Skylark intepreter may define additional types that
behave like Skylark values.  All values, whether core or
application-defined, implement a few basic behaviors:

```text
str(x)		-- return a string representation of x
type(x)		-- return a string describing the type of x
bool(x)		-- convert x to a Boolean truth value
hash(x)		-- return a hash code for x
```

### Identity and mutation

Skylark is an imperative language: programs consist of sequences of
statements executed for their side effects.
For example, an assignment statement updates the value held by a
variable, and calls to some built-in functions such as `print` change
the state of the application that embeds the interpreter.

Values of some data types, such as `NoneType`, `bool`, `int`, and
`string`, are _immutable_; they can never change.
Immutable values have no notion of _identity_: it is impossible for a
Skylark program to tell whether two integers, for instance, are
represented by the same object; it can tell only whether they are
equal.

Values of other data types, such as `list` and `dict`, are
_mutable_: they may be modified by a statement such as `a[i] = 0` or
`items.clear()`.  Although `tuple` and `function` values are not
directly mutable, they may refer to mutable values indirectly, so for
this reason we consider them mutable too.  Skylark values of these
types are actually _references_ to variables.

Copying a reference to a variable, using an assignment statement for
instance, creates an _alias_ for the variable, and the effects of
operations applied to the variable through one alias are visible
through all others.

```python
x = []                          # x refers to a new empty list variable
y = x                           # y becomes an alias for x
x.append(1)                     # changes the variable referred to by x
print(y)                        # "[1]"; y observes the mutation
```

Skylark uses _call-by-value_ parameter passing: in a function call,
argument values are assigned to function parameters as if by
assignment statements.  If the values are references, the caller and
callee may refer to the same variables, so if the called function
changes the variable referred to by a parameter, the effect may also
be observed by the caller:

```python
def f(y):
    y.append(1)                 # changes the variable referred to by x

x = []                          # x refers to a new empty list variable
f(x)                            # f's parameter y becomes an alias for x
print(x)                        # "[1]"; x observes the mutation
```


As in all imperative languages, understanding _aliasing_, the
relationship between reference values and the variables to which they
refer, is crucial to writing correct programs.

### Freezing a value

Skylark has a feature unusual among imperative programming languages:
a mutable value may be _frozen_ so that all subsequent attempts to
mutate it fail with a dynamic error; the value, and all other values
reachable from it, become _immutable_.

Immediately after execution of a Skylark module, all values in its
top-level environment are frozen. Because all the global variables of
an initialized Skylark module are immutable, the module may be published to
and used by other threads in a parallel program without the need for
locks. For example, the Bazel build system loads and executes BUILD
and .bzl files in parallel, and two modules being executed
concurrently may freely access variables or call functions from a
third without the possibility of a race condition.

### Hashing

The `dict` data type is implemented using hash tables, so
only _hashable_ values are suitable as keys of a `dict`.
Attempting to use a non-hashable value as the key in a hash
table, or as the operand of the `hash` built-in function, results in a
dynamic error.

The hash of a value is an unspecified integer chosen so that two equal
values have the same hash, in other words, `x == y => hash(x) == hash(y)`.
A hashable value has the same hash throughout its lifetime.

Values of the types `NoneType`, `bool`, `int`, and `string`,
which are all immutable, are hashable.

Values of mutable types such as `list` and `dict` are not
hashable, unless they have become immutable due to _freezing_.

A `tuple` value is hashable only if all its elements are hashable.
Thus `("localhost", 80)` is hashable but `([127, 0, 0, 1], 80)` is not.

Values of type `function` are also hashable.
Although functions are not necessarily immutable, as they may be
closures that refer to mutable variables, instances of these types
are compared by reference identity (see [Comparisons](#comparisons)),
so their hash values are derived from their identity.


### Sequence types

Many Skylark data types represent a _sequence_ of values: lists
and tuples are sequences of arbitrary values, and in many
contexts dictionaries act like a sequence of their keys.

We can classify different kinds of sequence types based on the
operations they support.

* `Iterable`: an _iterable_ value lets us process each of its elements in a fixed order.
  Examples: `dict`, `list`, `tuple`, but not `string`.
* `Sequence`: a _sequence of known length_ lets us know how many elements it
  contains without processing them.
  Examples: `dict`, `list`, `tuple`, but not `string`.
* `Indexable`: an _indexed_ type has a fixed length and provides efficient
  random access to its elements, which are identified by integer indices.
  Examples: `string`, `tuple`, and `list`.
* `SetIndexable`: a _settable indexed type_ additionally allows us to modify the
  element at a given integer index. Example: `list`.
* `Mapping`: a mapping is an association of keys to values. Example: `dict`.

Although all of Skylark's core data types for sequences implement at
least the `Sequence` contract, it's possible for an an application
that embeds the Skylark interpreter to define additional data types
representing sequences of unknown length that implement only the `Iterable` contract.

Strings are not iterable, though they do support the `len(s)` and
`s[i]` operations. Skylark deviates from Python here to avoid common
pitfall in which a string is used by mistake where a list containing a
single string was intended, resulting in its interpretation as a sequence
of bytes.

Most Skylark operators and built-in functions that need a sequence
of values will accept any iterable.

It is a dynamic error to mutate a sequence such as a list or a
dictionary while iterating over it.

```python
def increment_values(dict):
  for k in dict:
    dict[k] += 1			# error: cannot insert into hash table during iteration

dict = {"one": 1, "two": 2}
increment_values(dict)
```


### Indexing

Many Skylark operators and functions require an index operand `i`,
such as `a[i]` or `list.insert(i, x)`. Others require two indices `i`
and `j` that indicate the start and end of a subsequence, such as
`a[i:j]`, `list.index(x, i, j)`, or `string.find(x, i, j)`.
All such operations follow similar conventions, described here.

Indexing in Skylark is *zero-based*. The first element of a string
or list has index 0, the next 1, and so on. The last element of a
sequence of length `n` has index `n-1`.

```python
"hello"[0]			# "h"
"hello"[4]			# "o"
"hello"[5]			# error: index out of range
```

For subsequence operations that require two indices, the first is
_inclusive_ and the second _exclusive_. Thus `a[i:j]` indicates the
sequence starting with element `i` up to but not including element
`j`. The length of this subsequence is `j-i`. This convention is known
as *half-open indexing*.

```python
"hello"[1:4]			# "ell"
```

Either or both of the index operands may be omitted. If omitted, the
first is treated equivalent to 0 and the second is equivalent to the
length of the sequence:

```python
"hello"[1:]                     # "ello"
"hello"[:4]                     # "hell"
```

It is permissible to supply a negative integer to an indexing
operation. The effective index is computed from the supplied value by
the following two-step procedure. First, if the value is negative, the
length of the sequence is added to it. This provides a convenient way
to address the final elements of the sequence:

```python
"hello"[-1]                     # "o",  like "hello"[4]
"hello"[-3:-1]                  # "ll", like "hello"[2:4]
```

Second, for subsequence operations, if the value is still negative, it
is replaced by zero, or if it is greater than the length `n` of the
sequence, it is replaced by `n`. In effect, the index is "truncated" to
the nearest value in the range `[0:n]`.

```python
"hello"[-1000:1000]		# "hello"
```

This truncation step does not apply to indices of individual elements:

```python
"hello"[-6]		# error: index out of range
"hello"[-5]		# "h"
"hello"[4]		# "o"
"hello"[5]		# error: index out of range
```


## Expressions

An expression specifies the computation of a value.

The Skylark grammar defines several categories of expression.
An _operand_ is an expression consisting of a single token (such as an
identifier or a literal), or a bracketed expression.
Operands are self-delimiting.
An operand may be followed by any number of dot, call, or slice
suffixes, to form a _primary_ expression.
In some places in the Skylark grammar where an expression is expected,
it is legal to provide a comma-separated list of expressions denoting
a tuple.
The grammar uses `Expression` where a multiple-component expression is allowed,
and `Test` where it accepts an expression of only a single component.

```text
Expression = Test {',' Test} .

Test = IfExpr | PrimaryExpr | UnaryExpr | BinaryExpr .

PrimaryExpr = Operand
            | PrimaryExpr DotSuffix
            | PrimaryExpr CallSuffix
            | PrimaryExpr SliceSuffix
            .

Operand = identifier
        | int | string
        | ListExpr | ListComp
        | DictExpr | DictComp
        | '(' [Expression] [,] ')'
        | '-' PrimaryExpr
        .

DotSuffix   = '.' identifier .
CallSuffix  = '(' [Arguments [',']] ')' .
SliceSuffix = '[' [Expression] [':' Test [':' Test]] ']' .
```

TODO: resolve position of -x, and 'not x' in grammar: Operand or UnaryExpr?

### Identifiers

```text
Primary = identifier
```

An identifier is a name that identifies a value.

Lookup of locals and globals may fail if not yet defined.

### Literals

Skylark supports string literals of three different kinds:

```text
Primary = int | string
```

Evaluation of a literal yields a value of the given type (string, int) with the
given value.
See [Literals](#lexical elements) for details.

### Parenthesized expressions

```text
Primary = '(' [Expression] ')'
```

A single expression enclosed in parentheses yields the result of that expression.
Explicit parentheses may be used for clarity,
or to override the default association of subexpressions.

```python
1 + 2 * 3 + 4                   # 11
(1 + 2) * (3 + 4)               # 21
```

If the parentheses are empty, or contain a single expression followed
by a comma, or contain two or more expressions, the expression yields a tuple.

```python
()                              # (), the empty tuple
(1,)                            # (1,), a tuple of length 1
(1, 2)                          # (1, 2), a 2-tuple or pair
(1, 2, 3)                       # (1, 2, 3), a 3-tuple or triple
```

In some contexts, such as a `return` or assignment statement or the
operand of a `for` statement, a tuple may be expressed without
parentheses.

```python
x, y = 1, 2

return 1, 2

for x in 1, 2:
   print(x)
```

Skylark (like Python 3) does not accept an unparenthesized tuple
expression as the operand of a list comprehension:

```python
[2*x for x in 1, 2, 3]	       	# parse error: unexpected ','
```

### Dictionary expressions

A dictionary expression is a comma-separated list of colon-separated
key/value expression pairs, enclosed in curly brackets, and it yields
a new dictionary object.
An optional comma may follow the final pair.

```text
DictExpr = '{' [Entries [',']] '}' .
Entries  = Entry {',' Entry} .
Entry    = Test ':' Test .
```

Examples:


```python
{}
{"one": 1}
{"one": 1, "two": 2,}
```

The key and value expressions are evaluated in left-to-right order.
Evaluation fails if the same key is used multiple times.

Only [hashable](#hashing) values may be used as the keys of a dictionary.


### List expressions

A list expression is a comma-separated list of element expressions,
enclosed in square brackets, and it yields a new list object.
An optional comma may follow the last element expression.

```text
ListExpr = '[' [Expression [',']] ']' .
```

Element expressions are evaluated in left-to-right order.

Examples:

```python
[]                      # [], empty list
[1]                     # [1], a 1-element list
[1, 2, 3,]              # [1, 2, 3], a 3-element list
```

### Unary operators

There are two unary operators, both appearing before their operand:
`-`, and `not`.

```text
UnaryExpr = '-' PrimaryExpr
          | 'not' Test
          .
```

```text
- number        unary negation          (number)
not x           logical negation        (any type)
```

The `-` operators returns the opposite of any number.

```python
if x > 0:
	return 1
elif x < 0:
	return -1
else:
	return 0
```

The `not` operator returns the negation of the truth value of its
operand.

```python
not True                        # False
not False                       # True
not [1, 2, 3]                   # False
not ""                          # True
not 0                           # True
```


### Binary operators

Skylark has the following binary operators, arranged in order of increasing precedence:

```text
or
and
not
==   !=   <   >   <=   >=   in   not in
-   +
*   /   //   %
```

Comparison operators, `in`, and `not in` are non-associative,
so the parser will not accept `0 <= i < n`.
All other binary operators of equal precedence associate to the left.

```text
BinaryExpr = Test {Binop Test} .

Binop = 'or'
      | 'and'
      | 'not'
      | '==' | '!=' | '<' | '>' | '<=' | '>=' | 'in' | 'not' 'in'
      | '-' | '+'
      | '*' | '%' | '/' | '//'
      .
```

#### `or` and `and`

The `or` and `and` operators yield, respectively, the logical disjunction and
conjunction of their arguments, which need not be Booleans.
The expression `x or y` yields the value of `x` if its truth value is `True`,
or the value of `y` otherwise.

```python
False or False		# False
False or True		# True
True  or False		# True
True  or True		# True

0 or "hello"		# "hello"
1 or "hello"		# 1
```

Similarly, `x and y` yields the value of `x` if its truth value is
`False`, or the value of `y` otherwise.

```python
False and False		# False
False and True		# False
True  and False		# False
True  and True		# True

0 and "hello"		# 0
1 and "hello"		# "hello"
```

These operators use "short circuit" evaluation, so the second
expression is not evaluated if the value of the first expression has
already determined the result, allowing constructions like these:

```python
len(x) > 0 and x[0] == 1		# x[0] is not evaluated if x is empty
x and x[0] == 1
len(x) == 0 or x[0] == ""
not x or not x[0]
```

#### Comparisons

The `==` operator reports whether its operands are equal; the `!=`
operator is its negation.

The operators `<`, `>`, `<=`, and `>=` perform an ordered comparison
of their operands.  It is an error to apply these operators to
operands of unequal type.  Of the built-in types, only the following
support ordered comparison, using the ordering relation shown:

```text
NoneType        # None <= None
bool            # False < True
int             # mathematical
string          # lexicographical
tuple           # lexicographical
list            # lexicographical
```

Applications may define additional types that support ordered
comparison.

The remaining built-in types support only equality comparisons.
Values of type `dict` compare equal if their elements compare
equal, and values of type `function` are equal only to themselves.

```text
dict                            # equal contents
function                        # identity
```

#### Arithmetic operations

The following table summarizes the binary arithmetic operations
available for built-in types:

```text
Arithmetic
   number + number              # addition
   number - number              # subtraction
   number * number              # multiplication
   number // number             # floored division
   number % number              # remainder of floored division

Concatenation
   string + string
     list + list
    tuple + tuple
     dict + dict                # (deprecated)

Repetition (string/list/tuple)
      int * sequence
 sequence * int

String interpolation
   string % any                 # see String Interpolation
```

The operands of the arithmetic operators `+`, `-`, `*`, `//`, and
`%` must both be `int`. The type of the result has type `int`.

The `+` operator may be applied to non-numeric operands of the same
type, such as two lists, two tuples, or two strings, in which case it
computes the concatenation of the two operands and yields a new value of
the same type.

```python
"Hello, " + "world"		# "Hello, world"
(1, 2) + (3, 4)			# (1, 2, 3, 4)
[1, 2] + [3, 4]			# [1, 2, 3, 4]
```

The `*` operator may be applied to an integer _n_ and a value of type
`string`, `list`, or `tuple`, in which case it yields a new value
of the same sequence type consisting of _n_ repetitions of the original sequence.
The order of the operands is immaterial.
Negative values of _n_ behave like zero.

```python
'mur' * 2               # 'murmur'
3 * range(3)            # [0, 1, 2, 0, 1, 2, 0, 1, 2]
```

Applications may define additional types that support any subset of
these operators.


#### Membership tests

```text
      any in     sequence		(list, tuple, dict, string)
      any not in sequence
```

The `in` operator reports whether its first operand is a member of its
second operand, which must be a list, tuple, dict, or string.
The `not in` operator is its negation.
Both return a Boolean.

The meaning of membership varies by the type of the second operand:
the members of a list or tuple are its elements;
the members of a dict are its keys;
the members of a string are all its substrings.

```python
1 in [1, 2, 3]                  # True
4 not in (1, 2, 3)              # True

d = {"one": 1, "two": 2}
"one" in d                      # True
"three" in d                    # False
1 in d                          # False

"nasty" in "dynasty"            # True
"a" in "banana"                 # True
"f" not in "way"                # True
```

#### String interpolation

The expression `format % args` performs _string interpolation_, a
simple form of template expansion.
The `format` string is interpreted as a sequence of literal portions
and _conversions_.
Each conversion, which starts with a `%` character, is replaced by its
corresponding value from `args`.
The characters following `%` in each conversion determine which
argument it uses and how to convert it to a string.

Each `%` character marks the start of a conversion specifier, unless
it is immediately followed by another `%`, in which cases both
characters together denote a single literal percent sign.

The conversion's operand is the next element of `args`,
which must be a tuple with exactly one component per conversion,
unless the format string contains only a single conversion, in which
case `args` itself is its operand.

Skylark does not support the flag, width, and padding specifiers
supported by Python's `%` and other variants of C's `printf`.

After the `%` comes a single letter indicating what
operand types are valid and how to convert the operand `x` to a string:

```text
%       none            literal percent sign
s       any             as if by str(x)
r       any             as if by repr(x)
d       number          signed integer decimal
```

It is an error if the argument does not have the type required by the
conversion specifier.  A Boolean argument is not considered a number.

Examples:

```python
"Hello %s" % "Bob"                              # "Hello Bob"

"Hello %s, your score is %d" % ("Bob", 75)      # "Hello Bob, your score is 75"
)
```

One subtlety: to use a tuple as the operand of a conversion in format
string containing only a single conversion, you must wrap the tuple in
a singleton tuple:

```python
"coordinates=%s" % (40, -74)	# error: too many arguments for format string
"coordinates=%s" % ((40, -74),)	# "coordinates=(40, -74)"
```

### Conditional expressions

A conditional expression has the form `a if cond else b`.
It first evaluates the condition `cond`.
If it's true, it evaluates `a` and yields its value;
otherwise it yields the value of `b`.

```text
IfExpr = Test 'if' Test 'else' Test .
```

Example:

```python
"yes" if enabled else "no"
```

### Comprehensions

A comprehension constructs new list or dictionary value by looping
over one or more iterables and evaluating a _body_ expression that produces
successive elements of the result.

A list comprehension consists of a single expression followed by one
or more _clauses_, the first of which must be a `for` clause.
Each `for` clause resembles a `for` statement, and specifies an
iterable operand and a set of variables to be assigned by successive
values of the iterable.
An `if` cause resembles an `if` statement, and specifies a condition
that must be met for the body expression to be evaluated.
A sequence of `for` and `if` clauses acts like a nested sequence of
`for` and `if` statements.

```text
ListComp = '[' Test {CompClause} ']'.
DictComp = '{' Entry {CompClause} '}' .

CompClause = 'for' LoopVariables 'in' Test
           | 'if' Test .

LoopVariables = PrimaryExpr {',' PrimaryExpr} .
```

Examples:

```python
[x*x for x in range(5)]                 # [0, 1, 4, 9, 16]
[x*x for x in range(5) if x%2 == 0]     # [0, 4, 16]
[(x, y) for x in range(5)
        if x%2 == 0
        for y in range(5)
        if y > x]                       # [(0, 1), (0, 2), (0, 3), (0, 4), (2, 3), (2, 4)]
```

A dict comprehension resembles a list comprehension, but its body is a
pair of expressions, `key: value`, separated by a colon,
and its result is a dictionary containing the key/value pairs
for which the body expression was evaluated.
Evaluation fails if the value of any key is unhashable.

As with a `for` loop, the loop variables may exploit compound
assignment:

```python
[x*y+z for (x, y), z in [((2, 3), 5), (("o", 2), "!")]]         # [11, 'oo!']
```

Skylark, following Python 3, does not accept an unparenthesized
tuple as the operand of a `for` clause:

```python
[x*x for x in 1, 2, 3]		# parse error: unexpected comma
```

Comprehensions in Skylark, again following Python 3, define a new lexical
block, so assignments to loop variables have no effect on variables of
the same name in an enclosing block:

```python
x = 1
_ = [x for x in [2]]            # new variable x is local to the comprehension
print(x)                        # 1
```


### Function and method calls

```text
CallSuffix = '(' [Arguments [',']] ')' .

Arguments = Argument {',' Argument} .
Argument  = Test | identifier '=' Test | '*' Test | '**' Test .
```

A value `f` of type `function` may be called using the expression `f(...)`.
Applications may define additional types whose values may be called in the same way.

A method call such as `filename.endswith(".sky")` is the composition
of two operations, `m = filename.endswith` and `m(".sky")`.
The first, a dot operation, yields a _bound method_, a function value
that pairs a receiver value (the `filename` string) with a choice of
method ([string·endswith](#string·endswith)).

Only built-in or application-defined types may have methods.

See [Functions](#functions) for an explanation of function parameter passing.

### Dot expressions

A dot expression `x.f` selects the attribute `f` (a field or method)
of the value `x`.

Fields are possessed by none of the main Skylark [data types](#data-types),
but some application-defined types have them.
Methods belong to the built-in types `string`, `list`, and `dict`,
and to many application-defined types.

```text
DotSuffix = '.' identifier .
```

A dot expression fails if the value does not have an attribute of the
specified name.

Use the built-in function `hasattr(x, "f")` to ascertain whether a
value has a specific attribute, or `dir(x)` to enumerate all its
attributes.  The `getattr(x, "f")` function can be used to select an
attribute when the name `"f"` is not known statically.

A dot expression that selects a method typically appears within a call
expression, as in these examples:

```python
["able", "baker", "charlie"].index("baker")     # 1
"banana".count("a")                             # 3
"banana".reverse()                              # error: string has no .reverse field or method
```

But when not called immediately, the dot expression evaluates to a
_bound method_, that is, a method coupled to a specific receiver
value.  A bound method can be called like an ordinary function,
without a receiver argument:

```python
f = "banana".count
f                                               # <built-in method count of string value>
f("a")                                          # 3
f("n")                                          # 2
```

<b>Implementation note:</b>
The Java implementation does not currently allow a method to be
selected but not immediately called.
See Google Issue b/21392896.

### Index expressions

An index expression `a[i]` yields the `i`th element of an _indexable_
type such as a string, tuple, or list.  The index `i` must be an `int`
value in the range -`n` ≤ `i` < `n`, where `n` is `len(a)`; any other
index results in an error.

```text
SliceSuffix = '[' [Expression] [':' Test [':' Test]] ']' .
```

A valid negative index `i` behaves like the non-negative index `n+i`,
allowing for convenient indexing relative to the end of the
sequence.

```python
"abc"[0]                        # "a"
"abc"[1]                        # "b"
"abc"[-1]                       # "c"

("zero", "one", "two")[0]       # "zero"
("zero", "one", "two")[1]       # "one"
("zero", "one", "two")[-1]      # "two"
```

An index expression `d[key]` may also be applied to a dictionary `d`,
to obtain the value associated with the specified key.  It is an error
if the dictionary contains no such key.

An index expression appearing on the left side of an assignment causes
the specified list or dictionary element to be updated:

```python
a = range(3)            # a == [0, 1, 2]
a[2] = 7                # a == [0, 1, 7]

coins["suzie b"] = 100
```

It is a dynamic error to attempt to update an element of an immutable
type, such as a tuple or string, or a frozen value of a mutable type.

### Slice expressions

A slice expression `a[start:stop:stride]` yields a new value containing a
subsequence of `a`, which must be a string, tuple, or list.

```text
SliceSuffix = '[' [Expression] [':' Test [':' Test]] ']' .
```

Each of the `start`, `stop`, and `stride` operands is optional;
if present, and not `None`, each must be an integer.
The `stride` value defaults to 1.
If the stride is not specified, the colon preceding it may be omitted too.
It is an error to specify a stride of zero.

Conceptually, these operands specify a sequence of values `i` starting
at `start` and successively adding `stride` until `i` reaches or
passes `stop`. The result consists of the concatenation of values of
`a[i]` for which `i` is valid.`

The effective start and stop indices are computed from the three
operands as follows.  Let `n` be the length of the sequence.

<b>If the stride is positive:</b>
If the `start` operand was omitted, it defaults to -infinity.
If the `end` operand was omitted, it defaults to +infinity.
For either operand, if a negative value was supplied, `n` is added to it.
The `start` and `end` values are then "clamped" to the
nearest value in the range 0 to `n`, inclusive.

<b>If the stride is negative:</b>
If the `start` operand was omitted, it defaults to +infinity.
If the `end` operand was omitted, it defaults to -infinity.
For either operand, if a negative value was supplied, `n` is added to it.
The `start` and `end` values are then "clamped" to the
nearest value in the range -1 to `n`-1, inclusive.

```python
"abc"[1:]               # "bc"  (remove first element)
"abc"[:-1]              # "ab"  (remove last element)
"abc"[1:-1]             # "b"   (remove first and last element)
"banana"[1::2]          # "aaa" (select alternate elements starting at index 1)
"banana"[4::-2]         # "nnb" (select alternate elements in reverse, starting at index 4)
```

Unlike Python, Skylark does not allow a slice expression on the left
side of an assignment.

Slicing a tuple or string may be more efficient than slicing a list
because tuples and strings are immutable, so the result of the
operation can share the underlying representation of the original
operand (when the stride is 1). By contrast, slicing a list requires
the creation of a new list and copying of the necessary elements.


## Statements

```text
Statement  = DefStmt | IfStmt | ForStmt | SimpleStmt .
SimpleStmt = SmallStmt {';' SmallStmt} [';'] '\n' .
SmallStmt  = ReturnStmt
           | BreakStmt | ContinueStmt | PassStmt
           | AssignStmt
           | ExprStmt
           | LoadStmt
           .
```

### Pass statements

A `pass` statement does nothing.  Use a `pass` statement when the
syntax requires a statement but no behavior is required, such as the
body of a function that does nothing.

```text
PassStmt = 'pass' .
```

Example:

```python
def noop():
   pass

def list_to_dict(items):
  # Convert list of tuples to dict
  m = {}
  for k, m[k] in items:
    pass
  return m
```

### Assignments

An assignment statement has the form `lhs = rhs`.  It evaluates the
expression on the right-hand side then assigns its value (or values) to
the variable (or variables) on the left-hand side.

```text
AssignStmt = Expression '=' Expression .
```

The expression on the left-hand side is called a _target_.  The
simplest target is the name of a variable, but a target may also have
the form of an index expression, to update the element of a list or
dictionary, to update the field of an object:

```python
k = 1
a[i] = v
m.f = ""
```

Compound targets may consist of a comma-separated list of
subtargets, optionally surrounded by parentheses or square brackets,
and targets may be nested arbitarily in this way.
An assignment to a compound target checks that the right-hand value is a
sequence with the same number of elements as the target.
Each element of the sequence is then assigned to the corresponding
element of the target, recursively applying the same logic.
It is a static error if the sequence is empty.

```python
a, b = 2, 3
(x, y) = f()
[zero, one, two] = range(3)

[(a, b), (c, d)] = ("ab", "cd")
```

The same process for assigning a value to a target expression is used
in `for` loops and in comprehensions.


### Augmented assignments

An augmented assignment, which has the form `lhs op= rhs` updates the
variable `lhs` by applying a binary arithmetic operator `op` (one of
`+`, `-`, `*`, `/`, `//`, `%`) to the previous value of `lhs` and the value
of `rhs`.

```text
AssignStmt = Expression ('=' | '+=' | '-=' | '*=' | '/=' | '//=' | '%=') Expression .
```

The left-hand side must be a simple target:
a name, an index expression, or a dot expression.

```python
x -= 1
x.filename += ".sky"
a[index()] *= 2
```

Any subexpressions in the target on the left-hand side are evaluated
exactly once, before the evaluation of `rhs`.
The first two assignments above are thus equivalent to:

```python
x = x - 1
x.filename = x.filename + ".sky"
```

and the third assignment is similar in effect to the following two
statements but does not declare a new temporary variable `i`:

```python
i = index()
a[i] = a[i] * 2
```

### Function definitions

A `def` statement creates a named function and assigns it to a variable.

```text
DefStmt = 'def' identifier '(' [Parameters [',']] ')' ':' Suite .
```

Example:

```python
def twice(x):
    return x * 2

str(twice)              # "<function f>"
twice(2)                # 4
twice("two")            # "twotwo"
```

The function's name is preceded by the `def` keyword and followed by
the parameter list (which is enclosed in parentheses), a colon, and
then an indented block of statements which form the body of the function.

The parameter list is a comma-separated list whose elements are of
four kinds.  First come zero or more required parameters, which are
simple identifiers; all calls must provide an argument value for these parameters.

The required parameters are followed by zero or more optional
parameters, of the form `name=expression`.  The expression specifies
the default value for the parameter for use in calls that do not
provide an argument value for it.

The required parameters are optionally followed by a single parameter
name preceded by a `*`.  This is the called the _varargs_ parameter,
and it accumulates surplus positional arguments specified by a call.

Finally, there may be an optional parameter name preceded by `**`.
This is called the _keyword arguments_ parameter, and accumulates in a
dictionary any surplus `name=value` arguments that do not match a
prior parameter.

Here are some example parameter lists:

```python
def f(): pass
def f(a, b, c): pass
def f(a, b, c=1): pass
def f(a, b, c=1, *args): pass
def f(a, b, c=1, *args, **kwargs): pass
def f(**kwargs): pass
```

Execution of a `def` statement creates a new function object.  The
function object contains: the syntax of the function body; the default
value for each optional parameter; the value of each free variable
referenced within the function body; and the global dictionary of the
current module.

<!-- this is too implementation-oriented; it's not a spec. -->


### Return statements

A `return` statement ends the execution of a function and returns a
value to the caller of the function.

```text
ReturnStmt = 'return' [Expression] .
```

A return statement may have zero, one, or more
result expressions separated by commas.
With no expressions, the function has the result `None`.
With a single expression, the function's result is the value of that expression.
With multiple expressions, the function's result is a tuple.

```python
return                  # returns None
return 1                # returns 1
return 1, 2             # returns (1, 2)
```

### Expression statements

An expression statement evaluates an expression and discards its result.

```text
ExprStmt = Expression .
```

Any expression may be used as a statement, but an expression statement is
most often used to call a function for its side effects.

```python
list.append(1)
```

### If statements

An `if` statement evaluates an expression (the _condition_), then, if
the truth value of the condition is `True`, executes a list of
statements.

```text
IfStmt = 'if' Test ':' Suite {'elif' Test ':' Suite} ['else' ':' Suite] .
```

Example:

```python
if score >= 100:
    print("You win!")
    return
```

An `if` statement may have an `else` block defining a second list of
statements to be executed if the condition is false.

```python
if score >= 100:
        print("You win!")
        return
else:
        print("Keep trying...")
        continue
```

It is common for the `else` block to contain another `if` statement.
To avoid increasing the nesting depth unnecessarily, the `else` and
following `if` may be combined as `elif`:

```python
if x > 0:
        result = 1
elif x < 0:
        result = -1
else:
        result = 0
```

An `if` statement is permitted only within a function definition.
An `if` statement at top level results in a static error.

### For loops

A `for` loop evaluates its operand, which must be an iterable value.
Then, for each element of the iterable's sequence, the loop assigns
the successive element values to one or more variables and executes a
list of statements, the _loop body_.

```text
ForStmt = 'for' LoopVariables 'in' Expression ':' Suite .
```

Example:

```python
for x in range(10):
   print(10)
```

The assignment of each value to the loop variables follows the same
rules as an ordinary assignment.  In this example, two-element lists
are repeatedly assigned to the pair of variables (a, i):

```python
for a, i in [["a", 1], ["b", 2], ["c", 3]]:
  print(a, i)                          # prints "a 1", "b 2", "c 3"
```

Because Skylark loops always iterate over a finite sequence, they are
guaranteed to terminate, unlike loops in most languages which can
execute an arbitrary and perhaps unbounded number of iterations.

Within the body of a `for` loop, `break` and `continue` statements may
be used to stop the execution of the loop or advance to the next
iteration.

In Skylark, a `for` loop is permitted only within a function definition.
A `for` loop at top level results in a static error.


### Break and Continue

The `break` and `continue` statements terminate the current iteration
of a `for` loop.  Whereas the `continue` statement resumes the loop at
the next iteration, a `break` statement terminates the entire loop.

```text
BreakStmt    = 'break' .
ContinueStmt = 'continue' .
```

Example:

```python
for x in range(10):
    if x%2 == 1:
        continue        # skip odd numbers
    if x > 7:
        break           # stop at 8
    print(x)            # prints "0", "2", "4", "6"
```

Both statements affect only the innermost lexically enclosing loop.
It is a static error to use a `break` or `continue` statement outside a
loop.


### Load statements

The `load` statement loads another Skylark module, extracts one or
more values from it, and binds them to names in the current module.

<!--
The awkwardness of load statements is a consequence of staying a
strict subset of Python syntax, which allows reuse of existing tools
such as editor support. Python import statements are inadequate for
Skylark because they don't allow arbitrary file names for module names.
-->

Syntactically, a load statement looks like a function call `load(...)`.

```text
LoadStmt = 'load' '(' string {',' [identifier '='] string} [','] ')' .
```

A load statement requires at least two "arguments".
The first must be a literal string; it identifies the module to load.
Its interpretation is determined by the application into which the
Skylark interpreter is embedded, and is not specified here.

During execution, the application determines what action to take for a
load statement.
A typical implementation locates and executes a Skylark file,
populating a cache of files executed so far to avoid duplicate work,
to obtain a module, which is a mapping from global names to values.

The remaining arguments are a mixture of literal strings, such as
`"x"`, or named literal strings, such as `y="x"`.

The literal string (`"x"`), which must denote a valid identifier not
starting with `_`, specifies the name to extract from the loaded
module.  In effect, names starting with `_` are not exported.
The name (`y`) specifies the local name;
if no name is given, the local name matches the quoted name.

```python
load("module.sky", "x", "y", "z")       # assigns x, y, and z
load("module.sky", "x", y2="y", "z")    # assigns x, y2, and z
```

A load statement within a function is a static error.


## Module execution

Each Skylark file defines a _module_, which is a mapping from the
names of global variables to their values.
When a Skylark file is executed, whether directly by the application
or indirectly through a `load` statement, a new Skylark thread is
created, and this thread executes all the top-level statements in the
file.
Because if-statements and for-loops cannot appear outside of a function,
control flows from top to bottom.

If execution reaches the end of the file, module initialization is
successful.
At that point, the value of each of the module's global variables is
frozen, rendering subsequent mutation impossible.
The module is then ready for use by another Skylark thread, such as
one executing a load statement.
Such threads may access values or call functions defined in the loaded
module.

A Skylark thread may carry state on behalf of the application into
which it is embedded, and application-defined functions may behave
differently depending on this thread state.
Because module initialization always occurs in a new thread, thread
state is never carried from a higher-level module into a lower-level
one.
The initialization behavior of a module is thus independent of
whichever module triggered its initialization.

If a Skylark thread encounters an error, execution stops and the error
is reported to the application, along with a backtrace showing the
stack of active function calls at the time of the error.
If an error occurs during initialization of a Skylark module, any
active `load` statements waiting for initialization of the module also
fail.

Skylark provides no mechanism by which errors can be handled within
the language.


## Built-in constants and functions

The outermost block of the Skylark environment is known as the "universe" block.
It defines a number of fundamental values and functions needed by all Skylark programs,
such as `None`, `True`, `False`, and `len`.

These names are not reserved words so Skylark programs are free to
redefine them in a smaller block such as a function body or even at
the top level of a module.  However, doing so may be confusing to the
reader.  Nonetheless, this rule permits names to be added to the
universe block in later versions of the language without breaking
existing programs.


### None

`None` is the distinguished value of the type `NoneType`.

### True and False

`True` and `False` are the two values of type `bool`.

### any

`any(x)` returns `True` if any element of the iterable sequence x is true.
If the iterable is empty, it returns `False`.

### all

`all(x)` returns `False` if any element of the iterable sequence x is false.
If the iterable is empty, it returns `True`.

### bool

`bool(x)` interprets `x` as a Boolean value---`True` or `False`.
With no argument, `bool()` returns `False`.

### dict

`dict` creates a dictionary.  It accepts up to one positional
argument, which is interpreted as an iterable of two-element
sequences (pairs), each specifying a key/value pair in
the resulting dictionary.

`dict` also accepts any number of keyword arguments, each of which
specifies a key/value pair in the resulting dictionary;
each keyword is treated as a string.

```python
dict()                          # {}, empty dictionary
dict([(1, 2), (3, 4)])          # {1: 2, 3: 4}
dict([(1, 2), ["a", "b"]])      # {1: 2, "a": "b"}
dict(one=1, two=2)              # {"one": 1, "two", 1}
dict([(1, 2)], x=3)             # {1: 2, "x": 3}
```

With no arguments, `dict()` returns a new empty dictionary.

`dict(x)` where x is a dictionary returns a new copy of x.

### dir

`dir(x)` returns a list of the names of the attributes (fields and methods) of its operand.
The attributes of a value `x` are the names `f` such that `x.f` is a valid expression.

For example,

```python
dir("hello")                    # ['capitalize', 'count', ...], the methods of a string
```

Several types known to the interpreter, such as list, string, and dict, have methods, but none have fields.
However, an application may define types with fields that may be read or set by statements such as these:

```text
y = x.f
x.f = y
```

### enumerate

`enumerate(x)` returns a list of (index, value) pairs, each containing
successive values of the iterable sequence xand the index of the value
within the sequence.

The optional second parameter, `start`, specifies an integer value to
add to each index.

```python
enumerate(["zero", "one", "two"])               # [(0, "zero"), (1, "one"), (2, "two")]
enumerate(["one", "two"], 1)                    # [(1, "one"), (2, "two")]
```

### getattr

`getattr(x, name)` returns the value of the attribute (field or method) of x named `name`.
It is a dynamic error if x has no such attribute.

`getattr(x, "f")` is equivalent to `x.f`.

```python
getattr("banana", "split")("a")	       # ["b", "n", "n", ""], equivalent to "banana".split("a")
```

### hasattr

`hasattr(x, name)` reports whether x has an attribute (field or method) named `name`.

### hash

`hash(x)` returns an integer hash value for a string x such that `x == y`
implies `hash(x) == hash(y)`.

### int

`int(x[, base])` interprets its argument as an integer.

If `x` is an `int`, the result is `x`.
If `x` is a `bool`, the result is 0 for `False` or 1 for `True`.

If `x` is a string, it is interpreted as a sequence of digits in the specified
base, decimal by default. If `base` is zero, `x` is interpreted like an integer
literal, the base being inferred from an optional base marker such as `0b`,
`0o`, or `0x` preceding the first digit. These markers may also be used if
`base` is the corresponding base. Irrespective of base, the string may start
with an optional `+` or `-` sign indicating the sign of the result.

```python
int("21")          # 21
int("1234", 16)    # 4660
int("0x1234", 16)  # 4660
int("0x1234", 0)   # 4660
int("0x1234")      # error (invalid base 10 number)
```

### len

`len(x)` returns the number of elements in its argument.

It is a dynamic error if its argument is not a sequence.

### list

`list` constructs a list.

`list(x)` returns a new list containing the elements of the
iterable sequence x.

With no argument, `list()` returns a new empty list.

### max

`max(x)` returns the greatest element in the iterable sequence x.

It is an error if any element does not support ordered comparison,
or if the sequence is empty.

The optional named parameter `key` specifies a function to be applied
to each element prior to comparison.

```python
max([3, 1, 4, 1, 5, 9])                         # 9
max("two", "three", "four")                     # "two", the lexicographically greatest
max("two", "three", "four", key=len)            # "three", the longest
```

### min

`min(x)` returns the least element in the iterable sequence x.

It is an error if any element does not support ordered comparison,
or if the sequence is empty.

```python
min([3, 1, 4, 1, 5, 9])                         # 1
min("two", "three", "four")                     # "four", the lexicographically least
min("two", "three", "four", key=len)            # "two", the shortest
```

### print

`print(*args, **kwargs)` prints its arguments, followed by a newline.
Arguments are formatted as if by `str(x)` and separated with a space.
Keyword arguments are preceded by their name.

Example:

```python
print(1, "hi", x=3)	# "1 hi x=3\n"
```

Typically the formatted string is printed to the standard error file,
but the exact behavior is a property of the Skylark thread and is
determined by the host application.

### range

`range` returns an immutable sequence of integers defined by the specified interval and stride.

```python
range(stop)                             # equivalent to range(0, stop)
range(start, stop)                      # equivalent to range(start, stop, 1)
range(start, stop, step)
```

`range` requires between one and three integer arguments.
With one argument, `range(stop)` returns the ascending sequence of non-negative integers less than `stop`.
With two arguments, `range(start, stop)` returns only integers not less than `start`.

With three arguments, `range(start, stop, step)` returns integers
formed by successively adding `step` to `start` until the value meets or passes `stop`.
A call to `range` fails if the value of `step` is zero.

A call to `range` does not materialize the entire sequence, but
returns a fixed-size value of type `"range"` that represents the
parameters that define the sequence.
The `range` value is iterable and may be indexed efficiently.

```python
list(range(10))                         # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list(range(3, 10))                      # [3, 4, 5, 6, 7, 8, 9]
list(range(3, 10, 2))                   # [3, 5, 7, 9]
list(range(10, 3, -2))                  # [10, 8, 6, 4]
```

The `len` function applied to a `range` value returns its length.
The truth value of a `range` value is `True` if its length is non-zero.

Range values are comparable: two `range` values compare equal if they
denote the same sequence of integers, even if they were created using
different parameters.

Range values are not hashable.  <!-- should they be? -->

The `str` function applied to a `range` value yields a string of the
form `range(10)`, `range(1, 10)`, or `range(1, 10, 2)`.

The `x in y` operator, where `y` is a range, reports whether `x` is equal to
some member of the sequence `y`; the operation fails unless `x` is a
number.

### repr

`repr(x)` formats its argument as a string.

All strings in the result are double-quoted.

```python
repr(1)                 # '1'
repr("x")               # '"x"'
repr([1, "x"])          # '[1, "x"]'
```

### reversed

`reversed(x)` returns a new list containing the elements of the iterable sequence x in reverse order.

```python
reversed(range(5))                              # [4, 3, 2, 1, 0]
reversed({"one": 1, "two": 2}.keys())           # ["two", "one"]
```

### sorted

`sorted(x)` returns a new list containing the elements of the iterable sequence x,
in sorted order.  The sort algorithm is stable.

```python
sorted([3, 1, 4, 1, 5, 9])              # [1, 1, 3, 4, 5, 9]

sorted(["two", "three", "four"])        # ["three", "two", "four"]
```

### str

`str(x)` formats its argument as a string.

If x is a string, the result is x (without quotation).
All other strings, such as elements of a list of strings, are double-quoted.

```python
str(1)                          # '1'
str("x")                        # 'x'
str([1, "x"])                   # '[1, "x"]'
```

### tuple

`tuple(x)` returns a tuple containing the elements of the iterable x.

With no arguments, `tuple()` returns the empty tuple.

### type

type(x) returns a string describing the type of its operand.

```python
type(None)              # "NoneType"
type(0)                 # "int"
```

### zip

`zip()` returns a new list of n-tuples formed from corresponding
elements of each of the n iterable sequences provided as arguments to
`zip`.  That is, the first tuple contains the first element of each of
the sequences, the second element contains the second element of each
of the sequences, and so on.  The result list is only as long as the
shortest of the input sequences.

```python
zip()                                   # []
zip(range(5))                           # [(0,), (1,), (2,), (3,), (4,)]
zip(range(5), "abc")                    # [(0, "a"), (1, "b"), (2, "c")]
```

## Built-in methods

This section lists the methods of built-in types.  Methods are selected
using [dot expressions](#dot-expressions).
For example, strings have a `count` method that counts
occurrences of a substring; `"banana".count("a")` yields `3`.

As with built-in functions, built-in methods accept only positional
arguments except where noted.
The parameter names serve merely as documentation.


<a id='dict·get'></a>
### dict·get

`D.get(key[, default])` returns the dictionary value corresponding to the given key.
If the dictionary contains no such value, `get` returns `None`, or the
value of the optional `default` parameter if present.

`get` fails if `key` is unhashable, or the dictionary is frozen or has active iterators.

```python
x = {"one": 1, "two": 2}
x.get("one")                            # 1
x.get("three")                          # None
x.get("three", 0)                       # 0
```

<a id='dict·items'></a>
### dict·items

`D.items()` returns a new list of key/value pairs, one per element in
dictionary D, in the same order as they would be returned by a `for` loop.

```python
x = {"one": 1, "two": 2}
x.items()                               # [("one", 1), ("two", 2)]
```

<a id='dict·keys'></a>
### dict·keys

`D.keys()` returns a new list containing the keys of dictionary D, in the
same order as they would be returned by a `for` loop.

```python
x = {"one": 1, "two": 2}
x.keys()                               # ["one", "two"]
```

<a id='dict·pop'></a>
### dict·pop

`D.pop(key[, default])` returns the value corresponding to the specified
key, and removes it from the dictionary.  If the dictionary contains no
such value, and the optional `default` parameter is present, `pop`
returns that value; otherwise, it fails.

`pop` fails if `key` is unhashable, or the dictionary is frozen or has active iterators.

```python
x = {"one": 1, "two": 2}
x.pop("one")                            # 1
x                                       # {"two": 2}
x.pop("three", 0)                       # 0
x.pop("four")                           # error: missing key
```

<a id='dict·popitem'></a>
### dict·popitem

`D.popitem()` returns the first key/value pair, removing it from the dictionary.

`popitem` fails if the dictionary is empty, frozen, or has active iterators.

```python
x = {"one": 1, "two": 2}
x.popitem()                             # ("one", 1)
x.popitem()                             # ("two", 2)
x.popitem()                             # error: empty dict
```

<a id='dict·setdefault'></a>
### dict·setdefault

`D.setdefault(key[, default])` returns the dictionary value corresponding to the given key.
If the dictionary contains no such value, `setdefault`, like `get`,
returns `None` or the value of the optional `default` parameter if
present; `setdefault` additionally inserts the new key/value entry into the dictionary.

`setdefault` fails if the key is unhashable, or if the dictionary is frozen or has active iterators.

```python
x = {"one": 1, "two": 2}
x.setdefault("one")                     # 1
x.setdefault("three", 0)                # 0
x                                       # {"one": 1, "two": 2, "three": 0}
x.setdefault("four")                    # None
x                                       # {"one": 1, "two": 2, "three": None}
```

<a id='dict·update'></a>
### dict·update

`D.update([pairs][, name=value[, ...])` makes a sequence of key/value
insertions into dictionary D, then returns `None.`

If the positional argument `pairs` is present, it must be `None`,
another `dict`, or some other iterable.
If it is another `dict`, then its key/value pairs are inserted into D.
If it is an iterable, it must provide a sequence of pairs (or other iterables of length 2),
each of which is treated as a key/value pair to be inserted into D.

For each `name=value` argument present, the name is converted to a
string and used as the key for an insertion into D, with its corresponding
value being `value`.

`update` fails if the dictionary is frozen or has active iterators.

```python
x = {}
x.update([("a", 1), ("b", 2)], c=3)
x.update({"d": 4})
x.update(e=5)
x                                       # {"a": 1, "b": "2", "c": 3, "d": 4, "e": 5}
```

<a id='dict·values'></a>
### dict·values

`D.values()` returns a new list containing the dictionary's values, in the
same order as they would be returned by a `for` loop over the
dictionary.

```python
x = {"one": 1, "two": 2}
x.values()                              # [1, 2]
```

<a id='list·append'></a>
### list·append

`L.append(x)` appends `x` to the list L, and returns `None`.

`append` fails if the list is frozen or has active iterators.

```python
x = []
x.append(1)                             # None
x.append(2)                             # None
x.append(3)                             # None
x                                       # [1, 2, 3]
```

<a id='list·clear'></a>
### list·clear

`L.clear()` removes all the elements of the list L and returns `None`.
It fails if the list is frozen or if there are active iterators.

```python
x = [1, 2, 3]
x.clear()                               # None
x                                       # []
```

<a id='list·extend'></a>
### list·extend

`L.extend(x)` appends the elements of `x`, which must be iterable, to
the list L, and returns `None`.

`extend` fails if `x` is not iterable, or if the list L is frozen or has active iterators.

```python
x = []
x.extend([1, 2, 3])                     # None
x.extend(["foo"])                       # None
x                                       # [1, 2, 3, "foo"]
```

<a id='list·index'></a>
### list·index

`L.insert(x[, start[, end]])` finds `x` within the list L and returns its index.

The optional `start` and `end` parameters restrict the portion of
list L that is inspected.  If provided and not `None`, they must be list
indices of type `int`. If an index is negative, `len(L)` is effectively
added to it, then if the index is outside the range `[0:len(L)]`, the
nearest value within that range is used; see [Indexing](#indexing).

`insert` fails if `x` is not found in L, or if `start` or `end`
is not a valid index (`int` or `None`).

```python
x = ["b", "a", "n", "a", "n", "a"]
x.index("a")                            # 1 (bAnana)
x.index("a", 2)                         # 3 (banAna)
x.index("a", -2)                        # 5 (bananA)
```

<a id='list·insert'></a>
### list·insert

`L.insert(i, x)` inserts the value `x` in the list L at index `i`, moving
higher-numbered elements along by one.  It returns `None`.

As usual, the index `i` must be an `int`. If its value is negative,
the length of the list is added, then its value is clamped to the
nearest value in the range `[0:len(L)]` to yield the effective index.

`insert` fails if the list is frozen or has active iterators.

```python
x = ["b", "c", "e"]
x.insert(0, "a")                        # None
x.insert(-1, "d")                       # None
x                                       # ["a", "b", "c", "d", "e"]
```

<a id='list·pop'></a>
### list·pop

`L.pop([index])` removes and returns the last element of the list L, or,
if the optional index is provided, at that index.

`insert` fails if the index is negative or not less than the length of
the list, of if the list is frozen or has active iterators.

```python
x = [1, 2, 3]
x.pop()                                 # 3
x.pop()                                 # 2
x                                       # [1]
```

<a id='list·remove'></a>
### list·remove

`L.remove(x)` removes the first occurrence of the value `x` from the list L, and returns `None`.

`remove` fails if the list does not contain `x`, is frozen, or has active iterators.

```python
x = [1, 2, 3, 2]
x.remove(2)                             # None (x == [1, 3, 2])
x.remove(2)                             # None (x == [1, 3])
x.remove(2)                             # error: element not found
```

<a id='string·capitalize'></a>
### string·capitalize

`S.capitalize()` returns a copy of string S with all Unicode letters
that begin words changed to their title case.

```python
"hello, world!".capitalize()		# "Hello, World!"
```

<a id='string·count'></a>
### string·count

`S.count(sub[, start[, end]])` returns the number of occcurences of
`sub` within the string S, or, if the optional substring indices
`start` and `end` are provided, within the designated substring of S.
They are interpreted according to Skylark's [indexing conventions](#indexing).

```python
"hello, world!".count("o")              # 2
"hello, world!".count("o", 7, 12)       # 1  (in "world")
```

<a id='string·endswith'></a>
### string·endswith

`S.endswith(suffix)` reports whether the string S has the specified suffix.

```python
"filename.sky".endswith(".sky")         # True
```

<a id='string·find'></a>
### string·find

`S.find(sub[, start[, end]])` returns the index of the first
occurrence of the substring `sub` within S.

If either or both of `start` or `end` are specified,
they specify a subrange of S to which the search should be restricted.
They are interpreted according to Skylark's [indexing conventions](#indexing).

If no occurrence is found, `found` returns -1.

```python
"bonbon".find("on")             # 1
"bonbon".find("on", 2)          # 4
"bonbon".find("on", 2, 5)       # -1
```

<a id='string·format'></a>
### string·format

`S.format(*args, **kwargs)` returns a version of the format string S
in which bracketed portions `{...}` are replaced
by arguments from `args` and `kwargs`.

Within the format string, a pair of braces `{{` or `}}` is treated as
a literal open or close brace.
Each unpaired open brace must be matched by a close brace `}`.
The optional text between corresponding open and close braces
specifies which argument to use.

```text
{}
{field}
```

The *field name* may be either a decimal number or a keyword.
A number is interpreted as the index of a positional argument;
a keyword specifies the value of a keyword argument.
If all the numeric field names form the sequence 0, 1, 2, and so on,
they may be omitted and those values will be implied; however,
the explicit and implicit forms may not be mixed.

```python
"a{x}b{y}c{}".format(1, x=2, y=3)               # "a2b3c1"
"a{}b{}c".format(1, 2)                          # "a1b2c"
"({1}, {0})".format("zero", "one")              # "(one, zero)"
```

<a id='string·index'></a>
### string·index

`S.index(sub[, start[, end]])` returns the index of the first
occurrence of the substring `sub` within S, like `S.find`, except
that if the substring is not found, the operation fails.

```python
"bonbon".index("on")             # 1
"bonbon".index("on", 2)          # 4
"bonbon".index("on", 2, 5)       # error: substring not found  (in "nbo")
```

<a id='string·isalnum'></a>
### string·isalnum

`S.isalnum()` reports whether the string S is non-empty and consists only
Unicode letters and digits.

```python
"base64".isalnum()              # True
"Catch-22".isalnum()            # False
```

<a id='string·isalpha'></a>
### string·isalpha

`S.isalpha()` reports whether the string S is non-empty and consists only of Unicode letters.

```python
"ABC".isalpha()                 # True
"Catch-22".isalpha()            # False
"".isalpha()                    # False
```

<a id='string·isdigit'></a>
### string·isdigit

`S.isdigit()` reports whether the string S is non-empty and consists only of Unicode digits.

```python
"123".isdigit()                 # True
"Catch-22".isdigit()            # False
"".isdigit()                    # False
```

<a id='string·islower'></a>
### string·islower

`S.islower()` reports whether the string S contains at least one cased Unicode
letter, and all such letters are lowercase.

```python
"hello, world".islower()        # True
"Catch-22".islower()            # False
"123".islower()                 # False
```

<a id='string·isspace'></a>
### string·isspace

`S.isspace()` reports whether the string S is non-empty and consists only of Unicode spaces.

```python
"    ".isspace()                # True
"\r\t\n".isspace()              # True
"".isspace()                    # False
```

<a id='string·istitle'></a>
### string·istitle

`S.istitle()` reports whether the string S contains at least one cased Unicode
letter, and all such letters that begin a word are in title case.

```python
"Hello, World!".istitle()       # True
"Catch-22".istitle()            # True
"HAL-9000".istitle()            # False
"123".istitle()                 # False
```

<a id='string·isupper'></a>
### string·isupper

`S.isupper()` reports whether the string S contains at least one cased Unicode
letter, and all such letters are uppercase.

```python
"HAL-9000".isupper()            # True
"Catch-22".isupper()            # False
"123".isupper()                 # False
```

<a id='string·join'></a>
### string·join

`S.join(iterable)` returns the string formed by concatenating each
element of its argument, with a copy of the string S between
successive elements. The argument must be an iterable whose elements
are strings.

```python
", ".join(["one", "two", "three"])      # "one, two, three"
"a".join("ctmrn")                       # "catamaran"
```

<a id='string·lower'></a>
### string·lower

`S.lower()` returns a copy of the string S with letters converted to lowercase.

```python
"Hello, World!".lower()                 # "hello, world!"
```

<a id='string·lstrip'></a>
### string·lstrip

`S.lstrip()` returns a copy of the string S with leading whitespace removed.

```python
"  hello  ".lstrip()                    # "  hello"
```

<a id='string·partition'></a>
### string·partition

`S.partition(x)` splits string S into three parts and returns them as
a tuple: the portion before the first occurrence of string `x`, `x` itself,
and the portion following it.
If S does not contain `x`, `partition` returns `(S, "", "")`.

`partition` fails if `x` is not a string, or is the empty string.

```python
"one/two/three".partition("/")		# ("one", "/", "two/three")
```

<a id='string·replace'></a>
### string·replace

`S.replace(old, new[, count])` returns a copy of string S with all
occurrences of substring `old` replaced by `new`. If the optional
argument `count`, which must be an `int`, is non-negative, it
specifies a maximum number of occurrences to replace.

```python
"banana".replace("a", "o")		# "bonono"
"banana".replace("a", "o", 2)		# "bonona"
```

<a id='string·rfind'></a>
### string·rfind

`S.rfind(sub[, start[, end]])` returns the index of the substring `sub` within
S, like `S.find`, except that `rfind` returns the index of the substring's
_last_ occurrence.

```python
"bonbon".rfind("on")             # 4
"bonbon".rfind("on", None, 5)    # 1
"bonbon".rfind("on", 2, 5)       # -1
```

<a id='string·rindex'></a>
### string·rindex

`S.rindex(sub[, start[, end]])` returns the index of the substring `sub` within
S, like `S.index`, except that `rindex` returns the index of the substring's
_last_ occurrence.

```python
"bonbon".rindex("on")             # 4
"bonbon".rindex("on", None, 5)    # 1                           (in "bonbo")
"bonbon".rindex("on", 2, 5)       # error: substring not found  (in "nbo")
```

<a id='string·rpartition'></a>
### string·rpartition

`S.rpartition(x)` is like `partition`, but splits `S` at the last occurrence of `x`.

```python
"one/two/three".partition("/")		# ("one/two", "/", "three")
```

<a id='string·rsplit'></a>
### string·rsplit

`S.rsplit([sep[, maxsplit]])` splits a string into substrings like `S.split`,
except that when a maximum number of splits is specified, `rsplit` chooses the
rightmost splits.

```python
"banana".rsplit("n")                         # ["ba", "a", "a"]
"banana".rsplit("n", 1)                      # ["bana", "a"]
"one two  three".rsplit(None, 1)             # ["one two", "three"]
```

<a id='string·rstrip'></a>
### string·rstrip

`S.rstrip()` returns a copy of the string S with trailing whitespace removed.

```python
"  hello  ".rstrip()                    # "hello  "
```

<a id='string·split'></a>
### string·split

`S.split([sep [, maxsplit]])` returns the list of substrings of S,
splitting at occurrences of the delimiter string `sep`.

Consecutive occurrences of `sep` are considered to delimit empty
strings, so `'food'.split('o')` returns `['f', '', 'd']`.
Splitting an empty string with a specified separator returns `['']`.
If `sep` is the empty string, `split` fails.

If `sep` is not specified or is `None`, `split` uses a different
algorithm: it removes all leading spaces from S
(or trailing spaces in the case of `rsplit`),
then splits the string around each consecutive non-empty sequence of
Unicode white space characters.

If S consists only of white space, `split` returns the empty list.

If `maxsplit` is given and non-negative, it specifies a maximum number of splits.

```python
"one two  three".split()                    # ["one", "two", "three"]
"one two  three".split(" ")                 # ["one", "two", "", "three"]
"one two  three".split(None, 1)             # ["one", "two  three"]
"banana".split("n")                         # ["ba", "a", "a"]
"banana".split("n", 1)                      # ["ba", "ana"]
```

<a id='string·splitlines'></a>
### string·splitlines

`S.splitlines([keepends])` returns a list whose elements are the
successive lines of S, that is, the strings formed by splitting S at
line terminators (currently assumed to be a single newline, `\n`,
regardless of platform).

The optional argument, `keepends`, is interpreted as a Boolean.
If true, line terminators are preserved in the result, though
the final element does not necessarily end with a line terminator.

```python
"one\n\ntwo".splitlines()       # ["one", "", "two"]
"one\n\ntwo".splitlines(True)   # ["one\n", "\n", "two"]
```


<a id='string·startswith'></a>
### string·startswith

`S.startswith(suffix)` reports whether the string S has the specified prefix.

```python
"filename.sky".startswith("filename")         # True
```

<a id='string·strip'></a>
### string·strip

`S.strip()` returns a copy of the string S with leading and trailing whitespace removed.

```python
"  hello  ".strip()                     # "hello"
```

<a id='string·title'></a>
### string·title

`S.lower()` returns a copy of the string S with letters converted to titlecase.

Letters are converted to uppercase at the start of words, lowercase elsewhere.

```python
"hElLo, WoRlD!".title()                 # "Hello, World!"
```

<a id='string·upper'></a>
### string·upper

`S.lower()` returns a copy of the string S with letters converted to lowercase.

```python
"Hello, World!".upper()                 # "HELLO, WORLD!"
```


## Grammar reference

```text
File = {Statement | newline} eof .

Statement = DefStmt | IfStmt | ForStmt | SimpleStmt .

DefStmt = 'def' identifier '(' [Parameters [',']] ')' ':' Suite .
# NOTE: trailing comma is not permitted if the last argument is `'*' identifier` or `'**' identifier`.

Parameters = Parameter {',' Parameter}.

Parameter = identifier | identifier '=' Test | '*' identifier | '**' identifier .

IfStmt = 'if' Test ':' Suite {'elif' Test ':' Suite} ['else' ':' Suite] .

ForStmt = 'for' LoopVariables 'in' Expression ':' Suite .

Suite = [newline indent {Statement} outdent] | SimpleStmt .

SimpleStmt = SmallStmt {';' SmallStmt} [';'] '\n' .
# NOTE: '\n' optional at EOF

SmallStmt = ReturnStmt
          | BreakStmt | ContinueStmt | PassStmt
          | AssignStmt
          | ExprStmt
          | LoadStmt
          .

ReturnStmt   = 'return' [Expression] .
BreakStmt    = 'break' .
ContinueStmt = 'continue' .
PassStmt     = 'pass' .
AssignStmt   = Expression ('=' | '+=' | '-=' | '*=' | '/=' | '//=' | '%=') Expression .
ExprStmt     = Expression .

LoadStmt = 'load' '(' string {',' [identifier '='] string} [','] ')' .

Test = IfExpr
     | PrimaryExpr
     | UnaryExpr
     | BinaryExpr
     .

IfExpr = Test 'if' Test 'else' Test .

PrimaryExpr = Operand
            | PrimaryExpr DotSuffix
            | PrimaryExpr CallSuffix
            | PrimaryExpr SliceSuffix
            .

Operand = identifier
        | int | string
        | ListExpr | ListComp
        | DictExpr | DictComp
        | '(' [Expression [',']] ')'
        | '-' PrimaryExpr
        .

DotSuffix   = '.' identifier .
SliceSuffix = '[' [Expression] [':' Test [':' Test]] ']' .
CallSuffix  = '(' [Arguments [',']] ')' .
# NOTE: trailing comma is not permitted if the last argument is `'*' Test` or `'**' Test`.

Arguments = Argument {',' Argument} .
Argument  = Test | identifier '=' Test | '*' Test | '**' Test .

ListExpr = '[' [Expression [',']] ']' .
ListComp = '[' Test {CompClause} ']'.

DictExpr = '{' [Entries [',']] '}' .
DictComp = '{' Entry {CompClause} '}' .
Entries  = Entry {',' Entry} .
Entry    = Test ':' Test .

CompClause = 'for' LoopVariables 'in' Test | 'if' Test .

UnaryExpr = 'not' Test .

BinaryExpr = Test {Binop Test} .

Binop = 'or'
      | 'and'
      | '==' | '!=' | '<' | '>' | '<=' | '>=' | 'in' | 'not' 'in'
      | '|'
      | '&'
      | '-' | '+'
      | '*' | '%' | '/' | '//'
      .

Expression = Test {',' Test} .
# NOTE: trailing comma permitted only when within [...] or (...).

LoopVariables = PrimaryExpr {',' PrimaryExpr} .
```

Tokens:

- spaces: newline, eof, indent, outdent.
- identifier.
- literals: string, int.
- plus all quoted tokens such as '+=', 'return'.

Notes:

- Ambiguity is resolved using operator precedence.
- The grammar does not enforce the legal order of params and args,
  nor that the first compclause must be a 'for'.
