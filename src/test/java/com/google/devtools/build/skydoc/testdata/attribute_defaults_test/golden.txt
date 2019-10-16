<!-- Generated with Stardoc: http://skydoc.bazel.build -->

<a name="#my_rule"></a>

## my_rule

<pre>
my_rule(<a href="#my_rule-name">name</a>, <a href="#my_rule-a">a</a>, <a href="#my_rule-b">b</a>, <a href="#my_rule-c">c</a>, <a href="#my_rule-d">d</a>, <a href="#my_rule-e">e</a>, <a href="#my_rule-f">f</a>, <a href="#my_rule-g">g</a>, <a href="#my_rule-h">h</a>, <a href="#my_rule-i">i</a>, <a href="#my_rule-j">j</a>, <a href="#my_rule-k">k</a>, <a href="#my_rule-l">l</a>, <a href="#my_rule-m">m</a>, <a href="#my_rule-n">n</a>, <a href="#my_rule-o">o</a>, <a href="#my_rule-p">p</a>, <a href="#my_rule-q">q</a>, <a href="#my_rule-r">r</a>, <a href="#my_rule-s">s</a>, <a href="#my_rule-t">t</a>, <a href="#my_rule-u">u</a>, <a href="#my_rule-v">v</a>, <a href="#my_rule-w">w</a>)
</pre>

This is my rule. It does stuff.

**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| name |  A unique name for this target.   | <a href="https://bazel.build/docs/build-ref.html#name">Name</a> | required |  |
| a |  Some bool   | Boolean | optional | False |
| b |  Some int   | Integer | optional | 2 |
| c |  Some int_list   | List of integers | optional | [0, 1] |
| d |  Some label   | <a href="https://bazel.build/docs/build-ref.html#labels">Label</a> | optional | //foo:bar |
| e |  Some label_keyed_string_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: Label -> String</a> | optional | {"//foo:bar": "hello", "//bar:baz": "goodbye"} |
| f |  Some label_list   | <a href="https://bazel.build/docs/build-ref.html#labels">List of labels</a> | optional | ["//foo:bar", "//bar:baz"] |
| g |  Some string   | String | optional | "" |
| h |  Some string_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: String -> String</a> | optional | {"animal": "bunny", "color": "orange"} |
| i |  Some string_list   | List of strings | optional | ["cat", "dog"] |
| j |  Some string_list_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: String -> List of strings</a> | optional | {"animal": ["cat", "bunny"], "color": ["blue", "orange"]} |
| k |  Some bool   | Boolean | required |  |
| l |  Some int   | Integer | required |  |
| m |  Some int_list   | List of integers | required |  |
| n |  Some label   | <a href="https://bazel.build/docs/build-ref.html#labels">Label</a> | required |  |
| o |  Some label_keyed_string_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: Label -> String</a> | required |  |
| p |  Some label_list   | <a href="https://bazel.build/docs/build-ref.html#labels">List of labels</a> | required |  |
| q |  Some string   | String | required |  |
| r |  Some string_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: String -> String</a> | required |  |
| s |  Some string_list   | List of strings | required |  |
| t |  Some string_list_dict   | <a href="https://bazel.build/docs/skylark/lib/dict.html">Dictionary: String -> List of strings</a> | required |  |
| u |  -   | String | optional | "" |
| v |  -   | <a href="https://bazel.build/docs/build-ref.html#labels">Label</a> | optional | None |
| w |  -   | Integer | optional | 0 |


<a name="#my_aspect"></a>

## my_aspect

<pre>
my_aspect(<a href="#my_aspect-name">name</a>, <a href="#my_aspect-y">y</a>, <a href="#my_aspect-z">z</a>)
</pre>

This is my aspect. It does stuff.

**ASPECT ATTRIBUTES**


| Name | Type |
| :-------------: | :-------------: |
| deps| String |
| attr_aspect| String |


**ATTRIBUTES**


| Name  | Description | Type | Mandatory | Default |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| name |  A unique name for this target.   | <a href="https://bazel.build/docs/build-ref.html#name">Name</a> | required |   |
| y |  some string   | String | optional |  "why" |
| z |  -   | String | required |   |


