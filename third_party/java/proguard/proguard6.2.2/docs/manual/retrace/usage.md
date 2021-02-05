You can find the ReTrace jar in the `lib` directory of the ProGuard
distribution. To run ReTrace, just type:

`java -jar retrace.jar `\[*options...*\] *mapping\_file*
\[*stacktrace\_file*\]

Alternatively, the `bin` directory contains some short Linux and Windows
scripts containing this command. These are the arguments:

*mapping\_file*
: Specifies the name of the mapping file, produced by ProGuard with the
  option "[`-printmapping`](../usage.md#printmapping) *mapping\_file*", while
  obfuscating the application that produced the stack trace.

*stacktrace\_file*
: Optionally specifies the name of the file containing the stack trace. If
  no file is specified, a stack trace is read from the standard input. The
  stack trace must be encoded with UTF-8 encoding. Blank lines and
  unrecognized lines are ignored.

The following options are supported:

`-verbose`
: Specifies to print out more informative stack traces that include not only
  method names, but also method return types and arguments.

`-regex` *regular\_expression*

: Specifies the regular expression that is used to parse the lines in the
  stack trace. Specifying a different regular expression allows to
  de-obfuscate more general types of input than just stack traces. The default
  is suitable for stack traces produced by most JVMs:

        (?:.*?\bat\s+%c\.%m\s*\(%s(?::%l)?\)\s*(?:~\[.*\])?)|(?:(?:.*?[:"]\s+)?%c(?::.*)?)


  The regular expression is a Java regular expression (cfr. the
  documentation of `java.util.regex.Pattern`), with a few additional
  wildcards:

  | Wildcard | Description                                | Example
  |----------|--------------------------------------------|-------------------------------------------
  | `%c`     | matches a class name                       | `com.example.MyClass`
  | `%C`     | matches a class name with slashes          | `com/example/MyClass`
  | `%t`     | matches a field type or method return type | `com.example.MyClass[]`
  | `%f`     | matches a field name                       | `myField`
  | `%m`     | matches a method name                      | `myMethod`
  | `%a`     | matches a list of method arguments         | `boolean,int`
  | `%s`     | matches a source file name                 | `MyClass.java`
  | `%l`     | matches a line number inside a method      | `123`

  Elements that match these wildcards are de-obfuscated,
  when possible. Note that regular expressions must not contain any
  capturing groups. Use non-capturing groups instead: `(?:`...`)`

  The default expression for instance matches the following lines:

    Exception in thread "main" com.example.MyException: Some message
        at com.example.MyClass.myMethod(MyClass.java:123)


The restored stack trace is printed to the standard output. The
completeness of the restored stack trace depends on the presence of line
number tables in the obfuscated class files:

- If all line numbers have been preserved while obfuscating the
  application, ReTrace will be able to restore the stack
  trace completely.
- If the line numbers have been removed, mapping obfuscated method
  names back to their original names has become ambiguous. Retrace
  will list all possible original method names for each line in the
  stack trace. The user can then try to deduce the actual stack trace
  manually, based on the logic of the program.

Preserving line number tables is explained in detail in this
[example](../examples.md#stacktrace) in the ProGuard User Manual.

Source file names are currently restored based on the names of the
outer-most classes. If you prefer to keep the obfuscated name, you can
replace `%s` in the default regular expression by `.*`

Unobfuscated elements and obfuscated elements for which no mapping is
available will be left unchanged.
