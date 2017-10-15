Jar Jar Links is a utility that makes it easy to repackage Java
libraries and embed them into your own distribution. This is useful
for two reasons:

You can easily ship a single jar file with no external dependencies.

You can avoid problems where your library depends on a specific
version of a library, which may conflict with the dependencies of
another library.

How does it work?

Jar Jar Links includes an Ant task that extends the built-in jar
task. The normal zipfileset element is used to embed jar files. A
new rule element is added which uses wildcards patterns to rename
the embedded class files. Bytecode transformation (via ASM) is used
to change references to the renamed classes, and special handling is
provided for moving resource files and transforming string literals.

Using with ant
--------------

In our imaginary project, the Ant "jar" target looks like:

```
<target name="jar" depends="compile">
    <jar jarfile="dist/example.jar">
        <fileset dir="build/main"/>
    </jar>
</target>
```

To use Jar Jar Links, we define a new task named "jarjar", and
substitute it wherever we used the jar task. Because the JarJarTask
class extends the normal Ant Jar task, you can use jarjar without
any of its additional features, if you want:

```
<target name="jar" depends="compile">
    <taskdef name="jarjar" classname="com.tonicsystems.jarjar.JarJarTask"
        classpath="lib/jarjar.jar"/>
    <jarjar jarfile="dist/example.jar">
        <fileset dir="build/main"/>
    </jarjar>
</target>
```

Just like with the "jar" task, we can include the contents of another
jar file using the "zipfileset" element. But simply including another
projects classes is not good enough to avoid jar hell, since the class
names remain unchanged and can still conflict with other versions.

To rename the classes, JarJarTask adds a new "rule" element. The
rule takes a "pattern" attribute, which uses wildcards to match
against class names, and a "result" attribute, which describes how
to transform the matched names.

In this example we include classes from jaxen.jar and add a rule
that changes any class name starting with "org.jaxen" to start with
"org.example.jaxen" instead (in our imaginary world we control the
example.org domain):

```
<target name="jar" depends="compile">
    <taskdef name="jarjar" classname="com.tonicsystems.jarjar.JarJarTask"
        classpath="lib/jarjar.jar"/>
    <jarjar jarfile="dist/example.jar">
        <fileset dir="build/main"/>
        <zipfileset src="lib/jaxen.jar"/>
        <rule pattern="org.jaxen.**" result="org.example.@1"/>
    </jarjar>
</target>
```

The ** in the pattern means to match against any valid package
substring. To match against a single package component (by excluding
dots (.) from the match), a single * may be used instead.

The @1 in the result is a reference to the ** portion of the rule. For
every * or ** in the rule, a numbered reference is available for use
in the result. References are numbered from left to right, starting
with @1, then @2, and so on.

The special @0 reference refers to the entire class name.

Using with gradle
-----------------

```
	dependencies {
		// Use jarjar.repackage in place of a dependency notation.
		compile jarjar.repackage {
			from 'com.google.guava:guava:18.0'

			classDelete "com.google.common.base.**"

			classRename "com.google.**" "org.private.google.@1"
		}
	}
```

See (jarjar-gradle/example/build.gradle) for some complete examples.

Using from the command line
---------------------------

From the command-line

```
java -jar jarjar.jar [help]
```

Prints this help message.

```
java -jar jarjar.jar strings <cp>
```

Dumps all string literals in classpath `<cp>`. Line numbers will be
included if the classes have debug information.

```
java -jar jarjar.jar find <level> <cp1> [<cp2>]
```

Prints dependencies on classpath `<cp2>` in classpath `<cp1>`. If `<cp2>`
is omitted, `<cp1>` is used for both arguments.

The level argument must be class or jar. The former prints dependencies
between individual classes, while the latter only prints jar->jar
dependencies. A "jar" in this context is actually any classpath
component, which can be a jar file, a zip file, or a parent directory
(see below).

```
java -jar jarjar.jar process <rulesFile> <inJar> <outJar>
```

Transform the `<inJar>` jar file, writing a new jar file to `<outJar>`. Any
existing file named by `<outJar>` will be deleted.

The transformation is defined by a set of rules in the file specified
by the rules argument (see below).  Classpath format

The classpath argument is a colon or semi-colon delimited
set (depending on platform) of directories, jar files,
or zip files. See the following page for more details:
http://java.sun.com/j2se/1.5.0/docs/tooldocs/solaris/classpath.html

Mustang-style wildcards are also supported:
http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=6268383 Rules
file format

The rules file is a text file, one rule per line. Leading and trailing
whitespace is ignored. There are three types of rules:

```
rule <pattern> <result>
zap <pattern>
keep <pattern>
```

The standard rule (rule) is used to rename classes. All references to
the renamed classes will also be updated. If a class name is matched
by more than one rule, only the first one will apply.

`<pattern>` is a class name with optional wildcards. `**` will match
against any valid class name substring. To match a single package
component (by excluding . from the match), a single `*` may be used
instead.

`<result>` is a class name which can optionally reference the substrings
matched by the wildcards. A numbered reference is available for every
`*` or `**` in the `<pattern>`, starting from left to right: @1, @2, etc. A
special @0 reference contains the entire matched class name.

The zap rule causes any matched class to be removed from the resulting
jar file. All zap rules are processed before renaming rules.

The keep rule marks all matched classes as "roots". If any keep rules
are defined all classes which are not reachable from the roots via
dependency analysis are discarded when writing the output jar. This
is the last step in the process, after renaming and zapping.

