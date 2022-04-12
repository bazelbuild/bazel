## Restoring a stack trace with line numbers {: #with}

Assume for instance ProGuard itself has been obfuscated using the
following extra options:

    -printmapping mapping.txt

    -renamesourcefileattribute MyApplication
    -keepattributes SourceFile,LineNumberTable

Now assume the processed application throws an exception:

    java.io.IOException: Can't read [dummy.jar] (No such file or directory)
        at proguard.y.a(MyApplication:188)
        at proguard.y.a(MyApplication:158)
        at proguard.y.a(MyApplication:136)
        at proguard.y.a(MyApplication:66)
        at proguard.ProGuard.c(MyApplication:218)
        at proguard.ProGuard.a(MyApplication:82)
        at proguard.ProGuard.main(MyApplication:538)
    Caused by: java.io.IOException: No such file or directory
        at proguard.d.q.a(MyApplication:50)
        at proguard.y.a(MyApplication:184)
        ... 6 more

If we have saved the stack trace in a file `stacktrace.txt`, we can use
the following command to recover the stack trace:

    retrace mapping.txt stacktrace.txt

The output will correspond to the original stack trace:

    java.io.IOException: Can't read [dummy.jar] (No such file or directory)
        at proguard.InputReader.readInput(InputReader.java:188)
        at proguard.InputReader.readInput(InputReader.java:158)
        at proguard.InputReader.readInput(InputReader.java:136)
        at proguard.InputReader.execute(InputReader.java:66)
        at proguard.ProGuard.readInput(ProGuard.java:218)
        at proguard.ProGuard.execute(ProGuard.java:82)
        at proguard.ProGuard.main(ProGuard.java:538)
    Caused by: java.io.IOException: No such file or directory
        at proguard.io.DirectoryPump.pumpDataEntries(DirectoryPump.java:50)
        at proguard.InputReader.readInput(InputReader.java:184)
        ... 6 more

## Restoring a stack trace with line numbers (verbose) {: #withverbose}

In the previous example, we could also use the verbose flag:

    retrace -verbose mapping.txt stacktrace.txt

The output will then look as follows:

    java.io.IOException: Can't read [dummy.jar] (No such file or directory)
        at proguard.InputReader.void readInput(java.lang.String,proguard.ClassPathEntry,proguard.io.DataEntryReader)(InputReader.java:188)
        at proguard.InputReader.void readInput(java.lang.String,proguard.ClassPath,int,int,proguard.io.DataEntryReader)(InputReader.java:158)
        at proguard.InputReader.void readInput(java.lang.String,proguard.ClassPath,proguard.io.DataEntryReader)(InputReader.java:136)
        at proguard.InputReader.void execute(proguard.classfile.ClassPool,proguard.classfile.ClassPool)(InputReader.java:66)
        at proguard.ProGuard.void readInput()(ProGuard.java:218)
        at proguard.ProGuard.void execute()(ProGuard.java:82)
        at proguard.ProGuard.void main(java.lang.String[])(ProGuard.java:538)
    Caused by: java.io.IOException: No such file or directory
        at proguard.io.DirectoryPump.void pumpDataEntries(proguard.io.DataEntryReader)(DirectoryPump.java:50)
        at proguard.InputReader.void readInput(java.lang.String,proguard.ClassPathEntry,proguard.io.DataEntryReader)(InputReader.java:184)
        ... 6 more

## Restoring a stack trace without line numbers {: #without}

Assume for instance ProGuard itself has been obfuscated using the
following extra options, this time without preserving the line number
tables:

    -printmapping mapping.txt

A stack trace `stacktrace.txt` will then lack line number information,
showing "Unknown source" instead:

    java.io.IOException: Can't read [dummy.jar] (No such file or directory)
        at proguard.y.a(Unknown Source)
        at proguard.y.a(Unknown Source)
        at proguard.y.a(Unknown Source)
        at proguard.y.a(Unknown Source)
        at proguard.ProGuard.c(Unknown Source)
        at proguard.ProGuard.a(Unknown Source)
        at proguard.ProGuard.main(Unknown Source)
    Caused by: java.io.IOException: No such file or directory
        at proguard.d.q.a(Unknown Source)
        ... 7 more

We can still use the same command to recover the stack trace:

    java -jar retrace.jar mapping.txt stacktrace.txt

The output will now list all alternative original method names for each
ambiguous obfuscated method name:

    java.io.IOException: Can't read [dummy.jar] (No such file or directory)
        at proguard.InputReader.execute(InputReader.java)
                                readInput(InputReader.java)
        at proguard.InputReader.execute(InputReader.java)
                                readInput(InputReader.java)
        at proguard.InputReader.execute(InputReader.java)
                                readInput(InputReader.java)
        at proguard.InputReader.execute(InputReader.java)
                                readInput(InputReader.java)
        at proguard.ProGuard.readInput(ProGuard.java)
        at proguard.ProGuard.execute(ProGuard.java)
                             optimize(ProGuard.java)
                             createPrintStream(ProGuard.java)
                             closePrintStream(ProGuard.java)
                             fileName(ProGuard.java)
        at proguard.ProGuard.main(ProGuard.java)
    Caused by: java.io.IOException: No such file or directory
        at proguard.io.DirectoryPump.pumpDataEntries(DirectoryPump.java)
                                     readFiles(DirectoryPump.java)

For instance, ReTrace can't tell if the method `a` corresponds to
`execute` or to `readInput`, so it lists both. You need to figure it out
based on your knowledge of the application. Having line numbers and
unambiguous names clearly is a lot easier, so you should consider
[preserving the line numbers](../examples.md#stacktrace) when you
obfuscate your application.
