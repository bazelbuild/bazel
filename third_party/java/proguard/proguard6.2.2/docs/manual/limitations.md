When using ProGuard, you should be aware of a few technical issues, all of
which are easily avoided or resolved:

- For best results, ProGuard's optimization algorithms assume that the
  processed code never **intentionally throws NullPointerExceptions** or
  ArrayIndexOutOfBoundsExceptions, or even OutOfMemoryErrors or
  StackOverflowErrors, in order to achieve something useful. For instance, it
  may remove a method call `myObject.myMethod()` if that call wouldn't have
  any effect. It ignores the possibility that `myObject` might be null,
  causing a NullPointerException. In some way this is a good thing: optimized
  code may throw fewer exceptions. Should this entire assumption be false,
  you'll have to switch off optimization using the `-dontoptimize` option.

- ProGuard's optimization algorithms currently also assume that the processed
  code never creates **busy-waiting loops** without at least testing on a
  volatile field. Again, it may remove such loops. Should this assumption be
  false, you'll have to switch off optimization using the `-dontoptimize`
  option.

- When obfuscating, ProGuard writes out class files named "`a.class`",
  "`b.class`", etc. If a package contains a large number of classes, ProGuard
  may also write out **"`aux.class`"**. Inconveniently, Windows refuses to
  create files with this reserved name (among a few other names). It's
  generally better to write the output to a jar, in order to avoid such
  problems. Alternatively, you can use the Uniform Naming Convention (UNC) for
  your Windows paths, which doesn't suffer from the constraint.
