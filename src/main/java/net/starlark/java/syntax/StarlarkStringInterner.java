package net.starlark.java.syntax;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;

/**
 * Common place to intern strings in Starlark interpreter.
 *
 * <p>Interned strings are much faster to lookup, which is important, for example, when evaluating
 * expression {@code foo.bar}.
 */
public class StarlarkStringInterner {
  private StarlarkStringInterner() {}

  private static final Interner<String> INTERNER = Interners.newWeakInterner();

  /** Weak intern the string. */
  public static String intern(String string) {
    return INTERNER.intern(string);
  }
}
