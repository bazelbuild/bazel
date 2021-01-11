package net.starlark.java.eval;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

public class DataStructureConverter {

  private DataStructureConverter() { throw new IllegalStateException("Utility class"); }

  /**
   * Converts a Java value {@code x} to a Starlark one, if x is not already a valid Starlark value.
   * An Integer, Long, or BigInteger is converted to a Starlark int, a double is converted to a
   * Starlark float, a Java List or Map is converted to a Starlark list or dict, respectively, and
   * null becomes {@link #Starlark.NONE}. Any other non-Starlark value causes the function to throw
   * IllegalArgumentException.
   *
   * <p>Elements of Lists and Maps must be valid Starlark values; they are not recursively
   * converted. (This avoids excessive unintended deep copying.)
   *
   * <p>This function is applied to the results of StarlarkMethod-annotated Java methods.
   */
  public static Object fromJava(Object x, @Nullable Mutability mutability) {
    if (x == null) {
      return Starlark.NONE;
    } else if (Starlark.valid(x)) {
      return x;
    } else if (x instanceof Number) {
        return fromJava((Number) x);
    } else if (x instanceof List) {
      return StarlarkList.copyOf(mutability, (List<?>) x);
    } else if (x instanceof Map) {
      return StarlarkDict.copyOf(mutability, (Map<?, ?>) x);
    }
    throw new IllegalArgumentException("cannot expose internal type to Starlark: " + x.getClass());
  }

  public static Object fromJava(Number x) {
    if (x instanceof Integer) {
      return StarlarkInt.of((Integer) x);
    } else if (x instanceof Long) {
      return StarlarkInt.of((Long) x);
    } else if (x instanceof BigInteger) {
      return StarlarkInt.of((BigInteger) x);
    } else if (x instanceof Double) {
      return StarlarkFloat.of((double) x);
    }
      throw new IllegalArgumentException("cannot expose internal type to Starlark: " + x.getClass());
  }
}
