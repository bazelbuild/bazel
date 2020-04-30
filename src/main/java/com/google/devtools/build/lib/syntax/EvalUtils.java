// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.starlark.spelling.SpellChecker;
import java.util.IllegalFormatException;

/** Utilities used by the evaluator. */
// TODO(adonovan): move all fundamental values and operators of the language to Starlark
// class---equal, compare, getattr, index, slice, parse, exec, eval, and so on---and make this
// private.
public final class EvalUtils {

  private EvalUtils() {}

  /**
   * The exception that SKYLARK_COMPARATOR might throw. This is an unchecked exception because
   * Comparator doesn't let us declare exceptions. It should normally be caught and wrapped in an
   * EvalException.
   */
  static class ComparisonException extends RuntimeException {
    ComparisonException(String msg) {
      super(msg);
    }
  }

  /**
   * Compare two Starlark values.
   *
   * <p>It may throw an unchecked exception ComparisonException that should be wrapped in an
   * EvalException.
   */
  // TODO(adonovan): consider what API to expose around comparison and ordering. Java's three-valued
  // comparator cannot properly handle weakly or partially ordered values such as IEEE754 floats.
  static final Ordering<Object> SKYLARK_COMPARATOR =
      new Ordering<Object>() {
        private int compareLists(Sequence<?> o1, Sequence<?> o2) {
          if (o1 instanceof RangeList || o2 instanceof RangeList) {
            throw new ComparisonException("Cannot compare range objects");
          }

          for (int i = 0; i < Math.min(o1.size(), o2.size()); i++) {
            int cmp = compare(o1.get(i), o2.get(i));
            if (cmp != 0) {
              return cmp;
            }
          }
          return Integer.compare(o1.size(), o2.size());
        }

        @Override
        @SuppressWarnings("unchecked")
        public int compare(Object o1, Object o2) {

          // optimize the most common cases

          if (o1 instanceof String && o2 instanceof String) {
            return ((String) o1).compareTo((String) o2);
          }
          if (o1 instanceof Integer && o2 instanceof Integer) {
            return Integer.compare((Integer) o1, (Integer) o2);
          }

          o1 = Starlark.fromJava(o1, null);
          o2 = Starlark.fromJava(o2, null);

          if (o1 instanceof Sequence
              && o2 instanceof Sequence
              && o1 instanceof Tuple == o2 instanceof Tuple) {
            return compareLists((Sequence) o1, (Sequence) o2);
          }

          if (o1 instanceof ClassObject) {
            throw new ComparisonException("Cannot compare structs");
          }
          try {
            return ((Comparable<Object>) o1).compareTo(o2);
          } catch (ClassCastException e) {
            throw new ComparisonException(
                "Cannot compare " + Starlark.type(o1) + " with " + Starlark.type(o2));
          }
        }
      };

  /** Throws EvalException if x is not hashable. */
  static void checkHashable(Object x) throws EvalException {
    if (!isHashable(x)) {
      // This results in confusing errors such as "unhashable type: tuple".
      // TODO(adonovan): ideally the error message would explain which
      // element of, say, a tuple is unhashable. The only practical way
      // to implement this is by implementing isHashable as a call to
      // Object.hashCode within a try/catch, and requiring all
      // unhashable Starlark values to throw a particular unchecked exception
      // with a helpful error message.
      throw Starlark.errorf("unhashable type: '%s'", Starlark.type(x));
    }
  }

  /**
   * Is this object known or assumed to be recursively hashable by Starlark?
   *
   * @param o an Object
   * @return true if the object is known to be a hashable value.
   */
  // TODO(adonovan): obviate isHashable query by "try hash, catch StarlarkUnhashable",
  // an unchecked exception. It is inefficient and potentially inconsistent to ask before doing.
  static boolean isHashable(Object o) {
    if (o instanceof StarlarkValue) {
      return ((StarlarkValue) o).isHashable();
    }
    return isImmutable(o.getClass());
  }

  /**
   * Is this object known or assumed to be recursively immutable by Starlark?
   *
   * @param o an Object
   * @return true if the object is known to be an immutable value.
   */
  // TODO(adonovan): eliminate the concept of querying for immutability. It is currently used for
  // only one purpose, the precondition for adding an element to a Depset, but Depsets should check
  // hashability, like Dicts. (Similarly, querying for hashability should go: just attempt to hash a
  // value, and be prepared for it to fail.) In practice, a value may be immutable, either
  // inherently (e.g. string) or because it has become frozen, but we don't need to query for it.
  // Just attempt a mutation and be preared for it to fail.
  // It is inefficient and potentially inconsistent to ask before doing.
  public static boolean isImmutable(Object o) {
    if (o instanceof StarlarkValue) {
      return ((StarlarkValue) o).isImmutable();
    }
    return isImmutable(o.getClass());
  }

  /**
   * Is this class known to be *recursively* immutable by Starlark? For instance, class Tuple is not
   * it, because it can contain mutable values.
   *
   * @param c a Class
   * @return true if the class is known to represent only recursively immutable values.
   */
  // NB: This is used as the basis for accepting objects in Depset-s,
  // as well as for accepting objects as keys for Starlark dict-s.
  private static boolean isImmutable(Class<?> c) {
    return c.isAnnotationPresent(Immutable.class) // TODO(bazel-team): beware of containers!
        || c.equals(String.class)
        || c.equals(Integer.class)
        || c.equals(Boolean.class);
  }

  static void addIterator(Object x) {
    if (x instanceof Mutability.Freezable) {
      ((Mutability.Freezable) x).updateIteratorCount(+1);
    }
  }

  static void removeIterator(Object x) {
    if (x instanceof Mutability.Freezable) {
      ((Mutability.Freezable) x).updateIteratorCount(-1);
    }
  }

  // The following functions for indexing and slicing match the behavior of Python.

  /**
   * Resolves a positive or negative index to an index in the range [0, length), or throws
   * EvalException if it is out of range. If the index is negative, it counts backward from length.
   */
  static int getSequenceIndex(int index, int length) throws EvalException {
    int actualIndex = index;
    if (actualIndex < 0) {
      actualIndex += length;
    }
    if (actualIndex < 0 || actualIndex >= length) {
      throw Starlark.errorf(
          "index out of range (index is %d, but sequence has %d elements)", index, length);
    }
    return actualIndex;
  }

  /**
   * Returns the effective index denoted by a user-supplied integer. First, if the integer is
   * negative, the length of the sequence is added to it, so an index of -1 represents the last
   * element of the sequence. Then, the integer is "clamped" into the inclusive interval [0,
   * length].
   */
  static int toIndex(int index, int length) {
    if (index < 0) {
      index += length;
    }

    if (index < 0) {
      return 0;
    } else if (index > length) {
      return length;
    } else {
      return index;
    }
  }

  /** @return true if x is Java null or Starlark None */
  public static boolean isNullOrNone(Object x) {
    return x == null || x == Starlark.NONE;
  }

  /** Returns the named field or method of value {@code x}, or null if not found. */
  // TODO(adonovan): publish this method as Starlark.getattr(Semantics, Mutability, Object, String).
  static Object getAttr(StarlarkThread thread, Object x, String name)
      throws EvalException, InterruptedException {
    StarlarkSemantics semantics = thread.getSemantics();
    Mutability mu = thread.mutability();

    // @SkylarkCallable-annotated field or method?
    MethodDescriptor method = CallUtils.getMethod(semantics, x.getClass(), name);
    if (method != null) {
      if (method.isStructField()) {
        return method.callField(x, semantics, mu);
      } else {
        return new BuiltinCallable(x, name, method);
      }
    }

    // user-defined field?
    if (x instanceof ClassObject) {
      // TODO(adonovan): merge SkylarkClassObject and ClassObject, using a default implementation.
      Object field =
          x instanceof SkylarkClassObject
              ? ((SkylarkClassObject) x).getValue(semantics, name)
              : ((ClassObject) x).getValue(name);
      if (field != null) {
        return Starlark.checkValid(field);
      }
    }

    return null;
  }

  static EvalException getMissingAttrException(
      Object object, String name, StarlarkSemantics semantics) {
    String suffix = "";
    if (object instanceof ClassObject) {
      String customErrorMessage = ((ClassObject) object).getErrorMessageForUnknownField(name);
      if (customErrorMessage != null) {
        return Starlark.errorf("%s", customErrorMessage);
      }
      suffix = SpellChecker.didYouMean(name, ((ClassObject) object).getFieldNames());
    } else {
      suffix = SpellChecker.didYouMean(name, CallUtils.getFieldNames(semantics, object));
    }
    return Starlark.errorf(
        "'%s' value has no field or method '%s'%s", Starlark.type(object), name, suffix);
  }

  /** Evaluates an eager binary operation, {@code x op y}. (Excludes AND and OR.) */
  static Object binaryOp(
      TokenKind op, Object x, Object y, StarlarkSemantics semantics, Mutability mu)
      throws EvalException {
    switch (op) {
      case PLUS:
        if (x instanceof Integer) {
          if (y instanceof Integer) {
            // int + int
            int xi = (Integer) x;
            int yi = (Integer) y;
            int z = xi + yi;
            // Overflow Detection, §2-13 Hacker's Delight:
            // "operands have the same sign and the sum
            // has a sign opposite to that of the operands."
            if (((xi ^ z) & (yi ^ z)) < 0) {
              throw Starlark.errorf("integer overflow in addition");
            }
            return z;
          }

        } else if (x instanceof String) {
          if (y instanceof String) {
            // string + string
            return (String) x + (String) y;
          }

        } else if (x instanceof Tuple) {
          if (y instanceof Tuple) {
            // tuple + tuple
            return Tuple.concat((Tuple<?>) x, (Tuple<?>) y);
          }

        } else if (x instanceof StarlarkList) {
          if (y instanceof StarlarkList) {
            // list + list
            return StarlarkList.concat((StarlarkList<?>) x, (StarlarkList<?>) y, mu);
          }

        }
        break;

      case PIPE:
        if (x instanceof Integer) {
          if (y instanceof Integer) {
            // int | int
            return ((Integer) x) | (Integer) y;
          }
        }
        break;

      case AMPERSAND:
        if (x instanceof Integer && y instanceof Integer) {
          // int & int
          return (Integer) x & (Integer) y;
        }
        break;

      case CARET:
        if (x instanceof Integer && y instanceof Integer) {
          // int ^ int
          return (Integer) x ^ (Integer) y;
        }
        break;

      case GREATER_GREATER:
        if (x instanceof Integer && y instanceof Integer) {
          // int >> int
          int xi = (Integer) x;
          int yi = (Integer) y;
          if (yi < 0) {
            throw Starlark.errorf("negative shift count: %d", yi);
          } else if (yi >= Integer.SIZE) {
            return xi < 0 ? -1 : 0;
          }
          return xi >> yi;
        }
        break;

      case LESS_LESS:
        if (x instanceof Integer && y instanceof Integer) {
          // int << int
          int xi = (Integer) x;
          int yi = (Integer) y;
          if (yi < 0) {
            throw Starlark.errorf("negative shift count: %d", yi);
          }
          int z = xi << yi;
          if (z >> yi != xi) {
            throw Starlark.errorf("integer overflow");
          }
          return z;
        }
        break;

      case MINUS:
        if (x instanceof Integer && y instanceof Integer) {
          // int - int
          int xi = (Integer) x;
          int yi = (Integer) y;
          int z = xi - yi;
          if (((xi ^ yi) & (xi ^ z)) < 0) {
            throw Starlark.errorf("integer overflow in subtraction");
          }
          return z;
        }
        break;

      case STAR:
        if (x instanceof Integer) {
          int xi = (Integer) x;
          if (y instanceof Integer) {
            // int * int
            long z = (long) xi * (long) (Integer) y;
            if ((int) z != z) {
              throw Starlark.errorf("integer overflow in multiplication");
            }
            return (int) z;
          } else if (y instanceof String) {
            // int * string
            return repeatString((String) y, xi);
          } else if (y instanceof Tuple) {
            //  int * tuple
            return ((Tuple<?>) y).repeat(xi);
          } else if (y instanceof StarlarkList) {
            // int * list
            return ((StarlarkList<?>) y).repeat(xi, mu);
          }

        } else if (x instanceof String) {
          if (y instanceof Integer) {
            // string * int
            return repeatString((String) x, (Integer) y);
          }

        } else if (x instanceof Tuple) {
          if (y instanceof Integer) {
            // tuple * int
            return ((Tuple<?>) x).repeat((Integer) y);
          }

        } else if (x instanceof StarlarkList) {
          if (y instanceof Integer) {
            // list * int
            return ((StarlarkList<?>) x).repeat((Integer) y, mu);
          }
        }
        break;

      case SLASH:
        throw Starlark.errorf("The `/` operator is not allowed. For integer division, use `//`.");

      case SLASH_SLASH:
        if (x instanceof Integer && y instanceof Integer) {
          // int // int
          int xi = (Integer) x;
          int yi = (Integer) y;
          if (yi == 0) {
            throw Starlark.errorf("integer division by zero");
          }
          // http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
          int quo = xi / yi;
          int rem = xi % yi;
          if ((xi < 0) != (yi < 0) && rem != 0) {
            quo--;
          }
          return quo;
        }
        break;

      case PERCENT:
        if (x instanceof Integer) {
          if (y instanceof Integer) {
            // int % int
            int xi = (Integer) x;
            int yi = (Integer) y;
            if (yi == 0) {
              throw Starlark.errorf("integer modulo by zero");
            }
            // In Starlark, the sign of the result is the sign of the divisor.
            int z = xi % yi;
            if ((xi < 0) != (yi < 0) && z != 0) {
              z += yi;
            }
            return z;
          }

        } else if (x instanceof String) {
          // string % any
          String xs = (String) x;
          try {
            if (y instanceof Tuple) {
              return Starlark.formatWithList(xs, (Tuple) y);
            } else {
              return Starlark.format(xs, y);
            }
          } catch (IllegalFormatException ex) {
            throw new EvalException(null, ex.getMessage());
          }
        }
        break;

      case EQUALS_EQUALS:
        return x.equals(y);

      case NOT_EQUALS:
        return !x.equals(y);

      case LESS:
        return compare(x, y) < 0;

      case LESS_EQUALS:
        return compare(x, y) <= 0;

      case GREATER:
        return compare(x, y) > 0;

      case GREATER_EQUALS:
        return compare(x, y) >= 0;

      case IN:
        if (y instanceof SkylarkQueryable) {
          return ((SkylarkQueryable) y).containsKey(semantics, x);
        } else if (y instanceof String) {
          if (!(x instanceof String)) {
            throw Starlark.errorf(
                "'in <string>' requires string as left operand, not '%s'", Starlark.type(x));
          }
          return ((String) y).contains((String) x);
        }
        break;

      case NOT_IN:
        Object z = binaryOp(TokenKind.IN, x, y, semantics, mu);
        if (z != null) {
          return !Starlark.truth(z);
        }
        break;

      default:
        throw new AssertionError("not a binary operator: " + op);
    }

    // custom binary operator?
    if (x instanceof HasBinary) {
      Object z = ((HasBinary) x).binaryOp(op, y, true);
      if (z != null) {
        return z;
      }
    }
    if (y instanceof HasBinary) {
      Object z = ((HasBinary) y).binaryOp(op, x, false);
      if (z != null) {
        return z;
      }
    }

    throw Starlark.errorf(
        "unsupported binary operation: %s %s %s", Starlark.type(x), op, Starlark.type(y));
  }

  /** Implements comparison operators. */
  private static int compare(Object x, Object y) throws EvalException {
    try {
      return SKYLARK_COMPARATOR.compare(x, y);
    } catch (ComparisonException e) {
      throw new EvalException(null, e.getMessage());
    }
  }

  private static String repeatString(String s, int n) {
    return n <= 0 ? "" : Strings.repeat(s, n);
  }

  /** Evaluates a unary operation. */
  static Object unaryOp(TokenKind op, Object x) throws EvalException {
    switch (op) {
      case NOT:
        return !Starlark.truth(x);

      case MINUS:
        if (x instanceof Integer) {
          int xi = (Integer) x;
          if (xi == Integer.MIN_VALUE) {
            throw Starlark.errorf("integer overflow in negation");
          }
          return -xi;
        }
        break;

      case PLUS:
        if (x instanceof Integer) {
          return x;
        }
        break;

      case TILDE:
        if (x instanceof Integer) {
          return ~((Integer) x);
        }
        break;

      default:
        /* fall through */
    }
    throw Starlark.errorf("unsupported unary operation: %s%s", op, Starlark.type(x));
  }

  /**
   * Returns the element of sequence or mapping {@code object} indexed by {@code key}.
   *
   * @throws EvalException if {@code object} is not a sequence or mapping.
   */
  static Object index(Mutability mu, StarlarkSemantics semantics, Object object, Object key)
      throws EvalException {
    if (object instanceof SkylarkIndexable) {
      Object result = ((SkylarkIndexable) object).getIndex(semantics, key);
      // TODO(bazel-team): We shouldn't have this fromJava call here. If it's needed at all,
      // it should go in the implementations of SkylarkIndexable#getIndex that produce non-Starlark
      // values.
      return result == null ? null : Starlark.fromJava(result, mu);
    } else if (object instanceof String) {
      String string = (String) object;
      int index = Starlark.toInt(key, "string index");
      index = getSequenceIndex(index, string.length());
      return string.substring(index, index + 1);
    } else {
      throw Starlark.errorf(
          "type '%s' has no operator [](%s)", Starlark.type(object), Starlark.type(key));
    }
  }

  /**
   * Updates an object as if by the Starlark statement {@code object[key] = value}.
   *
   * @throws EvalException if the object is not a list or dict.
   */
  static void setIndex(Object object, Object key, Object value) throws EvalException {
    if (object instanceof Dict) {
      @SuppressWarnings("unchecked")
      Dict<Object, Object> dict = (Dict<Object, Object>) object;
      dict.put(key, value, (Location) null);

    } else if (object instanceof StarlarkList) {
      @SuppressWarnings("unchecked")
      StarlarkList<Object> list = (StarlarkList<Object>) object;
      int index = Starlark.toInt(key, "list index");
      index = EvalUtils.getSequenceIndex(index, list.size());
      list.set(index, value, (Location) null);

    } else {
      throw Starlark.errorf(
          "can only assign an element in a dictionary or a list, not in a '%s'",
          Starlark.type(object));
    }
  }

  /**
   * Parses the input as a file, resolves it in the module environment using the specified options
   * and returns the syntax tree. Scan/parse/resolve errors are recorded in the StarlarkFile. It is
   * the caller's responsibility to inspect them.
   */
  public static StarlarkFile parseAndValidate(
      ParserInput input, FileOptions options, Module module) {
    StarlarkFile file = StarlarkFile.parse(input, options);
    Resolver.resolveFile(file, module);
    return file;
  }

  /**
   * Parses the input as a file, resolves it in the module environment using the specified options
   * and executes it. It returns None, unless the final statement is an expression, in which case it
   * returns the expression's value.
   */
  public static Object exec(
      ParserInput input, FileOptions options, Module module, StarlarkThread thread)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    StarlarkFile file = parseAndValidate(input, options, module);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return exec(file, module, thread);
  }

  /** Executes a parsed, resolved Starlark file in the given StarlarkThread. */
  public static Object exec(StarlarkFile file, Module module, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(
        file.resolved,
        "cannot evaluate unresolved syntax (use other exec method, or parseAndValidate)");

    Tuple<Object> defaultValues = Tuple.empty();
    StarlarkFunction toplevel = new StarlarkFunction(file.resolved, defaultValues, module);

    return Starlark.fastcall(thread, toplevel, NOARGS, NOARGS);
  }

  /**
   * Parses the input as an expression, resolves it in the module environment using the specified
   * options, and evaluates it.
   */
  public static Object eval(
      ParserInput input, FileOptions options, Module module, StarlarkThread thread)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    Expression expr = Expression.parse(input, options);

    Resolver.Function rfn = Resolver.resolveExpr(expr, module, options);

    // Turn expression into a no-arg StarlarkFunction.
    Tuple<Object> defaultValues = Tuple.empty();
    StarlarkFunction exprFunc = new StarlarkFunction(rfn, defaultValues, module);

    return Starlark.fastcall(thread, exprFunc, NOARGS, NOARGS);
  }

  private static final Object[] NOARGS = {};
}
