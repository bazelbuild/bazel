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

import com.google.common.base.Strings;
import com.google.common.collect.Ordering;
import java.util.IllegalFormatException;

/** Internal declarations used by the evaluator. */
final class EvalUtils {

  private EvalUtils() {}

  /**
   * The exception that STARLARK_COMPARATOR might throw. This is an unchecked exception because
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
  static final Ordering<Object> STARLARK_COMPARATOR =
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
   * Reports whether a legal Starlark value is considered hashable to Starlark, and thus suitable as
   * a key in a dict.
   */
  static boolean isHashable(Object o) {
    // Bazel makes widespread assumptions that all Starlark values can be hashed
    // by Java code, so we cannot implement isHashable by having
    // StarlarkValue.hashCode throw an unchecked exception, which would be more
    // efficient. Instead, before inserting a value in a dict, we must first ask
    // it whether it isHashable, and then call its hashCode method only if so.
    // For structs and tuples, this unfortunately visits the object graph twice.
    //
    // One subtlety: the struct.isHashable recursively asks whether its
    // elements are immutable, not hashable. Consequently, even though a list
    // may not be used as a dict key (even if frozen), a struct containing
    // a list is hashable. TODO(adonovan): fix this inconsistency.
    // Requires an incompatible change flag.
    if (o instanceof StarlarkValue) {
      return ((StarlarkValue) o).isHashable();
    }
    return Starlark.isImmutable(o);
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
          int z = xi << yi; // only uses low 5 bits of yi
          if ((z >> yi) != xi || yi >= 32) {
            throw Starlark.errorf("integer overflow in left shift");
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
          if (xi == Integer.MIN_VALUE && yi == -1) { // HD 2-13
            throw Starlark.errorf("integer overflow in division");
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
            throw new EvalException(ex);
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
        if (y instanceof StarlarkIndexable) {
          return ((StarlarkIndexable) y).containsKey(semantics, x);
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
      return STARLARK_COMPARATOR.compare(x, y);
    } catch (ComparisonException e) {
      throw new EvalException(e);
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
    if (object instanceof StarlarkIndexable) {
      Object result = ((StarlarkIndexable) object).getIndex(semantics, key);
      // TODO(bazel-team): We shouldn't have this fromJava call here. If it's needed at all,
      // it should go in the implementations of StarlarkIndexable#getIndex that produce non-Starlark
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
}
