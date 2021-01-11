package net.starlark.java.eval;

public class IndexingSlicingUtils {

  private IndexingSlicingUtils() { throw new IllegalStateException("Utility class"); }


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
      throw new EvalException(
          "index out of range (index is) " + index + ", but sequence has " + length + "elements)");
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
}
