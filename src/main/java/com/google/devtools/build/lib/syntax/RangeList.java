// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import java.util.AbstractList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * A sequence returned by the {@code range} function invocation.
 *
 * <p>Instead of eagerly allocating an array with all elements of the sequence, this class uses
 * simple math to compute a value at each index. This is particularly useful when range is huge or
 * only a few elements from it are used.
 *
 * <p>Eventually {@code range} function should produce an instance of the {@code range} type as is
 * the case in Python 3, but for now to preserve backwards compatibility with Python 2, {@code list}
 * is returned.
 */
@SkylarkModule(
    name = "range",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A language built-in type to support ranges. Example of range literal:<br>"
            + "<pre class=language-python>x = range(1, 10, 3)</pre>"
            + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
            + "<pre class=language-python>e = x[1]   # e == 2</pre>"
            + "Ranges do not support the <code>+</code> operator for concatenation."
            + "Similar to strings, ranges support slice operations:"
            + "<pre class=language-python>range(10)[1:3]   # range(1, 3)\n"
            + "range(10)[::2]  # range(0, 10, 2)\n"
            + "range(10)[3:0:-1]  # range(3, 0, -1)</pre>"
            + "Ranges are immutable, as in Python 3.")
public final class RangeList extends SkylarkList<Integer> {

  private final int step;
  private final int start;

  private static int computeItem(int start, int step, int index) {
    return start + step * index;
  }

  /** Provides access to range elements based on their index. */
  private static class RangeListView extends AbstractList<Integer> {

    /** Iterator for increasing/decreasing sequences. */
    private static class RangeListIterator extends UnmodifiableIterator<Integer> {
      private final int stop;
      private final int step;

      private int cursor;

      private RangeListIterator(int start, int stop, int step) {
        this.cursor = start;
        this.stop = stop;
        this.step = step;
      }

      @Override
      public boolean hasNext() {
        return (step > 0) ? cursor < stop : cursor > stop;
      }

      @Override
      public Integer next() {
        if (!hasNext()) {
          throw new NoSuchElementException();
        }
        int current = cursor;
        cursor += step;
        return current;
      }
    }

    /**
     * @return The size of the range specified by {@code start}, {@code stop} and {@code step}.
     *     Python version:
     *     https://github.com/python/cpython/blob/09bb918a61031377d720f1a0fa1fe53c962791b6/Objects/rangeobject.c#L144
     */
    private static int computeSize(int start, int stop, int step) {
      // low and high represent bounds of the interval with only one of the sides being open.
      int low;
      int high;
      if (step > 0) {
        low = start;
        high = stop;
      } else {
        low = stop;
        high = start;
        step = -step;
      }
      if (low >= high) {
        return 0;
      }

      int diff = high - low - 1;
      return diff / step + 1;
    }

    private final int start;
    private final int stop;
    private final int step;
    private final int size;

    private RangeListView(int start, int stop, int step) {
      this.start = start;
      this.stop = stop;
      this.step = step;
      this.size = computeSize(start, stop, step);
    }

    @Override
    public Integer get(int index) {
      if (index < 0 || index >= size()) {
        throw new ArrayIndexOutOfBoundsException(index);
      }
      return computeItem(start, step, index);
    }

    @Override
    public int size() {
      return size;
    }

    /**
     * Returns an iterator optimized for traversing range elements, since it's the most frequent
     * operation for which ranges are used.
     */
    @Override
    public Iterator<Integer> iterator() {
      return new RangeListIterator(start, stop, step);
    }

    /** @return the start of the range. */
    public int getStart() {
      return start;
    }

    /** @return the stop element (next after the last one) of the range. */
    public int getStop() {
      return stop;
    }

    /** @return the step between each element of the range. */
    public int getStep() {
      return step;
    }
  }

  private final RangeListView contents;

  private RangeList(int start, int stop, int step) {
    this.step = step;
    this.start = start;
    this.contents = new RangeListView(start, stop, step);
  }

  @Override
  public boolean isTuple() {
    return false;
  }

  @Override
  public ImmutableList<Integer> getImmutableList() {
    return ImmutableList.copyOf(contents);
  }

  @Override
  public SkylarkList<Integer> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException {
    Slice slice = Slice.from(size(), start, end, step, loc);
    int substep = slice.step * this.step;
    int substart = computeItem(this.start, this.step, slice.start);
    int substop = computeItem(this.start, this.step, slice.stop);
    return RangeList.of(substart, substop, substep);
  }

  @Override
  public SkylarkList<Integer> repeat(int times, Mutability mutability) {
    throw new UnsupportedOperationException("Ranges do not support repetition.");
  }

  @Override
  protected List<Integer> getContentsUnsafe() {
    return contents;
  }

  @Override
  public Mutability mutability() {
    return Mutability.IMMUTABLE;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    if (contents.getStep() == 1) {
      printer.format("range(%d, %d)", contents.getStart(), contents.getStop());
    } else {
      printer.format(
          "range(%d, %d, %d)", contents.getStart(), contents.getStop(), contents.getStep());
    }
  }

  /**
   * @return A half-opened range defined by its starting value (inclusive), stop value (exclusive)
   *     and a step from previous value to the next one.
   */
  public static RangeList of(int start, int stop, int step) {
    Preconditions.checkArgument(step != 0);
    return new RangeList(start, stop, step);
  }

  /**
   * Represents a slice produced by applying {@code [start:end:step]} to a {@code range}.
   *
   * <p>{@code start} and {@code stop} define a half-open interval
   *
   * <pre>[start, stop)</pre>
   */
  private static class Slice {

    private final int start;
    private final int stop;
    private final int step;

    private Slice(int start, int stop, int step) {
      this.start = start;
      this.stop = stop;
      this.step = step;
    }

    /**
     * Computes slice indices for the requested range slice.
     *
     * <p>The implementation is based on CPython
     * https://github.com/python/cpython/blob/09bb918a61031377d720f1a0fa1fe53c962791b6/Objects/sliceobject.c#L366-L509
     */
    public static Slice from(
        int length, Object startObj, Object endObj, Object stepObj, Location loc)
        throws EvalException {
      int start;
      int stop;
      int step;

      if (stepObj == Runtime.NONE) {
        step = 1;
      } else if (stepObj instanceof Integer) {
        step = (Integer) stepObj;
      } else {
        throw new EvalException(
            loc, String.format("slice step must be an integer, not '%s'", stepObj));
      }
      if (step == 0) {
        throw new EvalException(loc, "slice step cannot be zero");
      }

      int upper; // upper bound for stop (exclusive)
      int lower; // lower bound for start (inclusive)
      if (step < 0) {
        lower = -1;
        upper = length - 1;
      } else {
        lower = 0;
        upper = length;
      }

      if (startObj == Runtime.NONE) {
        start = step < 0 ? upper : lower;
      } else if (startObj instanceof Integer) {
        start = (Integer) startObj;
        if (start < 0) {
          start += length;
          start = Math.max(start, lower);
        } else {
          start = Math.min(start, upper);
        }
      } else {
        throw new EvalException(
            loc, String.format("slice start must be an integer, not '%s'", startObj));
      }
      if (endObj == Runtime.NONE) {
        stop = step < 0 ? lower : upper;
      } else if (endObj instanceof Integer) {
        stop = (Integer) endObj;
        if (stop < 0) {
          stop += length;
          stop = Math.max(stop, lower);
        } else {
          stop = Math.min(stop, upper);
        }
      } else {
        throw new EvalException(
            loc, String.format("slice end must be an integer, not '%s'", endObj));
      }
      return new Slice(start, stop, step);
    }
  }
}
