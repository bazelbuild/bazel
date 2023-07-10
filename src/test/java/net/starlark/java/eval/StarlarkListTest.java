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
package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.testing.GcFinalization;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.lang.ref.WeakReference;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of StarlarkList's Java API. */
// TODO(adonovan): duplicate/share these tests for Tuple where applicable.
@RunWith(JUnit4.class)
public final class StarlarkListTest {

  @Test
  public void listAfterRemoveHasExpectedEqualsAndHashCode() throws Exception {
    StarlarkList<String> l = StarlarkList.of(Mutability.create(), "1", "2", "3");
    l.removeElement("3");
    assertThat(l.hashCode()).isEqualTo(StarlarkList.immutableOf("1", "2").hashCode());
  }

  @Test
  public void testListAddWithIndex() throws Exception {
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> list = StarlarkList.newList(mutability);
    list.addElement("a");
    list.addElement("b");
    list.addElement("c");
    list.addElementAt(0, "d");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"b\", \"c\"]");
    list.addElementAt(2, "e");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"c\"]");
    list.addElementAt(4, "f");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\"]");
    list.addElementAt(6, "g");
    assertThat(list.toString()).isEqualTo("[\"d\", \"a\", \"e\", \"b\", \"f\", \"c\", \"g\"]");
    assertThrows(ArrayIndexOutOfBoundsException.class, () -> list.addElementAt(8, "h"));
  }

  @Test
  public void testMutatorsCheckMutability() {
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> list =
        StarlarkList.copyOf(
            mutability, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3)));
    mutability.freeze();
    checkImmutable(list);
  }

  @Test
  public void testCannotMutateAfterShallowFreeze() {
    Mutability mutability = Mutability.createAllowingShallowFreeze("test");
    StarlarkList<Object> list =
        StarlarkList.copyOf(
            mutability, ImmutableList.of(StarlarkInt.of(1), StarlarkInt.of(2), StarlarkInt.of(3)));
    list.unsafeShallowFreeze();

    EvalException e = assertThrows(EvalException.class, () -> list.addElement(StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
  }

  @Test
  public void testCopyOfTakesCopy() throws EvalException {
    ArrayList<String> copyFrom = Lists.newArrayList("hi");
    Mutability mutability = Mutability.create("test");
    StarlarkList<String> mutableList = StarlarkList.copyOf(mutability, copyFrom);
    copyFrom.add("added1");
    mutableList.addElement("added2");

    assertThat(copyFrom).containsExactly("hi", "added1").inOrder();
    assertThat((List<String>) mutableList).containsExactly("hi", "added2").inOrder();
  }

  @Test
  public void testWrapTakesOwnershipOfArray() {
    Object[] wrapped = {"hello"};
    Mutability mutability = Mutability.create("test");
    StarlarkList<Object> mutableList = StarlarkList.wrap(mutability, wrapped);

    // Big no-no, but we're proving a point.
    wrapped[0] = "goodbye";
    assertThat((List<?>) mutableList).containsExactly("goodbye");
  }

  @Test
  public void testOfReturnsListWhoseArrayElementTypeIsObject() throws EvalException {
    Mutability mu = Mutability.create("test");
    StarlarkList<Object> list = StarlarkList.of(mu, "a", "b");
    list.addElement(StarlarkInt.of(1)); // no ArrayStoreException
    assertThat(list.toString()).isEqualTo("[\"a\", \"b\", 1]");
  }

  @Test
  public void immutableSingleton() {
    StarlarkList<Object> list = StarlarkList.immutableOf("a");
    checkImmutable(list);
    assertThat((List<?>) list).containsExactly("a");
  }

  @Test
  public void immutableMultiElement() {
    StarlarkList<Object> list = StarlarkList.immutableOf("a", "b", "c");
    checkImmutable(list);
    assertThat((List<?>) list).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void lazyImmutable() {
    AtomicBoolean called = new AtomicBoolean(false);
    Supplier<ImmutableList<Object>> supplier =
        () -> {
          assertThat(called.getAndSet(true)).isFalse();
          return ImmutableList.of("a", "b", "c");
        };
    StarlarkList<Object> list = StarlarkList.lazyImmutable(supplier);
    assertThat(called.get()).isFalse();
    assertThat((List<?>) list).containsExactly("a", "b", "c").inOrder();
    assertThat(list.get(0)).isEqualTo("a"); // Supplier not called twice.

    // Supplier is discarded.
    var ref = new WeakReference<>(supplier);
    supplier = null;
    GcFinalization.awaitClear(ref);

    checkImmutable(list);
  }

  @Test
  public void testStarlarkListToArray() throws Exception {
    Mutability mu = Mutability.create("test");
    StarlarkList<String> list = StarlarkList.newList(mu);

    for (int i = 0; i < 10; ++i) {
      for (int len : new int[] {0, list.size() / 2, list.size(), list.size() * 2}) {
        for (Class<?> elemType : new Class<?>[] {Object.class, String.class}) {
          Object[] input = (Object[]) Array.newInstance(elemType, len);
          try {
            checkToArray(input, list);
          } catch (AssertionError ex) {
            fail("list.toArray(new %s[%d]): %s", elemType.getSimpleName(), len, ex.getMessage());
          }
        }
      }
      // Note we add elements in loop instead of recreating a list
      // to also check that code works correctly when list capacity exceeds size.
      list.addElement(Integer.toString(i));
    }
  }

  @Test
  public void testTupleToArray() {
    Tuple tuple = Tuple.of(IntStream.range(0, 10).mapToObj(Integer::toString).toArray());
    for (int len : new int[] {0, tuple.size() / 2, tuple.size(), tuple.size() * 2}) {
      for (Class<?> elemType : new Class<?>[] {Object.class, String.class}) {
        Object[] input = (Object[]) Array.newInstance(elemType, len);
        try {
          checkToArray(input, tuple);
        } catch (AssertionError ex) {
          fail("tuple.toArray(new %s[%d]): %s", elemType.getSimpleName(), len, ex.getMessage());
        }
      }
    }
  }

  @FormatMethod
  private static void fail(@FormatString String format, Object... args) {
    throw new AssertionError(String.format(format, args));
  }

  private static void checkImmutable(StarlarkList<Object> list) {
    EvalException e = assertThrows(EvalException.class, () -> list.addElement(StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.addElementAt(0, StarlarkInt.of(4)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e =
        assertThrows(
            EvalException.class,
            () ->
                list.addElements(
                    ImmutableList.of(StarlarkInt.of(4), StarlarkInt.of(5), StarlarkInt.of(6))));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.removeElementAt(0));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
    e = assertThrows(EvalException.class, () -> list.setElementAt(0, StarlarkInt.of(10)));
    assertThat(e).hasMessageThat().isEqualTo("trying to mutate a frozen list value");
  }

  // Asserts that seq.toArray(input) returns an array of class input.getClass(),
  // regardless of seq's element type, and contains the correct elements,
  // including trailing null padding if size < len.
  private static void checkToArray(Object[] input, Sequence<?> seq) {
    Arrays.fill(input, "x");

    Object[] output = seq.toArray(input);
    if (output.getClass() != input.getClass()) {
      fail("array class mismatch: input=%s, output=%s", input.getClass(), output.getClass());
    }
    if (input.length < seq.size()) {
      // assert input is unchanged
      for (int i = 0; i < input.length; i++) {
        if (!input[i].equals("x")) {
          fail("input[%d] = %s, want \"x\"", i, Starlark.repr(input[i]));
        }
      }

      Object[] expected = IntStream.range(0, seq.size()).mapToObj(Integer::toString).toArray();
      if (!Arrays.equals(output, expected)) {
        fail("output array = %s, want %s", Arrays.toString(output), Arrays.toString(expected));
      }
    } else if (output != input) {
      for (int j = 0; j < output.length; ++j) {
        String want = j < seq.size() ? Integer.toString(j) : null;
        if (!output[j].equals(want)) {
          fail("output[%d] = %s, want %s", j, output[j], want);
        }
      }
    }
  }
}
