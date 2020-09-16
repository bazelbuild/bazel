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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateCancelledFuture;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.MissingNestedSetException;
import com.google.protobuf.ByteString;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSet}. */
@RunWith(JUnit4.class)
public final class NestedSetTest {

  private static NestedSetBuilder<String> nestedSetBuilder(String... directMembers) {
    NestedSetBuilder<String> builder = NestedSetBuilder.stableOrder();
    builder.addAll(Lists.newArrayList(directMembers));
    return builder;
  }

  @Test
  public void simple() {
    NestedSet<String> set = nestedSetBuilder("a").build();

    assertThat(set.toList()).containsExactly("a");
    assertThat(set.isEmpty()).isFalse();
  }

  @Test
  public void flatToString() {
    assertThat(nestedSetBuilder().build().toString()).isEqualTo("[]");
    assertThat(nestedSetBuilder("a").build().toString()).isEqualTo("[a]");
    assertThat(nestedSetBuilder("a", "b").build().toString()).isEqualTo("[a, b]");
  }

  @Test
  public void nestedToString() {
    NestedSet<String> b = nestedSetBuilder("b1", "b2").build();
    NestedSet<String> c = nestedSetBuilder("c1", "c2").build();

    assertThat(nestedSetBuilder("a").addTransitive(b).build().toString()).isEqualTo("[b1, b2, a]");
    assertThat(nestedSetBuilder("a").addTransitive(b).addTransitive(c).build().toString())
        .isEqualTo("[b1, b2, c1, c2, a]");
    NestedSet<String> linkOrderSet =
        NestedSetBuilder.<String>linkOrder().add("a").addTransitive(b).addTransitive(c).build();
    assertThat(linkOrderSet.toString()).isEqualTo("[a, b2, b1, c2, c1]");

    assertThat(nestedSetBuilder().addTransitive(b).build().toString()).isEqualTo("[b1, b2]");
  }

  @Test
  public void tooLongToString() {
    NestedSetBuilder<Integer> builder = NestedSetBuilder.stableOrder();
    for (int i = 0; i < NestedSet.MAX_ELEMENTS_TO_STRING + 3; i++) {
      builder.add(i);
    }
    String stringRep = builder.build().toString();
    assertThat(stringRep).contains("[0, 1, 2, 3");
    assertThat(stringRep)
        .containsMatch(
            "\\[0, 1, 2, 3, .*"
                + (NestedSet.MAX_ELEMENTS_TO_STRING - 2)
                + ", "
                + (NestedSet.MAX_ELEMENTS_TO_STRING - 1)
                + "] \\(truncated, full size "
                + (NestedSet.MAX_ELEMENTS_TO_STRING + 3)
                + "\\)");
  }

  @Test
  public void isEmpty() {
    NestedSet<String> triviallyEmpty = nestedSetBuilder().build();
    assertThat(triviallyEmpty.isEmpty()).isTrue();

    NestedSet<String> emptyLevel1 = nestedSetBuilder().addTransitive(triviallyEmpty).build();
    assertThat(emptyLevel1.isEmpty()).isTrue();

    NestedSet<String> emptyLevel2 = nestedSetBuilder().addTransitive(emptyLevel1).build();
    assertThat(emptyLevel2.isEmpty()).isTrue();

    NestedSet<String> triviallyNonEmpty = nestedSetBuilder("mango").build();
    assertThat(triviallyNonEmpty.isEmpty()).isFalse();

    NestedSet<String> nonEmptyLevel1 = nestedSetBuilder().addTransitive(triviallyNonEmpty).build();
    assertThat(nonEmptyLevel1.isEmpty()).isFalse();

    NestedSet<String> nonEmptyLevel2 = nestedSetBuilder().addTransitive(nonEmptyLevel1).build();
    assertThat(nonEmptyLevel2.isEmpty()).isFalse();
  }

  @Test
  public void canIncludeAnyOrderInStableOrderAndViceVersa() {
    NestedSetBuilder.stableOrder()
        .addTransitive(
            NestedSetBuilder.compileOrder()
                .addTransitive(NestedSetBuilder.stableOrder().build())
                .build())
        .addTransitive(
            NestedSetBuilder.linkOrder()
                .addTransitive(NestedSetBuilder.stableOrder().build())
                .build())
        .addTransitive(
            NestedSetBuilder.naiveLinkOrder()
                .addTransitive(NestedSetBuilder.stableOrder().build())
                .build())
        .build();
    assertThrows(
        "Shouldn't be able to include a non-stable order inside a different non-stable order!",
        IllegalArgumentException.class,
        () ->
            NestedSetBuilder.compileOrder()
                .addTransitive(NestedSetBuilder.linkOrder().build())
                .build());
  }

  /**
   * A handy wrapper that allows us to use EqualsTester to test shallowEquals and shallowHashCode.
   */
  private static final class SetWrapper<E> {
    NestedSet<E> set;

    SetWrapper(NestedSet<E> wrapped) {
      set = wrapped;
    }

    @Override
    public int hashCode() {
      return set.shallowHashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SetWrapper)) {
        return false;
      }
      try {
        @SuppressWarnings("unchecked")
        SetWrapper<E> other = (SetWrapper<E>) o;
        return set.shallowEquals(other.set);
      } catch (ClassCastException e) {
        return false;
      }
    }
  }

  @SafeVarargs
  private static <E> SetWrapper<E> flat(E... directMembers) {
    NestedSetBuilder<E> builder = NestedSetBuilder.stableOrder();
    builder.addAll(Lists.newArrayList(directMembers));
    return new SetWrapper<>(builder.build());
  }

  @SafeVarargs
  private static <E> SetWrapper<E> nest(SetWrapper<E>... nested) {
    NestedSetBuilder<E> builder = NestedSetBuilder.stableOrder();
    for (SetWrapper<E> wrap : nested) {
      builder.addTransitive(wrap.set);
    }
    return new SetWrapper<>(builder.build());
  }

  @SafeVarargs
  // Restricted to <Integer> to avoid ambiguity with the other nest() function.
  private static SetWrapper<Integer> nest(Integer elem, SetWrapper<Integer>... nested) {
    NestedSetBuilder<Integer> builder = NestedSetBuilder.stableOrder();
    builder.add(elem);
    for (SetWrapper<Integer> wrap : nested) {
      builder.addTransitive(wrap.set);
    }
    return new SetWrapper<>(builder.build());
  }

  private static final int UNKNOWN_DEPTH = 7;

  @Test
  public void shallowEquality() {
    // Used below to check that inner nested sets can be compared by reference equality.
    SetWrapper<Integer> myRef = nest(nest(flat(7, 8)), flat(9));
    // Used to check equality for deserializing nested sets
    ListenableFuture<Object[]> contents = immediateFuture(new Object[] {"a", "b"});
    NestedSet<String> referenceNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, contents);
    NestedSet<String> otherReferenceNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, contents);

    // Each "equality group" contains elements that are equal to one another
    // (according to equals() and hashCode()), yet distinct from all elements
    // of all other equality groups.
    new EqualsTester()
        .addEqualityGroup(flat(), flat(), nest(flat())) // Empty set elision.
        .addEqualityGroup(NestedSetBuilder.<Integer>linkOrder().build())
        .addEqualityGroup(flat(3), flat(3), flat(3, 3)) // Element de-duplication.
        .addEqualityGroup(flat(4), nest(flat(4))) // Automatic elision of one-element nested sets.
        .addEqualityGroup(NestedSetBuilder.<Integer>linkOrder().add(4).build())
        .addEqualityGroup(nestedSetBuilder("4").build()) // Like flat("4").
        .addEqualityGroup(flat(3, 4), flat(3, 4))
        // Make a couple sets deep enough that shallowEquals() fails.
        // If this test case fails because you improve the representation, just delete it.
        .addEqualityGroup(nest(nest(flat(3, 4), flat(5)), nest(flat(6, 7), flat(8))))
        .addEqualityGroup(nest(nest(flat(3, 4), flat(5)), nest(flat(6, 7), flat(8))))
        .addEqualityGroup(nest(myRef), nest(myRef), nest(myRef, myRef)) // Set de-duplication.
        .addEqualityGroup(nest(3, myRef))
        .addEqualityGroup(nest(4, myRef))
        .addEqualityGroup(
            new SetWrapper<>(referenceNestedSet), new SetWrapper<>(otherReferenceNestedSet))
        .testEquals();

    // Some things that are not tested by the above:
    //  - ordering among direct members
    //  - ordering among transitive sets
  }

  @Test
  public void shallowInequality() {
    assertThat(nestedSetBuilder("a").build().shallowEquals(null)).isFalse();
    Object[] contents = {"a", "b"};
    assertThat(
            NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateFuture(contents))
                .shallowEquals(null))
        .isFalse();

    // shallowEquals() should require reference equality for underlying futures
    assertThat(
            NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateFuture(contents))
                .shallowEquals(
                    NestedSet.withFuture(
                        Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateFuture(contents))))
        .isFalse();
  }

  /** Checks that the builder always return a nested set with the correct order. */
  @Test
  public void correctOrder() {
    for (Order order : Order.values()) {
      for (int numDirects = 0; numDirects < 3; numDirects++) {
        for (int numTransitives = 0; numTransitives < 3; numTransitives++) {
          assertThat(createNestedSet(order, numDirects, numTransitives, order).getOrder())
              .isEqualTo(order);
          // We allow mixing orders if one of them is stable. This tests that the top level order is
          // the correct one.
          assertThat(
                  createNestedSet(order, numDirects, numTransitives, Order.STABLE_ORDER).getOrder())
              .isEqualTo(order);
        }
      }
    }
  }

  private static NestedSet<Integer> createNestedSet(
      Order order, int numDirects, int numTransitives, Order transitiveOrder) {
    NestedSetBuilder<Integer> builder = new NestedSetBuilder<>(order);

    for (int direct = 0; direct < numDirects; direct++) {
      builder.add(direct);
    }
    for (int transitive = 0; transitive < numTransitives; transitive++) {
      builder.addTransitive(new NestedSetBuilder<Integer>(transitiveOrder).add(transitive).build());
    }
    return builder.build();
  }

  @Test
  public void memoizedFlattenAndGetSize() {
    NestedSet<String> empty = NestedSetBuilder.<String>stableOrder().build();
    checkSize(empty, 0); // {}

    NestedSet<String> singleton = NestedSetBuilder.<String>stableOrder().add("a").build();
    checkSize(singleton, 1); // {a}

    NestedSet<String> deuce = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    checkSize(deuce, 2); // {a, b}

    checkSize(
        NestedSetBuilder.<String>stableOrder()
            .add("a")
            .addTransitive(deuce)
            .addTransitive(singleton)
            .addTransitive(empty)
            .build(),
        2); // {a, b}
    checkSize(
        NestedSetBuilder.<String>stableOrder()
            .add("c")
            .addTransitive(deuce)
            .addTransitive(singleton)
            .addTransitive(empty)
            .build(),
        3); // {a, b, c}

    // 25000 has a 3-digit base128 encoding.
    NestedSetBuilder<Integer> largeShallow = NestedSetBuilder.stableOrder();
    for (int i = 0; i < 25000; ++i) {
      largeShallow.add(i);
    }
    checkSize(largeShallow.build(), 25000); // {0, 1, ..., 24999}

    // a deep and narrow graph
    NestedSet<String> deep = deuce;
    for (int i = 0; i < 200; ++i) {
      deep = NestedSetBuilder.<String>stableOrder().addTransitive(deep).add("c").build();
    }
    checkSize(deep, 3); // {a, b, c}
  }

  private static void checkSize(NestedSet<?> set, int size) {
    assertThat(set.memoizedFlattenAndGetSize()).isEqualTo(size); // first call: flattens
    assertThat(set.memoizedFlattenAndGetSize()).isEqualTo(size); // second call: memoized
  }

  @Test
  public void hoistingKeepsSetSmall() {
    NestedSet<String> first = NestedSetBuilder.<String>stableOrder().add("a").build();
    NestedSet<String> second = NestedSetBuilder.<String>stableOrder().add("a").build();
    NestedSet<String> singleton =
        NestedSetBuilder.<String>stableOrder().addTransitive(first).addTransitive(second).build();
    assertThat(singleton.toList()).containsExactly("a");
    assertThat(singleton.isSingleton()).isTrue();
  }

  @Test
  public void buildInterruptibly_propagatesInterrupt() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    NestedSetBuilder<String> builder =
        NestedSetBuilder.<String>stableOrder().addTransitive(deserializingNestedSet).add("a");
    Thread.currentThread().interrupt();
    assertThrows(InterruptedException.class, builder::buildInterruptibly);
  }

  @Test
  public void getChildrenInterruptibly_propagatesInterrupt() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    Thread.currentThread().interrupt();
    assertThrows(InterruptedException.class, deserializingNestedSet::getChildrenInterruptibly);
  }

  @Test
  public void toListInterruptibly_propagatesInterrupt() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    Thread.currentThread().interrupt();
    assertThrows(InterruptedException.class, deserializingNestedSet::toListInterruptibly);
  }

  @Test
  public void toListInterruptibly_propagatesMissingNestedSetException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(
                new MissingNestedSetException(ByteString.copyFromUtf8("fingerprint"))));
    assertThrows(MissingNestedSetException.class, deserializingNestedSet::toListInterruptibly);
  }

  @Test
  public void toListWithTimeout_propagatesInterrupt() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    Thread.currentThread().interrupt();
    assertThrows(
        InterruptedException.class,
        () -> deserializingNestedSet.toListWithTimeout(Duration.ofDays(1)));
  }

  @Test
  public void toListWithTimeout_propagatesMissingNestedSetException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(
                new MissingNestedSetException(ByteString.copyFromUtf8("fingerprint"))));
    assertThrows(
        MissingNestedSetException.class,
        () -> deserializingNestedSet.toListWithTimeout(Duration.ofNanos(1)));
  }

  @Test
  public void toListWithTimeout_timesOut() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    assertThrows(
        TimeoutException.class,
        () -> deserializingNestedSet.toListWithTimeout(Duration.ofNanos(1)));
  }

  @Test
  public void toListWithTimeout_waits() throws Exception {
    SettableFuture<Object[]> future = SettableFuture.create();
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, future);
    Future<ImmutableList<String>> result =
        Executors.newSingleThreadExecutor()
            .submit(() -> deserializingNestedSet.toListWithTimeout(Duration.ofMinutes(1)));
    Thread.sleep(100);
    assertThat(result.isDone()).isFalse();
    future.set(new Object[] {"a", "b"});
    assertThat(result.get()).containsExactly("a", "b");
  }

  @Test
  public void isFromStorage_true() {
    NestedSet<?> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, SettableFuture.create());
    assertThat(deserializingNestedSet.isFromStorage()).isTrue();
  }

  @Test
  public void isFromStorage_false() {
    NestedSet<?> inMemoryNestedSet = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    assertThat(inMemoryNestedSet.isFromStorage()).isFalse();
  }

  @Test
  public void isReady_inMemory() {
    NestedSet<?> inMemoryNestedSet = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b");
    assertThat(inMemoryNestedSet.isReady()).isTrue();
  }

  @Test
  public void isReady_fromStorage() {
    SettableFuture<Object[]> future = SettableFuture.create();
    NestedSet<?> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, future);
    assertThat(deserializingNestedSet.isReady()).isFalse();
    future.set(new Object[] {"a", "b"});
    assertThat(deserializingNestedSet.isReady()).isTrue();
  }

  @Test
  public void isReady_fromStorage_cancelled() {
    NestedSet<?> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateCancelledFuture());
    assertThat(deserializingNestedSet.isReady()).isFalse();
  }

  @Test
  public void isReady_fromStorage_failed() {
    NestedSet<?> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(
                new MissingNestedSetException(ByteString.copyFromUtf8("fingerprint"))));
    assertThat(deserializingNestedSet.isReady()).isFalse();
  }

  @Test
  public void getApproxDepth() {
    NestedSet<String> empty = nestedSetBuilder().build();
    NestedSet<String> justA = nestedSetBuilder("a").build();
    NestedSet<String> justB = nestedSetBuilder("b").build();
    NestedSet<String> ab = nestedSetBuilder().addTransitive(justA).addTransitive(justB).build();

    assertThat(empty.getApproxDepth()).isEqualTo(0);
    assertThat(
            nestedSetBuilder().addTransitive(empty).addTransitive(empty).build().getApproxDepth())
        .isEqualTo(0);
    assertThat(justA.getApproxDepth()).isEqualTo(1);
    assertThat(justB.getApproxDepth()).isEqualTo(1);
    assertThat(
            nestedSetBuilder().addTransitive(empty).addTransitive(empty).build().getApproxDepth())
        .isEqualTo(0);
    assertThat(
            nestedSetBuilder().addTransitive(empty).addTransitive(justA).build().getApproxDepth())
        .isEqualTo(1);
    assertThat(
            nestedSetBuilder().addTransitive(justA).addTransitive(empty).build().getApproxDepth())
        .isEqualTo(1);
    assertThat(
            nestedSetBuilder().addTransitive(justA).addTransitive(justA).build().getApproxDepth())
        .isEqualTo(1);
    assertThat(
            nestedSetBuilder().addTransitive(justA).addTransitive(justB).build().getApproxDepth())
        .isEqualTo(2);
    assertThat(
            nestedSetBuilder("a", "b", "c")
                .addTransitive(justA)
                .addTransitive(justB)
                .addTransitive(ab)
                .build()
                .getApproxDepth())
        .isEqualTo(3);
  }
}
