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
import static com.google.devtools.build.lib.collect.nestedset.Order.LINK_ORDER;
import static com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint.getFingerprintForTesting;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore.MissingFingerprintValueException;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.ByteString;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSet}. */
@RunWith(JUnit4.class)
public final class NestedSetTest {
  private final FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final ArtifactRoot artifactRoot =
      ActionsTestUtil.createArtifactRootFromTwoPaths(
          fileSystem.getPath("/root1"), fileSystem.getPath("/root1/root2"));

  private static <T> NestedSetBuilder<T> nestedSetBuilder(T... directMembers) {
    NestedSetBuilder<T> builder = NestedSetBuilder.stableOrder();
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
    NestedSet<String> triviallyEmpty = NestedSetTest.<String>nestedSetBuilder().build();
    assertThat(triviallyEmpty.isEmpty()).isTrue();

    NestedSet<String> emptyLevel1 =
        NestedSetTest.<String>nestedSetBuilder().addTransitive(triviallyEmpty).build();
    assertThat(emptyLevel1.isEmpty()).isTrue();

    NestedSet<String> emptyLevel2 =
        NestedSetTest.<String>nestedSetBuilder().addTransitive(emptyLevel1).build();
    assertThat(emptyLevel2.isEmpty()).isTrue();

    NestedSet<String> triviallyNonEmpty = nestedSetBuilder("mango").build();
    assertThat(triviallyNonEmpty.isEmpty()).isFalse();

    NestedSet<String> nonEmptyLevel1 =
        NestedSetTest.<String>nestedSetBuilder().addTransitive(triviallyNonEmpty).build();
    assertThat(nonEmptyLevel1.isEmpty()).isFalse();

    NestedSet<String> nonEmptyLevel2 =
        NestedSetTest.<String>nestedSetBuilder().addTransitive(nonEmptyLevel1).build();
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

  @Test
  public void reusesSingleTransitiveSet_noDirectMembers() {
    NestedSet<String> set = NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b", "c");
    NestedSet<String> built = NestedSetBuilder.<String>stableOrder().addTransitive(set).build();
    assertThat(built).isSameInstanceAs(set);
  }

  @Test
  public void reusesSingleTransitiveSet_singletonEqualsDirects() {
    NestedSet<String> set = NestedSetBuilder.create(Order.STABLE_ORDER, "a");
    NestedSet<String> built =
        NestedSetBuilder.<String>stableOrder().add("a").addTransitive(set).build();
    assertThat(built).isSameInstanceAs(set);
  }

  @Test
  public void addAll_rejectsNullElements() {
    NestedSetBuilder<String> builder1 = NestedSetBuilder.stableOrder();
    List<String> listWithNull1 = Arrays.asList("a", null);
    assertThrows(NullPointerException.class, () -> builder1.addAll(listWithNull1));

    NestedSetBuilder<String> builder2 = NestedSetBuilder.stableOrder();
    List<String> listWithNull2 = Arrays.asList("a", "b", "c", null);
    assertThrows(NullPointerException.class, () -> builder2.addAll(listWithNull2));
  }

  @Test
  public void builder_rejectsObjectArrayAndByteStringElements() {
    NestedSetBuilder<Object> builder1 = NestedSetBuilder.stableOrder();
    Object[] objArray = new String[] {"a", "b"};
    assertThrows(IllegalArgumentException.class, () -> builder1.add(objArray));

    NestedSetBuilder<Object> builder2 = NestedSetBuilder.stableOrder();
    ByteString byteString = ByteString.copyFromUtf8("abc");
    assertThrows(IllegalArgumentException.class, () -> builder2.add(byteString));

    NestedSetBuilder<Object> builder3 = NestedSetBuilder.stableOrder();
    List<Object> listWithObjectArray = Arrays.asList("a", "b", objArray);
    assertThrows(IllegalArgumentException.class, () -> builder3.addAll(listWithObjectArray));

    NestedSetBuilder<Object> builder4 = NestedSetBuilder.stableOrder().add("a");
    assertThrows(IllegalArgumentException.class, () -> builder4.add(objArray));

    NestedSetBuilder<Object> builder5 = NestedSetBuilder.stableOrder().add("a");
    assertThrows(IllegalArgumentException.class, () -> builder5.add(byteString));
  }

  @Test
  public void directElements_supportsArrayElements() {
    byte[] a = new byte[] {1, 2};
    byte[] b = new byte[] {3, 4};

    NestedSet<byte[]> set = NestedSetBuilder.<byte[]>stableOrder().add(a).add(b).build();

    assertThat(set.toList()).containsExactly(a, b).inOrder();
  }

  @Test
  public void directElements_supportsSetElements() {
    ImmutableSet<String> s1 = ImmutableSet.of("x");
    ImmutableSet<String> s2 = ImmutableSet.of("y");

    NestedSet<ImmutableSet<String>> set =
        NestedSetBuilder.<ImmutableSet<String>>stableOrder().add(s1).add(s2).build();

    assertThat(set.toList()).containsExactly(s1, s2).inOrder();
  }

  @Test
  public void noReuseOfSingleTransitiveSet_orderWouldDiffer() {
    NestedSet<String> set = NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, "b", "a");
    NestedSet<String> built =
        NestedSetBuilder.<String>naiveLinkOrder().add("a").add("b").addTransitive(set).build();
    assertThat(built).isNotSameInstanceAs(set);
    assertThat(set.toList()).containsExactly("b", "a").inOrder();
    assertThat(built.toList()).containsExactly("a", "b").inOrder();
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
      if (o instanceof SetWrapper<?> other) {
        return set.shallowEquals(other.set);
      }
      return false;
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
        // NestedSet<String> gets interned under the hood.
        .addEqualityGroup(
            nest(nest(flat("a", "b"), flat("c")), nest(flat("d", "e"), flat("f"))),
            nest(nest(flat("a", "b"), flat("c")), nest(flat("d", "e"), flat("f"))))
        // NestedSet<Integer> does not get interned under the hood.
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
    NestedSetBuilder<Integer> builder = NestedSetBuilder.newBuilder(order);

    for (int direct = 0; direct < numDirects; direct++) {
      builder.add(direct);
    }
    for (int transitive = 0; transitive < numTransitives; transitive++) {
      builder.addTransitive(NestedSet.<Integer>builder(transitiveOrder).add(transitive).build());
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
  public void concurrentMemoizedFlattenAndGetSize() throws Exception {
    NestedSet<String> deep = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    for (int i = 0; i < 200; ++i) {
      deep = NestedSetBuilder.<String>stableOrder().addTransitive(deep).add("c").build();
    }
    NestedSet<String> underTest = deep;
    List<TestThread> threads = new ArrayList<>(20);
    for (int i = 0; i < 20; i++) {
      threads.add(new TestThread(underTest::memoizedFlattenAndGetSize));
    }
    for (TestThread thread : threads) {
      thread.start();
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    }
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
  public void toListInterruptibly_propagatesMissingFingerprintValueException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(
                new MissingFingerprintValueException(getFingerprintForTesting("fingerprint"))));
    assertThrows(
        MissingFingerprintValueException.class, deserializingNestedSet::toListInterruptibly);
  }

  @Test
  public void toListInterruptibly_propagatesSerializationException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(new SerializationException("test exception")));
    assertThrows(SerializationException.class, deserializingNestedSet::toListInterruptibly);
  }

  @Test
  public void toListInterruptibly_propagatesCancellationAsMissingFingerprintValueException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateCancelledFuture());
    assertThrows(
        MissingFingerprintValueException.class, deserializingNestedSet::toListInterruptibly);
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
  public void toListWithTimeout_propagatesMissingFingerprintValueException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(
                new MissingFingerprintValueException(getFingerprintForTesting("fingerprint"))));
    assertThrows(
        MissingFingerprintValueException.class,
        () -> deserializingNestedSet.toListWithTimeout(Duration.ofNanos(1)));
  }

  @Test
  public void toListWithTimeout_propagatesSerializationException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(
            Order.STABLE_ORDER,
            UNKNOWN_DEPTH,
            immediateFailedFuture(new SerializationException("test exception")));
    assertThrows(
        SerializationException.class,
        () -> deserializingNestedSet.toListWithTimeout(Duration.ofNanos(1)));
  }

  @Test
  public void toListWithTimeout_propagatesCancellationAsMissingFingerprintValueException() {
    NestedSet<String> deserializingNestedSet =
        NestedSet.withFuture(Order.STABLE_ORDER, UNKNOWN_DEPTH, immediateCancelledFuture());
    assertThrows(
        MissingFingerprintValueException.class,
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
                new MissingFingerprintValueException(getFingerprintForTesting("fingerprint"))));
    assertThat(deserializingNestedSet.isReady()).isFalse();
  }

  @Test
  public void getApproxDepth() {
    NestedSet<String> empty = NestedSetTest.<String>nestedSetBuilder().build();
    NestedSet<String> justA = nestedSetBuilder("a").build();
    NestedSet<String> justB = nestedSetBuilder("b").build();
    NestedSet<String> ab =
        NestedSetTest.<String>nestedSetBuilder().addTransitive(justA).addTransitive(justB).build();

    assertThat(empty.getApproxDepth()).isEqualTo(0);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(empty)
                .addTransitive(empty)
                .build()
                .getApproxDepth())
        .isEqualTo(0);
    assertThat(justA.getApproxDepth()).isEqualTo(1);
    assertThat(justB.getApproxDepth()).isEqualTo(1);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(empty)
                .addTransitive(empty)
                .build()
                .getApproxDepth())
        .isEqualTo(0);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(empty)
                .addTransitive(justA)
                .build()
                .getApproxDepth())
        .isEqualTo(1);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(justA)
                .addTransitive(empty)
                .build()
                .getApproxDepth())
        .isEqualTo(1);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(justA)
                .addTransitive(justA)
                .build()
                .getApproxDepth())
        .isEqualTo(1);
    assertThat(
            NestedSetTest.<String>nestedSetBuilder()
                .addTransitive(justA)
                .addTransitive(justB)
                .build()
                .getApproxDepth())
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

  @Test
  public void linkOrder_toList_withTransitiveInputAliases_areConsistent() {
    NestedSet<String> inputA = NestedSetBuilder.create(LINK_ORDER, "A");
    NestedSet<String> inputB = NestedSetBuilder.create(LINK_ORDER, "B");
    NestedSet<String> inputC = NestedSetBuilder.create(LINK_ORDER, "C");
    NestedSet<String> inputB2 = NestedSetBuilder.create(LINK_ORDER, "B");

    NestedSet<String> withDuplicates =
        NestedSet.<String>builder(LINK_ORDER)
            .addTransitive(inputA)
            .addTransitive(inputB)
            .addTransitive(inputC)
            .addTransitive(inputB)
            .build();

    NestedSet<String> withAlias =
        NestedSet.<String>builder(LINK_ORDER)
            .addTransitive(inputA)
            .addTransitive(inputB)
            .addTransitive(inputC)
            .addTransitive(inputB2)
            .build();

    assertThat(withAlias.toList()).isEqualTo(withDuplicates.toList());
  }

  @Test
  public void linkOrder_toList_withDuplicateDirectInputs_keepsFirst() {
    NestedSet<String> duplicateInputs = NestedSetBuilder.create(LINK_ORDER, "A", "B", "C", "A");
    assertThat(duplicateInputs.toList()).containsExactly("A", "B", "C").inOrder();
  }

  @Test
  public void interning_multiElementNestedSetOfString_childrenArrayIsInterned() {
    NestedSet<String> nestedSetA = nestedSetBuilder("cat", "dog").build();
    NestedSet<String> nestedSetB = nestedSetBuilder("cat", "dog").build();
    assertThat(nestedSetB.getChildren()).isSameInstanceAs(nestedSetA.getChildren());
    assertThat(nestedSetB).isNotSameInstanceAs(nestedSetA);
    NestedSetInterner.clear();
    NestedSet<String> nestedSetC = nestedSetBuilder("cat", "dog").build();
    assertThat(nestedSetC.getChildren()).isNotSameInstanceAs(nestedSetA.getChildren());
  }

  @Test
  public void interning_singletonNestedSetOfString_isNotInterned() {
    NestedSet<String> singletonA = nestedSetBuilder("cat").build();
    NestedSet<String> singletonB = nestedSetBuilder("cat").build();
    assertThat(singletonB).isNotSameInstanceAs(singletonA);
  }

  @Test
  public void interning_multiElementNestedSetOfInteger_isNotInterned() {
    NestedSet<Integer> nestedSetA = nestedSetBuilder(1, 2).build();
    NestedSet<Integer> nestedSetB = nestedSetBuilder(1, 2).build();
    assertThat(nestedSetA.getChildren()).isNotSameInstanceAs(nestedSetB.getChildren());
  }

  @Test
  public void
      interning_multiElementNestedSetOfArtifact_childrenArrayIsInternedByArtifactIdentity() {
    Artifact a1 = ActionsTestUtil.createArtifact(artifactRoot, "a");
    Artifact a2 = ActionsTestUtil.createArtifact(artifactRoot, "a");
    Artifact b = ActionsTestUtil.createArtifact(artifactRoot, "b");

    NestedSet<Artifact> a1bFirst = nestedSetBuilder(a1, b).build();
    NestedSet<Artifact> a1bSecond = nestedSetBuilder(a1, b).build();
    NestedSet<Artifact> a2b = nestedSetBuilder(a2, b).build();
    assertThat(a1bSecond.getChildren()).isSameInstanceAs(a1bFirst.getChildren());
    assertThat(a1bSecond).isNotSameInstanceAs(a1bFirst);
    assertThat(a2b.getChildren()).isNotSameInstanceAs(a1bFirst.getChildren());
    assertThat(a2b.toList().getFirst()).isSameInstanceAs(a2);
    NestedSetInterner.clear();
    assertThat(nestedSetBuilder(a1, b).build().getChildren())
        .isNotSameInstanceAs(a1bFirst.getChildren());
  }

  @Test
  public void builder_inlineDirectItems_handlesDeduplicationAndPromotion() {
    // 0 items
    assertThat(NestedSetBuilder.<String>stableOrder().isEmpty()).isTrue();

    // 1 item + duplicates
    NestedSetBuilder<String> b1 = NestedSetBuilder.<String>stableOrder().add("x").add("x");
    assertThat(b1.isEmpty()).isFalse();
    NestedSet<String> set1 = b1.build();
    assertThat(set1.toList()).containsExactly("x");

    // 2 items + duplicates
    NestedSetBuilder<String> b2 =
        NestedSetBuilder.<String>stableOrder().add("x").add("y").add("x").add("y");
    NestedSet<String> set2 = b2.build();
    assertThat(set2.toList()).containsExactly("x", "y").inOrder();

    // 3 items (promotes to CompactHashSet) + duplicates
    NestedSetBuilder<String> b3 =
        NestedSetBuilder.<String>stableOrder().add("x").add("y").add("z").add("x").add("z");
    NestedSet<String> set3 = b3.build();
    assertThat(set3.toList()).containsExactly("x", "y", "z").inOrder();
  }

  @Test
  public void builder_inlineTransitiveSets_handlesDeduplicationAndPromotion() {
    NestedSet<String> sub1 = NestedSetBuilder.<String>stableOrder().add("a").build();
    NestedSet<String> sub2 = NestedSetBuilder.<String>stableOrder().add("b").build();
    NestedSet<String> sub3 = NestedSetBuilder.<String>stableOrder().add("c").build();

    // 1 transitive set + duplicates: should reuse the candidate instance
    NestedSetBuilder<String> b1 =
        NestedSetBuilder.<String>stableOrder().addTransitive(sub1).addTransitive(sub1);
    assertThat(b1.isEmpty()).isFalse();
    NestedSet<String> set1 = b1.build();
    assertThat(set1).isSameInstanceAs(sub1);

    // 2 transitive sets + duplicates
    NestedSetBuilder<String> b2 =
        NestedSetBuilder.<String>stableOrder()
            .addTransitive(sub1)
            .addTransitive(sub2)
            .addTransitive(sub1);
    NestedSet<String> set2 = b2.build();
    assertThat(set2.toList()).containsExactly("a", "b").inOrder();

    // 3 transitive sets (promotes to CompactHashSet) + duplicates
    NestedSetBuilder<String> b3 =
        NestedSetBuilder.<String>stableOrder()
            .addTransitive(sub1)
            .addTransitive(sub2)
            .addTransitive(sub3)
            .addTransitive(sub2);
    NestedSet<String> set3 = b3.build();
    assertThat(set3.toList()).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void builder_directArrayAssembly_preservesOrderAndDeduplicates() {
    NestedSet<String> linkOrderSet =
        NestedSetBuilder.<String>linkOrder().add("first").add("second").build();
    assertThat(linkOrderSet.toList()).containsExactly("first", "second").inOrder();

    NestedSet<String> stableOrderSet =
        NestedSetBuilder.<String>stableOrder().add("first").add("second").build();
    assertThat(stableOrderSet.toList()).containsExactly("first", "second").inOrder();
  }

  @Test
  public void builder_twoCompoundTransitiveSets_assemblesDirectly() {
    NestedSet<String> sub1 = NestedSetBuilder.<String>stableOrder().add("a1").add("a2").build();
    NestedSet<String> sub2 = NestedSetBuilder.<String>stableOrder().add("b1").add("b2").build();

    NestedSet<String> compound =
        NestedSetBuilder.<String>stableOrder().addTransitive(sub1).addTransitive(sub2).build();
    assertThat(compound.toList()).containsExactly("a1", "a2", "b1", "b2").inOrder();
  }

  @Test
  public void builder_mixedDirectAndTransitive_buildsCorrectly() {
    NestedSet<String> sub = NestedSetBuilder.<String>stableOrder().add("t1").add("t2").build();

    // In STABLE_ORDER, transitive elements are visited before direct elements
    NestedSet<String> set =
        NestedSetBuilder.<String>stableOrder().add("d1").add("d2").addTransitive(sub).build();
    assertThat(set.toList()).containsExactly("t1", "t2", "d1", "d2").inOrder();
  }

  @Test
  public void builder_addAll_normalizesDuplicates() {
    NestedSet<String> singleUnique =
        NestedSetBuilder.<String>stableOrder().addAll(ImmutableList.of("a", "a", "a")).build();
    assertThat(singleUnique.isSingleton()).isTrue();
    assertThat(singleUnique.toList()).containsExactly("a");

    NestedSet<String> twoUnique =
        NestedSetBuilder.<String>stableOrder().addAll(ImmutableList.of("a", "b", "a")).build();
    assertThat(twoUnique.toList()).containsExactly("a", "b").inOrder();
  }

  @Test
  public void builder_twoCompoundTransitiveSets_sharingSameInternedArray_collapses() {
    NestedSet<String> sub1 = NestedSetBuilder.<String>stableOrder().add("x").add("y").build();
    NestedSet<String> sub2 = NestedSetBuilder.<String>stableOrder().add("x").add("y").build();
    assertThat(sub1).isNotSameInstanceAs(sub2);
    assertThat(sub1.getChildren()).isSameInstanceAs(sub2.getChildren());

    NestedSet<String> sameOrder =
        NestedSetBuilder.<String>stableOrder().addTransitive(sub1).addTransitive(sub2).build();
    assertThat(sameOrder).isSameInstanceAs(sub1);

    NestedSet<String> subCompile =
        NestedSetBuilder.<String>compileOrder().add("x").add("y").build();
    NestedSet<String> matchesSecond =
        NestedSetBuilder.<String>stableOrder()
            .addTransitive(subCompile)
            .addTransitive(sub2)
            .build();
    assertThat(matchesSecond).isSameInstanceAs(sub2);

    NestedSet<String> diffOrder =
        NestedSetBuilder.<String>compileOrder().addTransitive(sub1).addTransitive(sub2).build();
    assertThat(diffOrder.getOrder()).isEqualTo(Order.COMPILE_ORDER);
    assertThat(diffOrder.getApproxDepth()).isEqualTo(sub1.getApproxDepth());
    assertThat(diffOrder.toList()).containsExactly("x", "y").inOrder();
  }

  @Test
  public void builder_singleTransitiveAndMatchingDirect_reusesSingletonCandidate() {
    NestedSet<String> singleton = NestedSetBuilder.create(Order.STABLE_ORDER, "x");

    // Matching direct member + singleton candidate -> candidate reused
    assertThat(NestedSetBuilder.<String>stableOrder().add("x").addTransitive(singleton).build())
        .isSameInstanceAs(singleton);

    // Non-matching direct member + singleton candidate -> new set
    NestedSet<String> notReused =
        NestedSetBuilder.<String>stableOrder().add("y").addTransitive(singleton).build();
    assertThat(notReused).isNotSameInstanceAs(singleton);
    assertThat(notReused.toList()).containsExactly("x", "y").inOrder();

    // Compound candidate + direct member -> new compound set without blocking
    NestedSet<String> compound = NestedSetBuilder.<String>stableOrder().add("x").add("y").build();
    NestedSet<String> withCompound =
        NestedSetBuilder.<String>stableOrder().add("x").addTransitive(compound).build();
    assertThat(withCompound).isNotSameInstanceAs(compound);
    assertThat(withCompound.toList()).containsExactly("x", "y").inOrder();
  }

  @Test
  public void visitDirectDeps_singleton_visitsLeafOnly() {
    Artifact a = ActionsTestUtil.createArtifact(artifactRoot, "a");
    NestedSet<Artifact> set = NestedSetBuilder.create(Order.STABLE_ORDER, a);

    List<Artifact> leaves = new ArrayList<>();
    List<SkyKey> nonLeaves = new ArrayList<>();
    ArtifactNestedSetKey.visitDirectDeps(set, leaves::add, nonLeaves::add);

    assertThat(leaves).containsExactly(a);
    assertThat(nonLeaves).isEmpty();
  }

  @Test
  public void visitDirectDeps_empty_visitsNothing() {
    NestedSet<Artifact> set = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

    List<Artifact> leaves = new ArrayList<>();
    List<SkyKey> nonLeaves = new ArrayList<>();
    ArtifactNestedSetKey.visitDirectDeps(set, leaves::add, nonLeaves::add);

    assertThat(leaves).isEmpty();
    assertThat(nonLeaves).isEmpty();
  }

  @Test
  public void visitDirectDeps_mixedLeavesAndNonLeaves_visitsBothInSinglePass() {
    Artifact a = ActionsTestUtil.createArtifact(artifactRoot, "a");
    Artifact b = ActionsTestUtil.createArtifact(artifactRoot, "b");
    Artifact c = ActionsTestUtil.createArtifact(artifactRoot, "c");

    NestedSet<Artifact> childSet = NestedSetBuilder.create(Order.STABLE_ORDER, b, c);
    NestedSet<Artifact> parentSet =
        NestedSet.<Artifact>builder(Order.STABLE_ORDER).add(a).addTransitive(childSet).build();

    List<Artifact> leaves = new ArrayList<>();
    List<SkyKey> nonLeaves = new ArrayList<>();
    ArtifactNestedSetKey.visitDirectDeps(parentSet, leaves::add, nonLeaves::add);

    assertThat(leaves).containsExactly(a);
    assertThat(nonLeaves).containsExactly(ArtifactNestedSetKey.create(childSet));
  }
}
