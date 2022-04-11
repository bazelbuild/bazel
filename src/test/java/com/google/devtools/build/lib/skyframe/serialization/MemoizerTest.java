// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import java.io.IOException;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests memo-based encoding and decoding, especially for cyclic data structures. */
@RunWith(JUnit4.class)
public class MemoizerTest {

  // These classes are used to model a potentially cyclic data structure with both mutable and
  // immutable components.

  private interface DummyLinkedList {

    String getValue();

    @Nullable
    DummyLinkedList getNext();
  }

  private static class MutableDummy implements DummyLinkedList {

    private final String value;

    @Nullable private DummyLinkedList next;

    MutableDummy(String value, @Nullable DummyLinkedList next) {
      this.value = value;
      this.next = next;
    }

    @Override
    public String getValue() {
      return value;
    }

    @Override
    @Nullable
    public DummyLinkedList getNext() {
      return next;
    }

    void setNext(@Nullable DummyLinkedList next) {
      this.next = next;
    }
  }

  private static class ImmutableDummy implements DummyLinkedList {

    private final String value;

    @Nullable private final DummyLinkedList next;

    ImmutableDummy(String value, @Nullable DummyLinkedList next) {
      this.value = value;
      this.next = next;
    }

    @Override
    public String getValue() {
      return value;
    }

    @Override
    @Nullable
    public DummyLinkedList getNext() {
      return next;
    }
  }

  @Test
  public void chainOfMutables() throws IOException, SerializationException {
    DummyLinkedList c = new MutableDummy("C", null);
    DummyLinkedList b = new MutableDummy("B", c);
    DummyLinkedList a = new MutableDummy("A", b);
    assertABC(TestUtils.roundTripMemoized(a));
  }

  @Test
  public void chainOfMixed() throws IOException, SerializationException {
    DummyLinkedList c = new MutableDummy("C", null);
    DummyLinkedList b = new ImmutableDummy("B", c);
    DummyLinkedList a = new MutableDummy("A", b);
    assertABC(TestUtils.roundTripMemoized(a));
  }

  @Test
  public void cycleOfMutables() throws IOException, SerializationException {
    MutableDummy b = new MutableDummy("B", null);
    DummyLinkedList a = new MutableDummy("A", b);
    b.setNext(a);
    assertABcycle(TestUtils.roundTripMemoized(a));
  }

  @Test
  public void cycleOfMixedWithMutableRoot() throws IOException, SerializationException {
    MutableDummy a = new MutableDummy("A", null);
    DummyLinkedList b = new ImmutableDummy("B", a);
    a.setNext(b);
    assertABcycle(TestUtils.roundTripMemoized(a));
  }

  @Test
  public void cycleOfMixedWithImmutableRoot() throws IOException, SerializationException {
    MutableDummy b = new MutableDummy("B", null);
    DummyLinkedList a = new ImmutableDummy("A", b);
    b.setNext(a);
    assertABcycle(TestUtils.roundTripMemoized(a));
  }

  /** Asserts that {@code value} has the linked list structure {@code A -> B -> C}. */
  private static void assertABC(DummyLinkedList value) {
    assertThat(value.getValue()).isEqualTo("A");
    assertThat(value.getNext()).isNotNull();
    assertThat(value.getNext().getValue()).isEqualTo("B");
    assertThat(value.getNext().getNext()).isNotNull();
    assertThat(value.getNext().getNext().getValue()).isEqualTo("C");
    assertThat(value.getNext().getNext().getNext()).isNull();
  }

  /** Asserts that {@code value} has the cyclic linked list structure {@code A -> B -> A...}. */
  private static void assertABcycle(DummyLinkedList value) {
    assertThat(value.getValue()).isEqualTo("A");
    assertThat(value.getNext()).isNotNull();
    assertThat(value.getNext().getValue()).isEqualTo("B");
    assertThat(value.getNext().getNext()).isNotNull();
    // Check instance identity to ensure we reproduced the object graph without creating duplicates.
    assertThat(value.getNext().getNext()).isSameInstanceAs(value);
  }
}
