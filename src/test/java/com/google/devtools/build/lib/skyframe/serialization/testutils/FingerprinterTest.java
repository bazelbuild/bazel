// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Fingerprinter.computeFingerprints;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Fingerprinter.fingerprintString;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FingerprinterTest {
  // This string is very long:
  // com.google.devtools.build.lib.skyframe.serialization.testutils.FingerprinterTest.
  // and ruins the spacing of assertions.
  private static final String NAMESPACE = FingerprinterTest.class.getCanonicalName() + ".";

  @Test
  public void inlinedValues_leaveNoFingerprints() {
    var subject =
        new TypeWithInlinedData(
            (byte) 0x10,
            (short) 12345,
            65536,
            4294967296L,
            0.01f,
            12.123456789,
            true,
            'c',
            "text",
            ImmutableList.of());
    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    // There's one entry for `subject`, and one for the `nonInlinedValue` field. None of the inlined
    // fields have fingerprint entries.
    assertThat(fingerprints)
        .containsExactly(
            subject,
            fingerprintString(
                NAMESPACE
                    + "TypeWithInlinedData:"
                    + " [boolValue=java.lang.Boolean:true, byteValue=java.lang.Byte:16,"
                    + " charValue=java.lang.Character:c, doubleValue=java.lang.Double:12.123456789,"
                    + " floatValue=java.lang.Float:0.01, intValue=java.lang.Integer:65536,"
                    + " longValue=java.lang.Long:4294967296,"
                    + " nonInlinedValue=7cf245bfdbbfb29e4da6143f7b67ac98,"
                    + " shortValue=java.lang.Short:12345, stringValue=java.lang.String:text]"),
            subject.nonInlinedValue,
            fingerprintString("com.google.common.collect.RegularImmutableList: []"));
  }

  @SuppressWarnings({"UnusedVariable", "FieldCanBeLocal"})
  private static final class TypeWithInlinedData {
    private final Byte byteValue;
    private final Short shortValue;
    private final Integer intValue;
    private final Long longValue;
    private final Float floatValue;
    private final Double doubleValue;
    private final Boolean boolValue;
    private final Character charValue;
    private final String stringValue;

    /** A non-inlined value for contrast. */
    private final ImmutableList<Object> nonInlinedValue;

    private TypeWithInlinedData(
        byte byteValue,
        short shortValue,
        int intValue,
        long longValue,
        float floatValue,
        double doubleValue,
        boolean boolValue,
        char charValue,
        String stringValue,
        ImmutableList<Object> nonInlinedValue) {
      this.byteValue = byteValue;
      this.shortValue = shortValue;
      this.intValue = intValue;
      this.longValue = longValue;
      this.floatValue = floatValue;
      this.doubleValue = doubleValue;
      this.boolValue = boolValue;
      this.charValue = charValue;
      this.stringValue = stringValue;
      this.nonInlinedValue = nonInlinedValue;
    }
  }

  @Test
  public void specialCaseArrays() {
    var subject =
        new TypeWithSpecialArrays(
            new byte[] {(byte) 0xDE, (byte) 0xAD, (byte) 0xBE, (byte) 0xEF},
            new String[] {"abc", "def", "hij"});
    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    assertThat(fingerprints)
        .containsExactly(
            subject,
            fingerprintString(
                NAMESPACE
                    + "TypeWithSpecialArrays: [byteArray=63deb80b9d484a1af76057b22fa1f403,"
                    + " stringArray=9efba83af9eba70ede40159aa606209c]"),
            // byte[] inlines into hex.
            subject.byteArray,
            fingerprintString("byte[]: [DEADBEEF]"),
            // String[] values inline.
            subject.stringArray,
            fingerprintString(
                "java.lang.String[]: [java.lang.String:abc, java.lang.String:def,"
                    + " java.lang.String:hij]"));
  }

  private static final class TypeWithSpecialArrays {
    private final byte[] byteArray;
    private final String[] stringArray;

    private TypeWithSpecialArrays(byte[] byteArray, String[] stringArray) {
      this.byteArray = byteArray;
      this.stringArray = stringArray;
    }
  }

  @Test
  public void map() {
    var subject = new LinkedHashMap<Object, Object>();
    subject.put("abc", "def");
    subject.put(10, 12.5);
    subject.put(null, 'c');
    subject.put("k1", null);
    var position = new Position(5, 10);
    subject.put(position, "Value");

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    assertThat(fingerprints)
        .containsExactly(
            subject,
                fingerprintString(
                    "java.util.LinkedHashMap: [key=java.lang.String:abc,"
                        + " value=java.lang.String:def, key=java.lang.Integer:10,"
                        + " value=java.lang.Double:12.5, key=null, value=java.lang.Character:c,"
                        + " key=java.lang.String:k1, value=null,"
                        + " key=39b554d661d8e4c35f6657467f4e492c, value=java.lang.String:Value]"),
            position, fingerprintString(NAMESPACE + "Position: [x=5, y=10]"));
  }

  @Test
  public void iterable() {
    var subject = new ArrayList<Object>();
    subject.add("abc");
    subject.add(10);
    var position = new Position(12, 24);
    subject.add(position);
    subject.add(subject); // Creates a cycle.
    subject.add(null);

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    assertThat(fingerprints).hasSize(2);
    // These checks are element-wise because containsExactly somehow tries to compute the hashCode
    // of `subject`, leading to a StackOverflowError.
    assertThat(fingerprints.get(subject))
        .isEqualTo(
            // The cyclic self-reference at the position 3 is 0, as expected.
            fingerprintString(
                "java.util.ArrayList: [java.lang.String:abc, java.lang.Integer:10,"
                    + " 3a563d247e43e61db06a1e2931834994, 0, null]"));
    assertThat(fingerprints.get(position))
        .isEqualTo(fingerprintString(NAMESPACE + "Position: [x=12, y=24]"));
  }

  @Test
  public void plainObject() {
    var subject = new ExamplePlainObject();
    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    assertThat(fingerprints)
        .containsExactly(
            subject,
            fingerprintString(
                NAMESPACE
                    + "ExamplePlainObject: [booleanValue=false, byteValue=16, charValue=c,"
                    + " classValue=java.lang.Class:interface java.lang.Runnable,"
                    + " doubleValue=12.123456789, floatValue=0.01, intValue=65536,"
                    + " longValue=4294967296, nullClass=null, nullString=null, shortValue=12345,"
                    + " stringValue=java.lang.String:text]"));
  }

  @SuppressWarnings({"UnusedVariable", "FieldCanBeStatic"})
  private static class ExamplePlainObject {
    private final boolean booleanValue = false;
    private final byte byteValue = (byte) 0x10;
    private final short shortValue = (short) 12345;
    private final char charValue = 'c';
    private final int intValue = 65536;
    private final long longValue = 4294967296L;
    private final float floatValue = 0.01f;
    private final double doubleValue = 12.123456789;
    private final String stringValue = "text";
    private final String nullString = null;
    private final Class<?> classValue = Runnable.class;
    private final Class<?> nullClass = null;
  }

  @Test
  public void cyclicComplex() {
    Node zero = new Node(0);
    Node one = new Node(1);
    Node two = new Node(2);
    Node three = new Node(3);
    Node four = new Node(4);

    // The example consists of two overlapping cycles.
    // 0 -> 1 -> 2 -> 0
    zero.left = one;
    one.left = two;
    two.left = zero;
    // 0 -> 1 -> 3 -> 4 -> 0
    one.right = three;
    three.left = four;
    four.left = zero;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(zero);

    // The fingerprints below are all transitively linked to fingerprint0 so an assertion on
    // fingerprint0 actually covers the rest of them as well.
    String fingerprint4 = fingerprintNode(4, "-3");
    String fingerprint3 = fingerprintNode(3, fingerprint4);
    String fingerprint2 = fingerprintNode(2, "-2");
    String fingerprint1 = fingerprintNode(1, fingerprint2, fingerprint3);
    String fingerprint0 = fingerprintNode(0, fingerprint1);

    // All the nodes are part of a common cyclic complex so there is only one fingerprint.
    assertThat(fingerprints).containsExactly(zero, fingerprint0);
  }

  @Test
  public void childCycle() {
    Node zero = new Node(0);
    Node one = new Node(1);
    Node two = new Node(2);
    Node three = new Node(3);
    Node four = new Node(4);
    Node five = new Node(5);

    // The example consists of one cycle hanging off from another one.
    // 0 -> 1 -> 2 -> 0
    zero.left = one;
    one.left = two;
    two.left = zero;
    // 1 -> 3 -> 4 -> 5 -> 3
    one.right = three;
    three.left = four;
    four.left = five;
    five.left = three;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(zero);

    String fingerprint5 = fingerprintNode(5, "-2");
    String fingerprint4 = fingerprintNode(4, fingerprint5);
    String fingerprint3 = fingerprintNode(3, fingerprint4);
    String fingerprint2 = fingerprintNode(2, "-2");
    String fingerprint1 = fingerprintNode(1, fingerprint2, fingerprint3);
    String fingerprint0 = fingerprintNode(0, fingerprint1);

    // There are two independent cycles, so there are two fingerprints.
    assertThat(fingerprints)
        .containsExactly(
            zero, fingerprint0,
            three, fingerprint3);
  }

  @Test
  public void peerCycles() {
    Node zero = new Node(0);
    Node one = new Node(1);
    Node two = new Node(2);
    Node three = new Node(3);
    Node four = new Node(4);
    Node five = new Node(5);

    // The example consists of two cycles hanging off a common root, 0. The left cycle has a higher
    // absolute backreference level than the right cycle. They should not interfere with one
    // another.
    // 0 -> 1 -> 2 -> 1
    zero.left = one;
    one.left = two;
    two.left = one;
    // 0 -> 3 -> 4 -> 5 -> 4
    zero.right = three;
    three.left = four;
    four.left = five;
    five.left = four;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(zero);

    String fingerprint5 = fingerprintNode(5, "-1");
    String fingerprint4 = fingerprintNode(4, fingerprint5);
    String fingerprint3 = fingerprintNode(3, fingerprint4);
    String fingerprint2 = fingerprintNode(2, "-1");
    String fingerprint1 = fingerprintNode(1, fingerprint2);
    String fingerprint0 = fingerprintNode(0, fingerprint1, fingerprint3);

    // The fingerprints for 2 and 5 are suppressed because they are part of the cycles owned by
    // 1 and 4, respectively.
    assertThat(fingerprints)
        .containsExactly(
            zero, fingerprint0,
            one, fingerprint1,
            three, fingerprint3,
            four, fingerprint4);
  }

  private static String fingerprintNode(int id, String left, String right) {
    return fingerprintString(
        NAMESPACE + "Node: [id=" + id + ", left=" + left + ", right=" + right + "]");
  }

  private static String fingerprintNode(int id, String left) {
    return fingerprintNode(id, left, "null");
  }

  /** Class for demonstrating reference cycles. */
  @SuppressWarnings("UnusedVariable")
  private static final class Node {
    private final int id;
    @Nullable private Node left;
    @Nullable private Node right;

    private Node(int id) {
      this.id = id;
    }

    @Override
    public String toString() {
      return Integer.toString(id);
    }
  }

  /** An arbitrary class used as test data. */
  @SuppressWarnings({"UnusedVariable", "FieldCanBeLocal"})
  private static class Position {
    private final int x;
    private final int y;

    private Position(int x, int y) {
      this.x = x;
      this.y = y;
    }
  }
}
