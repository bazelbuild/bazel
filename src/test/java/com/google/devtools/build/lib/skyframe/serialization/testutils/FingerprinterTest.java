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
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.computeVisitOrder;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Fingerprinter.computeFingerprints;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Fingerprinter.fingerprintString;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import java.lang.ref.WeakReference;
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
    String contents = "contents";
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
            new WeakReference<Object>(contents),
            ImmutableList.of("x"));
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
                    + " nonInlinedValue=82ed5f9ce61ef24fba1beda7a918eb90,"
                    + " shortValue=java.lang.Short:12345, stringValue=java.lang.String:text,"
                    + " weakReferenceValue=java.lang.ref.WeakReference]"),
            subject.nonInlinedValue,
            fingerprintString(
                "com.google.common.collect.SingletonImmutableList: [java.lang.String:x]"));
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
    private final WeakReference<Object> weakReferenceValue;

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
        WeakReference<Object> weakReferenceValue,
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
      this.weakReferenceValue = weakReferenceValue;
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
  public void selfReferenceArray_fingerprintSucceeds() {
    Object[] subject = new Object[1];
    subject[0] = subject;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    var dumpOut = new StringBuilder();
    var unused =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, subject, dumpOut);
    String fingerprint = fingerprintString(dumpOut.toString());
    assertThat(fingerprints).containsExactly(subject, fingerprint + "[0]");
  }

  @Test
  public void symmetricalArrays_emitsRelativeFingerprintOnly() {
    Object[] subject = new Object[1];
    Object[] reflection = new Object[] {subject};
    subject[0] = reflection;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(subject);
    var dumpOut = new StringBuilder();
    var unused =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, subject, dumpOut);
    String fingerprint = fingerprintString(dumpOut.toString());
    // Contains only the "relative" fingerprint for subject and no fingerprint for reflection.
    assertThat(fingerprints).containsExactly(subject, fingerprint);
  }

  @Test
  public void distinctLocalFingerprintArrayCycle_fullyFingerprints() {
    var pointA = new Position(0, 1);
    var pointB = new Position(1, 2);

    // Constructs a cycle, a1 - > a2 -> b -> a1.
    // (a1, a2) will have the same local fingerprint, but b1's local fingerprint is unique.
    Object[] a1 = new Object[2];
    Object[] a2 = new Object[2];
    Object[] b = new Object[2];

    a1[0] = pointA;
    a1[1] = a2;
    a2[0] = pointA;
    a2[1] = b;

    b[0] = pointB;
    b[1] = a1;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(a1);

    var leafFingerprints = new IdentityHashMap<Object, String>();
    leafFingerprints.put(pointA, fingerprints.get(pointA));
    leafFingerprints.put(pointB, fingerprints.get(pointB));
    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> visitOrder =
        computeVisitOrder(/* registry= */ null, leafFingerprints, b, dumpOut);
    String fingerprint = fingerprintString(dumpOut.toString());

    var expected = new IdentityHashMap<Object, String>();
    visitOrder.forEach((obj, index) -> expected.put(obj, fingerprint + '[' + index + ']'));
    leafFingerprints.forEach(expected::put);
    assertThat(fingerprints).isEqualTo(expected);

    // Verifies that fingerprints are independent of starting node.
    for (Object[] root : ImmutableList.of(a2, b)) {
      assertThat(computeFingerprints(root)).isEqualTo(fingerprints);
    }
  }

  @Test
  public void localFingerprintIndistinguishableArrayCycle_fulllyFingerprints() {
    var pointA = new Position(0, 1);
    var pointB = new Position(1, 2);

    // Constructs a cycle, a1 - > a2 -> a3 -> b1 -> b2 -> a1.
    // (a1, a2, a3) and (b1, b2) have the same local fingerprints so local fingerprinting is not
    // enough to resolve this cycle. However, the cycle is slightly asymmetrical so full
    // fingerprints are distinct.
    Object[] a1 = new Object[2];
    Object[] a2 = new Object[2];
    Object[] a3 = new Object[2];
    Object[] b1 = new Object[2];
    Object[] b2 = new Object[2];

    a1[0] = pointA;
    a2[0] = pointA;
    a3[0] = pointA;
    b1[0] = pointB;
    b2[0] = pointB;

    a1[1] = a2;
    a2[1] = a3;
    a3[1] = b1;
    b1[1] = b2;
    b2[1] = a1;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(a1);
    var leafFingerprints = new IdentityHashMap<Object, String>();
    leafFingerprints.put(pointA, fingerprints.get(pointA));
    leafFingerprints.put(pointB, fingerprints.get(pointB));

    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> b1VisitOrder =
        computeVisitOrder(/* registry= */ null, leafFingerprints, b1, dumpOut);
    String b1Fingerprint = fingerprintString(dumpOut.toString());

    dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> b2VisitOrder =
        computeVisitOrder(/* registry= */ null, leafFingerprints, b2, dumpOut);
    String b2Fingerprint = fingerprintString(dumpOut.toString());

    assertThat(b1Fingerprint).isNotEqualTo(b2Fingerprint);

    IdentityHashMap<Object, Integer> visitOrder;
    String fingerprint;
    if (b1Fingerprint.compareTo(b2Fingerprint) < 0) {
      visitOrder = b1VisitOrder;
      fingerprint = b1Fingerprint;
    } else {
      visitOrder = b2VisitOrder;
      fingerprint = b2Fingerprint;
    }

    var expected = new IdentityHashMap<Object, String>();
    visitOrder.forEach((obj, index) -> expected.put(obj, fingerprint + '[' + index + ']'));
    leafFingerprints.forEach(expected::put);
    assertThat(fingerprints).isEqualTo(expected);

    // Verifies that fingerprints are independent of starting node.
    for (Object[] root : ImmutableList.of(a2, a3, b1, b2)) {
      assertThat(computeFingerprints(root)).isEqualTo(fingerprints);
    }
  }

  @Test
  public void overlySymmetricalArrayCycle_emitsRelativeFingerprintOnly() {
    var pointA = new Position(0, 1);
    var pointB = new Position(1, 2);

    // Constructs the cycle a1 -> b1 -> a2 -> b2 -> a1.
    // This cycle is perfectly symmetrical so can't even be resolved by full fingerprinting.
    Object[] a1 = new Object[2];
    Object[] a2 = new Object[2];
    Object[] b1 = new Object[2];
    Object[] b2 = new Object[2];

    a1[0] = pointA;
    a2[0] = pointA;
    b1[0] = pointB;
    b2[0] = pointB;

    a1[1] = b1;
    b1[1] = a2;
    a2[1] = b2;
    b2[1] = a1;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(a1);

    var leafFingerprints = new IdentityHashMap<Object, String>();
    leafFingerprints.put(pointA, fingerprints.get(pointA));
    leafFingerprints.put(pointB, fingerprints.get(pointB));

    var dumpOut = new StringBuilder();
    var unused = computeVisitOrder(/* registry= */ null, leafFingerprints, a1, dumpOut);
    String fingerprint = fingerprintString(dumpOut.toString());

    var expected = new IdentityHashMap<Object, String>();
    // Only pointA, pointB and a1 have fingerprints.
    expected.put(a1, fingerprint);
    leafFingerprints.forEach(expected::put);
    assertThat(fingerprints).isEqualTo(expected);
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
  public void symmetricalMapCycle_emitsRelativeFingerprintOnly() {
    // Constructs the cycle a1 -> b1 -> a2 -> b2 -> a1.
    // This cycle is perfectly symmetrical so can't even be resolved by full fingerprinting.
    var a1 = new LinkedHashMap<String, Object>();
    var b1 = new LinkedHashMap<String, Object>();
    var a2 = new LinkedHashMap<String, Object>();
    var b2 = new LinkedHashMap<String, Object>();

    a1.put("A", b1);
    b1.put("B", a2);
    a2.put("A", b2);
    b2.put("B", a1);

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(a1);

    var dumpOut = new StringBuilder();
    var unused = computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, a1, dumpOut);
    String fingerprint = fingerprintString(dumpOut.toString());

    var expected = new IdentityHashMap<>();
    expected.put(a1, fingerprint);
    assertThat(fingerprints).isEqualTo(expected);
  }

  @Test
  @SuppressWarnings("ContainsEntryAfterGetString") // Causes test failure: b/381569717
  public void collection_fingerprints() {
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
    String positionFingerprint = fingerprintString(NAMESPACE + "Position: [x=12, y=24]");
    assertThat(fingerprints.get(position)).isEqualTo(positionFingerprint);

    String subjectFingerprint =
        fingerprintString(
                String.format(
                    """
java.util.ArrayList(0) [
  abc
  10
  %s[%s]
  java.util.ArrayList(0)
  null
]\
""",
                    NAMESPACE + "Position", positionFingerprint))
            + "[0]";
    assertThat(fingerprints.get(subject)).isEqualTo(subjectFingerprint);
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
  public void singleReferenceConstant() {
    String subject = "constant";
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder().addReferenceConstant(subject).build();
    assertThat(Fingerprinter.computeFingerprint(registry, subject))
        .isEqualTo("java.lang.String[SERIALIZATION_CONSTANT:1]");
  }

  @Test
  public void singleReferenceConstant_defaultRegistry() {
    String subject = "constant";
    assertThat(Fingerprinter.computeFingerprint(ObjectCodecRegistry.newBuilder().build(), subject))
        .isEqualTo("java.lang.String:constant");
  }

  @Test
  public void singleReferenceConstant_nullRegistry() {
    String subject = "constant";
    assertThat(Fingerprinter.computeFingerprint(/* registry= */ null, subject))
        .isEqualTo("java.lang.String:constant");
  }

  @Test
  public void multipleReferenceConstants() {
    String constant1 = "constant1";
    Integer constant2 = 256;
    ObjectCodecRegistry registry =
        ObjectCodecRegistry.newBuilder()
            .addReferenceConstant(constant1)
            .addReferenceConstant(constant2)
            .build();
    var subject = ImmutableList.of(constant1, "a", constant2, constant1);
    IdentityHashMap<Object, String> fingerprints = computeFingerprints(registry, subject);
    assertThat(fingerprints)
        .containsExactly(
            subject,
            fingerprintString(
                "com.google.common.collect.RegularImmutableList:"
                    + " [java.lang.String[SERIALIZATION_CONSTANT:1], java.lang.String:a,"
                    + " java.lang.Integer[SERIALIZATION_CONSTANT:2],"
                    + " java.lang.String[SERIALIZATION_CONSTANT:1]]"));
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

    // `one` ends up being identified as the root (by lexicographical partial fingerprint).
    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> visitOrder =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, one, dumpOut);
    String rootFingerprint = fingerprintString(dumpOut.toString());

    assertThat(fingerprints)
        .isEqualTo(Maps.transformValues(visitOrder, index -> rootFingerprint + '[' + index + ']'));

    // Verfiies that all rotations of the cyclic complex result in the same fingerprints.
    for (Node rotated : new Node[] {one, two, three, four}) {
      assertThat(computeFingerprints(rotated)).isEqualTo(fingerprints);
    }
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

    // `four` ends up being identified as root of the child loop.
    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> childVisitOrder =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, four, dumpOut);
    String childRootFingerprint = fingerprintString(dumpOut.toString());
    var expectedFingerprints = new IdentityHashMap<Object, String>();
    childVisitOrder.forEach(
        (obj, index) -> expectedFingerprints.put(obj, childRootFingerprint + '[' + index + ']'));

    // `one` ends up being identified as the root of the parent loop.
    dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> parentVisitOrder =
        computeVisitOrder(/* registry= */ null, expectedFingerprints, one, dumpOut);
    String parentRootFingerprint = fingerprintString(dumpOut.toString());
    parentVisitOrder.forEach(
        (obj, index) -> expectedFingerprints.put(obj, parentRootFingerprint + '[' + index + ']'));

    assertThat(fingerprints).isEqualTo(expectedFingerprints);
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
    // (left) 0 -> 1 -> 2 -> 1
    zero.left = one;
    one.left = two;
    two.left = one;
    // (right) 0 -> 3 -> 4 -> 5 -> 4
    zero.right = three;
    three.left = four;
    four.left = five;
    five.left = four;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(zero);

    var expectedFingerprints = new IdentityHashMap<Object, String>();

    // `two` ends up being identified as root of the left loop.
    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> leftVisitOrder =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, two, dumpOut);
    String leftRootFingerprint = fingerprintString(dumpOut.toString());
    leftVisitOrder.forEach(
        (obj, index) -> expectedFingerprints.put(obj, leftRootFingerprint + '[' + index + ']'));

    // `four` ends up being identified as the root of the right loop.
    dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> rightVisitOrder =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, four, dumpOut);
    String rightRootFingerprint = fingerprintString(dumpOut.toString());
    rightVisitOrder.forEach(
        (obj, index) -> expectedFingerprints.put(obj, rightRootFingerprint + '[' + index + ']'));

    String fingerprint3 = fingerprintNode(3, expectedFingerprints.get(four));
    String fingerprint0 = fingerprintNode(0, expectedFingerprints.get(one), fingerprint3);

    expectedFingerprints.put(zero, fingerprint0);
    expectedFingerprints.put(three, fingerprint3);

    assertThat(fingerprints).isEqualTo(expectedFingerprints);
  }

  @Test
  public void unlabeledCyclicComplex_fingerprints() {
    // The nodes all have the same label to prevent prevent local fingerprinting. This exercises
    // the full, relative fingerprinting fallback.
    Node zero = new Node(0);
    Node one = new Node(0);
    Node two = new Node(0);
    Node three = new Node(0);
    Node four = new Node(0);

    // The example consists of a somewhat complex graph.
    //    0
    //  ↗ ↓ ↖
    //  | 1  \
    //  |↙ ↘  \
    //  2   3  |
    //   ↖  ↓ /
    //      4
    // Note that (1, 4) and (0, 2, 3) are indistinguishable by local fingerprint.
    zero.left = one;
    one.left = two;
    two.left = zero;
    one.right = three;
    three.left = four;
    four.left = two;
    four.right = zero;

    IdentityHashMap<Object, String> fingerprints = computeFingerprints(zero);

    // The group (1, 4) has lower cardinality, so it's examined first. 1 has a lexicographically
    // earlier fingerprint than 4 so it becomes the canonical root.
    var dumpOut = new StringBuilder();
    IdentityHashMap<Object, Integer> visitOrder =
        computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, one, dumpOut);
    String oneFingerprint = fingerprintString(dumpOut.toString());

    dumpOut = new StringBuilder();
    var unused = computeVisitOrder(/* registry= */ null, /* fingerprints= */ null, four, dumpOut);
    String fourFingerprint = fingerprintString(dumpOut.toString());

    assertThat(oneFingerprint).isLessThan(fourFingerprint);

    // Verifies that the fingerprints are generated by 1's fingerprint and visitation order.
    assertThat(fingerprints)
        .isEqualTo(Maps.transformValues(visitOrder, index -> oneFingerprint + '[' + index + ']'));

    // Verfies all starting points produce the identical fingerprint maps.
    for (Node rotated : new Node[] {one, two, three, four}) {
      assertThat(computeFingerprints(rotated)).isEqualTo(fingerprints);
    }
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
  private record Position(int x, int y) {}
}
