// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link DottedVersion}.
 */
@RunWith(JUnit4.class)
public class DottedVersionTest {

  @Test
  public void testCompareTo() throws Exception {
    new ComparatorTester()
        .addEqualityGroup(DottedVersion.fromString("0"), DottedVersion.fromString("0.0.0"))
        .addEqualityGroup(DottedVersion.fromString("0.1"), DottedVersion.fromString("0.01"))
        .addEqualityGroup(DottedVersion.fromString("0.2"), DottedVersion.fromString("0.2.0"))
        .addEqualityGroup(DottedVersion.fromString("0.2.1"))
        .addEqualityGroup(DottedVersion.fromString("1.2alpha"))
        .addEqualityGroup(DottedVersion.fromString("1.2alpha1"))
        .addEqualityGroup(DottedVersion.fromString("1.2alpha2"))
        .addEqualityGroup(DottedVersion.fromString("1.2beta1"))
        .addEqualityGroup(DottedVersion.fromString("1.2beta12"))
        .addEqualityGroup(DottedVersion.fromString("1.2beta12.1"))
        .addEqualityGroup(DottedVersion.fromString("1.2.0"), DottedVersion.fromString("1.2"))
        .addEqualityGroup(DottedVersion.fromString("1.20"))
        .addEqualityGroup(DottedVersion.fromString("10"), DottedVersion.fromString("10.0"))
        .addEqualityGroup(DottedVersion.fromString("10.0.0.10A255"))
        .addEqualityGroup(DottedVersion.fromString("10.2"))
        .addEqualityGroup(DottedVersion.fromString("10.2.0.10P99q"))
        .addEqualityGroup(
            DottedVersion.fromString("10.10.simulator.internal"), DottedVersion.fromString("10.10"))
        .testCompare();
  }

  @Test
  public void testEquals() throws Exception {
    new EqualsTester()
        .addEqualityGroup(DottedVersion.fromString("0"), DottedVersion.fromString("0.0.0"))
        .addEqualityGroup(DottedVersion.fromString("0.1"), DottedVersion.fromString("0.01"))
        .addEqualityGroup(DottedVersion.fromString("0.2"), DottedVersion.fromString("0.2.0"))
        .addEqualityGroup(DottedVersion.fromString("1.2xy2"), DottedVersion.fromString("1.2xy2"))
        .addEqualityGroup(
            DottedVersion.fromString("10.2.0.10P99q"), DottedVersion.fromString("10.2.0.10P99q0"))
        .addEqualityGroup(
            DottedVersion.fromString("1.2x"),
            DottedVersion.fromString("1.2x0"),
            DottedVersion.fromString("1.2x0.0"))
        .testEquals();
  }

  @Test
  public void testToStringWithComponents() throws Exception {
    DottedVersion dottedVersion = DottedVersion.fromString("42.8");
    assertThat(dottedVersion.toStringWithComponents(1)).isEqualTo("42");
    assertThat(dottedVersion.toStringWithComponents(2)).isEqualTo("42.8");
    assertThat(dottedVersion.toStringWithComponents(3)).isEqualTo("42.8.0");
    assertThat(dottedVersion.toStringWithComponents(4)).isEqualTo("42.8.0.0");
  }

  @Test
  public void testToStringWithComponents_trailingZero() throws Exception {
    DottedVersion dottedVersion = DottedVersion.fromString("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithComponents(1)).isEqualTo("4");
    assertThat(dottedVersion.toStringWithComponents(2)).isEqualTo("4.3alpha3");
    assertThat(dottedVersion.toStringWithComponents(3)).isEqualTo("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithComponents(4)).isEqualTo("4.3alpha3.0.0");
    assertThat(dottedVersion.toStringWithComponents(5)).isEqualTo("4.3alpha3.0.0.0");
  }

  @Test
  public void testToStringWithComponents_zeroComponent() throws Exception {
    DottedVersion zeroComponent = DottedVersion.fromString("0");
    assertThat(zeroComponent.toStringWithComponents(1)).isEqualTo("0");
    assertThat(zeroComponent.toStringWithComponents(2)).isEqualTo("0.0");
    assertThat(zeroComponent.toStringWithComponents(3)).isEqualTo("0.0.0");
  }

  @Test
  public void testToStringWithMinimumComponent() throws Exception {
    DottedVersion dottedVersion = DottedVersion.fromString("42.8");
    assertThat(dottedVersion.toStringWithMinimumComponents(0)).isEqualTo("42.8");
    assertThat(dottedVersion.toStringWithMinimumComponents(1)).isEqualTo("42.8");
    assertThat(dottedVersion.toStringWithMinimumComponents(2)).isEqualTo("42.8");
    assertThat(dottedVersion.toStringWithMinimumComponents(3)).isEqualTo("42.8.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(4)).isEqualTo("42.8.0.0");
  }

  @Test
  public void testToStringWithMinimumComponent_trailingZero() throws Exception {
    DottedVersion dottedVersion = DottedVersion.fromString("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(0)).isEqualTo("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(1)).isEqualTo("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(2)).isEqualTo("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(3)).isEqualTo("4.3alpha3.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(4)).isEqualTo("4.3alpha3.0.0");
    assertThat(dottedVersion.toStringWithMinimumComponents(5)).isEqualTo("4.3alpha3.0.0.0");
  }

  @Test
  public void testToStringWithMinimumComponents_zeroComponent() throws Exception {
    DottedVersion zeroComponent = DottedVersion.fromString("0");
    assertThat(zeroComponent.toStringWithMinimumComponents(0)).isEqualTo("0");
    assertThat(zeroComponent.toStringWithMinimumComponents(1)).isEqualTo("0");
    assertThat(zeroComponent.toStringWithMinimumComponents(2)).isEqualTo("0.0");
    assertThat(zeroComponent.toStringWithMinimumComponents(3)).isEqualTo("0.0.0");
  }

  @Test
  public void testIllegalVersion_noLeadingInteger() throws Exception {
    Throwable expected =
        assertThrows(
            DottedVersion.InvalidDottedVersionException.class, () -> DottedVersion.fromString("a"));
    assertThat(expected).hasMessageThat().contains("a");
  }

  @Test
  public void testIllegalVersion_empty() throws Exception {
    assertThrows(
        DottedVersion.InvalidDottedVersionException.class, () -> DottedVersion.fromString(""));
  }

  @Test
  public void testIllegalVersion_punctuation() throws Exception {
    assertThrows(
        DottedVersion.InvalidDottedVersionException.class, () -> DottedVersion.fromString("2:3"));
  }

  @Test
  public void testIllegalVersion_emptyComponent() throws Exception {
    assertThrows(
        DottedVersion.InvalidDottedVersionException.class, () -> DottedVersion.fromString("1..3"));
  }

  @Test
  public void testIllegalVersion_negativeComponent() throws Exception {
    assertThrows(
        DottedVersion.InvalidDottedVersionException.class, () -> DottedVersion.fromString("1.-1"));
  }
}
