// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.packages.AttributeContainer.Mutable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link AttributeContainer}.
 */
@RunWith(JUnit4.class)
public class AttributeContainerTest {

  private static final int ATTR1 = 2;
  private static final int ATTR2 = 6;

  @Test
  public void testAttributeSettingAndRetrieval() {
    AttributeContainer container = new Mutable((short) 10);
    Object someValue1 = new Object();
    Object someValue2 = new Object();
    container.setAttributeValue(ATTR1, someValue1, /*explicit=*/ true);
    container.setAttributeValue(ATTR2, someValue2, /*explicit=*/ true);
    assertThat(container.getAttributeValue(ATTR1)).isEqualTo(someValue1);
    assertThat(container.getAttributeValue(ATTR2)).isEqualTo(someValue2);
    assertThrows(IndexOutOfBoundsException.class, () -> container.getAttributeValue(10));
  }

  @Test
  public void testAttributeSettingAndRetrieval_afterFreezing() {
    AttributeContainer container = new Mutable((short) 10);
    Object someValue1 = new Object();
    Object someValue2 = new Object();
    container.setAttributeValue(ATTR1, someValue1, /*explicit=*/ true);
    container.setAttributeValue(ATTR2, someValue2, /*explicit=*/ true);
    AttributeContainer frozen = container.freeze();
    assertThat(frozen.getAttributeValue(ATTR1)).isEqualTo(someValue1);
    assertThat(frozen.getAttributeValue(ATTR2)).isEqualTo(someValue2);
    assertThrows(IndexOutOfBoundsException.class, () -> container.getAttributeValue(10));
  }

  @Test
  public void testExplicitSpecificationsByInstance() {
    AttributeContainer container = new Mutable((short) 10);
    Object someValue = new Object();
    container.setAttributeValue(ATTR1, someValue, true);
    container.setAttributeValue(ATTR2, someValue, false);
    assertThat(container.isAttributeValueExplicitlySpecified(ATTR1)).isTrue();
    assertThat(container.isAttributeValueExplicitlySpecified(ATTR2)).isFalse();
  }

  @Test
  public void testExplicitStateForOffByOneError() {
    AttributeContainer container = new Mutable((short) 30);
    // Set index 3 explicitly and check neighbouring indices dont leak that.
    Object valA = new Object();
    Object valB = new Object();
    Object valC = new Object();
    container.setAttributeValue(2, valA, true);
    container.setAttributeValue(3, valB, true);
    container.setAttributeValue(4, valC, false);
    assertThat(container.isAttributeValueExplicitlySpecified(2)).isTrue();
    assertThat(container.isAttributeValueExplicitlySpecified(3)).isTrue();
    assertThat(container.isAttributeValueExplicitlySpecified(4)).isFalse();
  }

  @Test
  public void testPackedState() throws Exception {
    Random rng = new Random();
    // The state packing machinery has special behavior at multiples of 8,
    // so set enough explicit values to exercise that.
    final int numAttributes = 17;
    List<Integer> attrIndices = new ArrayList<>();
    for (int attrIndex = 0; attrIndex < numAttributes; ++attrIndex) {
      attrIndices.add(attrIndex);
    }

    Object someValue = new Object();
    for (int explicitCount = 0; explicitCount <= numAttributes; ++explicitCount) {
      AttributeContainer container = new Mutable((short) 20);
      // Shuffle the attributes each time through, to exercise
      // different stored indices and orderings.
      Collections.shuffle(attrIndices);
        // Also randomly interleave calls to the two setters.
        int valuePassKey = rng.nextInt(1 << numAttributes);
        for (int pass = 0; pass <= 1; ++pass) {
          for (int i = 0; i < explicitCount; ++i) {
            if (pass == ((valuePassKey >> i) & 1)) {
            container.setAttributeValue(i, someValue, true);
            }
          }
        }

        for (int i = 0; i < numAttributes; ++i) {
          boolean expected = i < explicitCount;
        assertThat(container.isAttributeValueExplicitlySpecified(i)).isEqualTo(expected);
        }
    }
  }

  private void checkFreezeWorks(
      short maxAttrCount, Class<? extends AttributeContainer> expectedImplClass) {
    AttributeContainer container = new Mutable(maxAttrCount);
    Object someValue1 = new Object();
    Object someValue2 = new Object();
    container.setAttributeValue(ATTR1, someValue1, /*explicit=*/ true);
    container.setAttributeValue(ATTR2, someValue2, /*explicit=*/ false);
    AttributeContainer frozen = container.freeze();
    assertThat(frozen).isInstanceOf(expectedImplClass);
    // freezing returned something else.
    assertThat(frozen).isNotSameInstanceAs(container);
    // Double freezing is a no-op
    assertThat(frozen.freeze()).isSameInstanceAs(frozen);
    // reads/explicit bits work as expected
    assertThat(frozen.getAttributeValue(ATTR1)).isEqualTo(someValue1);
    assertThat(frozen.isAttributeValueExplicitlySpecified(ATTR1)).isTrue();
    assertThat(frozen.getAttributeValue(ATTR2)).isEqualTo(someValue2);
    assertThat(frozen.isAttributeValueExplicitlySpecified(ATTR2)).isFalse();
    // Invalid attribute index.
    assertThrows(IndexOutOfBoundsException.class, () -> frozen.getAttributeValue(maxAttrCount));
    // writes no longer work.
    assertThrows(
        UnsupportedOperationException.class,
        () -> frozen.setAttributeValue(ATTR2, new Object(), true));
    // Updates to the original container no longer reflected in new container.
    Object newValue = new Object();
    container.setAttributeValue(ATTR2, newValue, true);
    assertThat(container.getAttributeValue(ATTR2)).isEqualTo(newValue);
    assertThat(frozen.getAttributeValue(ATTR2)).isEqualTo(someValue2);
  }

  @Test
  public void testFreezeWorks_smallImplementation() {
    checkFreezeWorks((short) 20, AttributeContainer.Small.class);
  }

  @Test
  public void testFreezeWorks_largeImplementation() {
    checkFreezeWorks((short) 150, AttributeContainer.Large.class);
  }

  private void testContainerSize(int size) {
    AttributeContainer container = new Mutable(size);
    for (int i = 0; i < size; i++) {
      container.setAttributeValue(i, "value " + i, i % 2 == 0);
    }
    AttributeContainer frozen = container.freeze();
    // Check values.
    for (int i = 0; i < size; i++) {
      assertThat(frozen.getAttributeValue(i)).isEqualTo("value " + i);
      assertThat(frozen.isAttributeValueExplicitlySpecified(i)).isEqualTo(i % 2 == 0);
    }
  }

  @Test
  public void testSmallContainer() {
    // At 127 attributes, we shift from AttributeContainer.Small to AttributeContainer.Large.
    testContainerSize(126);
  }

  @Test
  public void testLargeContainer() {
    // AttributeContainer.Large can handle at max 254 attributes.
    testContainerSize(254);
  }

  @Test
  public void testMutableGetRawAttributeValuesReturnsNullSafeCopy() {
    AttributeContainer container = new Mutable(1);
    assertThat(container.getRawAttributeValues()).containsExactly((Object) null);

    container.getRawAttributeValues().set(0, "foo");
    assertThat(container.getRawAttributeValues()).containsExactly((Object) null);
  }

  @Test
  public void testGetRawAttributeValuesReturnsCopy() {
    AttributeContainer mutable = new Mutable(2);
    mutable.setAttributeValue(0, "hi", /*explicit=*/ true);
    mutable.setAttributeValue(1, null, /*explicit=*/ false);

    AttributeContainer container = mutable.freeze();
    // Nulls don't make it into the frozen representation.
    assertThat(container.getRawAttributeValues()).containsExactly("hi");

    container.getRawAttributeValues().set(0, "foo");
    assertThat(container.getRawAttributeValues()).containsExactly("hi");
  }
}
