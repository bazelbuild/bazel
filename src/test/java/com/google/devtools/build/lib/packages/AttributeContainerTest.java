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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link AttributeContainer}.
 */
@RunWith(JUnit4.class)
public class AttributeContainerTest {

  private RuleClass ruleClass;
  private AttributeContainer container;
  private Attribute attribute1;
  private Attribute attribute2;

  @Before
  public final void createAttributeContainer() throws Exception  {
    ruleClass =
        TestRuleClassProvider.getRuleClassProvider().getRuleClassMap().get("testing_dummy_rule");
    attribute1 = ruleClass.getAttributeByName("srcs");
    attribute2 = ruleClass.getAttributeByName("dummyinteger");
    container = new AttributeContainer(ruleClass);
  }

  @Test
  public void testAttributeSettingAndRetrievalByName() throws Exception {
    Object someValue1 = new Object();
    Object someValue2 = new Object();
    container.setAttributeValueByName(attribute1.getName(), someValue1);
    container.setAttributeValueByName(attribute2.getName(), someValue2);
    assertEquals(someValue1, container.getAttr(attribute1.getName()));
    assertEquals(someValue2, container.getAttr(attribute2.getName()));
    assertNull(container.getAttr("nomatch"));
  }

  @Test
  public void testExplicitSpecificationsByName() throws Exception {
    // Name-based setters are automatically considered explicit.
    container.setAttributeValueByName(attribute1.getName(), new Object());
    assertTrue(container.isAttributeValueExplicitlySpecified(attribute1));
    assertFalse(container.isAttributeValueExplicitlySpecified("nomatch"));
  }

  @Test
  public void testExplicitSpecificationsByInstance() throws Exception {
    Object someValue = new Object();
    container.setAttributeValue(attribute1, someValue, true);
    container.setAttributeValue(attribute2, someValue, false);
    assertTrue(container.isAttributeValueExplicitlySpecified(attribute1));
    assertFalse(container.isAttributeValueExplicitlySpecified(attribute2));
  }

  private static Location newLocation() {
    return Location.fromPathAndStartColumn(null, 0, 0, new LineAndColumn(0, 0));
  }

  @Test
  public void testAttributeLocation() throws Exception {
    Location location1 = newLocation();
    Location location2 = newLocation();
    container.setAttributeLocation(attribute1, location1);
    container.setAttributeLocation(attribute2, location2);
    assertEquals(location1, container.getAttributeLocation(attribute1.getName()));
    assertEquals(location2, container.getAttributeLocation(attribute2.getName()));
    assertNull(container.getAttributeLocation("nomatch"));
  }

  @Test
  public void testPackedState() throws Exception {
    Random rng = new Random();
    // The state packing machinery has special behavior at multiples of 8,
    // so set enough explicit values and locations to exercise that.
    final int N = 17;
    Attribute[] attributes = new Attribute[N];
    for (int i = 0; i < N; ++i) {
      attributes[i] = ruleClass.getAttribute(i);
    }
    Object someValue = new Object();
    Location[] locations = new Location[N];
    for (int i = 0; i < N; ++i) {
      locations[i] = newLocation();
    }
    assertTrue(locations[0] != locations[1]);  // test relies on checking reference inequality
    for (int explicitCount = 0; explicitCount <= N; ++explicitCount) {
      for (int locationCount = 0; locationCount <= N; ++locationCount) {
        AttributeContainer container = new AttributeContainer(ruleClass);
        // Shuffle the attributes each time through, to exercise
        // different stored indices and orderings.
        Collections.shuffle(Arrays.asList(attributes));
        // Also randomly interleave calls to the two setters.
        int valuePassKey = rng.nextInt(1 << N);
        int locationPassKey = rng.nextInt(1 << N);
        for (int pass = 0; pass <= 1; ++pass) {
          for (int i = 0; i < explicitCount; ++i) {
            if (pass == ((valuePassKey >> i) & 1)) {
              container.setAttributeValue(attributes[i], someValue, true);
            }
          }
          for (int i = 0; i < locationCount; ++i) {
            if (pass == ((locationPassKey >> i) & 1)) {
              container.setAttributeLocation(attributes[i], locations[i]);
            }
          }
        }
        for (int i = 0; i < N; ++i) {
          boolean expected = i < explicitCount;
          assertEquals(expected, container.isAttributeValueExplicitlySpecified(attributes[i]));
        }
        for (int i = 0; i < N; ++i) {
          Location expected = i < locationCount ? locations[i] : null;
          assertSame(expected, container.getAttributeLocation(attributes[i].getName()));
        }
      }
    }
  }
}
