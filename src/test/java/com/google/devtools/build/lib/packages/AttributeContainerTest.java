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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.analysis.util.MockRuleDefaults;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link AttributeContainer}. */
@RunWith(TestParameterInjector.class)
public final class AttributeContainerTest extends BuildViewTestCase {

  private static final String STRING_DEFAULT = Type.STRING.getDefaultValue();
  private static final int COMPUTED_DEFAULT_OFFSET = 1;

  private enum ContainerSize {
    SMALL(16, AttributeContainer.Small.class),
    LARGE(128, AttributeContainer.Large.class);

    private final int numAttrs;
    private final Class<? extends AttributeContainer> expectedFrozenClass;

    ContainerSize(int numAttrs, Class<? extends AttributeContainer> expectedFrozenClass) {
      this.numAttrs = numAttrs;
      this.expectedFrozenClass = expectedFrozenClass;
    }
  }

  @TestParameter private ContainerSize containerSize;

  private Rule rule;
  private int firstCustomAttrIndex;
  private int lastCustomAttrIndex;
  private int computedDefaultIndex;

  private AttributeContainer container;

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    int numDefaultAttrs = MockRuleDefaults.DEFAULT_ATTRIBUTES.size() + 1; // +1 for name.
    int numCustomAttrs = containerSize.numAttrs - numDefaultAttrs;
    MockRule exampleRule =
        () ->
            MockRule.define(
                "example_rule",
                IntStream.range(0, numCustomAttrs)
                    .mapToObj(
                        i -> {
                          var attr = attr("attr" + i, Type.STRING);
                          // Make one of the attributes a computed default.
                          if (i == COMPUTED_DEFAULT_OFFSET) {
                            attr.value(
                                new ComputedDefault() {
                                  @Override
                                  public Object getDefault(AttributeMap rule) {
                                    return "computed";
                                  }
                                });
                          }
                          return attr;
                        })
                    .toArray(Attribute.Builder[]::new));
    var builder = new ConfiguredRuleClassProvider.Builder().addRuleDefinition(exampleRule);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Before
  public void setUpForRule() throws Exception {
    scratch.file("foo/BUILD", "example_rule(name = 'example')");
    rule = (Rule) getTarget("//foo:example");
    firstCustomAttrIndex = rule.getRuleClassObject().getAttributeIndex("attr0");
    lastCustomAttrIndex = rule.getRuleClassObject().getAttributeCount() - 1;
    computedDefaultIndex = firstCustomAttrIndex + COMPUTED_DEFAULT_OFFSET;
    container = AttributeContainer.newMutableInstance(rule.getRuleClassObject());
  }

  @Test
  public void attributeSettingAndRetrieval(@TestParameter boolean frozen) {
    container.setAttributeValue(firstCustomAttrIndex, "val1", /* explicit= */ true);
    container.setAttributeValue(lastCustomAttrIndex, "val2", /* explicit= */ true);

    if (frozen) {
      container = container.freeze(rule);
    }

    assertThat(container.getAttributeValue(firstCustomAttrIndex)).isEqualTo("val1");
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex)).isTrue();
    assertThat(container.getAttributeValue(lastCustomAttrIndex)).isEqualTo("val2");
    assertThat(container.isAttributeValueExplicitlySpecified(lastCustomAttrIndex)).isTrue();
  }

  @Test
  public void indexOutOfBounds_throws(@TestParameter boolean frozen) {
    if (frozen) {
      container = container.freeze(rule);
    }
    assertThrows(
        IndexOutOfBoundsException.class,
        () -> container.getAttributeValue(lastCustomAttrIndex + 1));
  }

  @Test
  public void testForOffByOneError(@TestParameter boolean frozen) {
    // Set an index explicitly and check neighbouring indices don't leak that.
    container.setAttributeValue(firstCustomAttrIndex, "val", true);

    if (frozen) {
      container = container.freeze(rule);
    }

    assertThat(container.getAttributeValue(firstCustomAttrIndex - 1)).isNull();
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex - 1)).isFalse();
    assertThat(container.getAttributeValue(firstCustomAttrIndex + 1)).isNull();
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex + 1)).isFalse();
  }

  @Test
  public void testFreezeWorks() {
    container.setAttributeValue(firstCustomAttrIndex, "val1", /* explicit= */ true);
    container.setAttributeValue(lastCustomAttrIndex, "val2", /* explicit= */ false);
    assertThat(container.isFrozen()).isFalse();

    AttributeContainer frozen = container.freeze(rule);

    assertThat(frozen.isFrozen()).isTrue();
    assertThat(frozen).isInstanceOf(containerSize.expectedFrozenClass);
    // freezing returned something else.
    assertThat(frozen).isNotSameInstanceAs(container);
    // Double freezing is a no-op
    assertThat(frozen.freeze(rule)).isSameInstanceAs(frozen);
    // reads/explicit bits work as expected
    assertThat(frozen.getAttributeValue(firstCustomAttrIndex)).isEqualTo("val1");
    assertThat(frozen.isAttributeValueExplicitlySpecified(firstCustomAttrIndex)).isTrue();
    assertThat(frozen.getAttributeValue(lastCustomAttrIndex)).isEqualTo("val2");
    assertThat(frozen.isAttributeValueExplicitlySpecified(lastCustomAttrIndex)).isFalse();
    // writes no longer work.
    assertThrows(
        UnsupportedOperationException.class,
        () -> frozen.setAttributeValue(lastCustomAttrIndex, "different", true));
    // Updates to the original container no longer reflected in new container.
    container.setAttributeValue(lastCustomAttrIndex, "different", true);
    assertThat(container.getAttributeValue(lastCustomAttrIndex)).isEqualTo("different");
    assertThat(frozen.getAttributeValue(lastCustomAttrIndex)).isEqualTo("val2");
  }

  @Test
  public void fullContainer(@TestParameter boolean frozen) {
    int size = rule.getRuleClassObject().getAttributeCount();
    for (int i = 0; i < size; i++) {
      container.setAttributeValue(i, "value " + i, i % 2 == 0);
    }

    if (frozen) {
      container = container.freeze(rule);
    }

    for (int i = 0; i < size; i++) {
      assertThat(container.getAttributeValue(i)).isEqualTo("value " + i);
      assertWithMessage("attribute " + i)
          .that(container.isAttributeValueExplicitlySpecified(i))
          .isEqualTo(i % 2 == 0);
    }
  }

  @Test
  public void getRawAttributeValues_mutableContainer_returnsNullSafeCopy() {
    List<Object> rawValues = container.getRawAttributeValues();
    List<Object> expected =
        Collections.nCopies(rule.getRuleClassObject().getAttributeCount(), null);
    assertThat(rawValues).isEqualTo(expected);
    container.getRawAttributeValues().set(0, "foo");
    assertThat(container.getRawAttributeValues()).isEqualTo(expected);
  }

  @Test
  public void getRawAttributeValues_frozen_returnsCopyWithoutNulls() {
    container.setAttributeValue(firstCustomAttrIndex, "hi", /* explicit= */ true);
    container.setAttributeValue(lastCustomAttrIndex, null, /* explicit= */ false);

    container = container.freeze(rule);
    assertThat(container.getRawAttributeValues()).containsExactly("hi");

    container.getRawAttributeValues().set(0, "foo");
    assertThat(container.getRawAttributeValues()).containsExactly("hi");
  }

  /** Regression test for b/269593252. */
  @Test
  public void boundaryOfFrozenContainer() {
    container.setAttributeValue(0, "0", /* explicit= */ true);
    container.setAttributeValue(lastCustomAttrIndex, "last", /* explicit= */ true);

    AttributeContainer frozen = container.freeze(rule);

    assertThat(frozen.getAttributeValue(0)).isEqualTo("0");
    assertThat(frozen.isAttributeValueExplicitlySpecified(0)).isTrue();
    assertThat(frozen.getAttributeValue(lastCustomAttrIndex)).isEqualTo("last");
    assertThat(frozen.isAttributeValueExplicitlySpecified(lastCustomAttrIndex)).isTrue();
  }

  @Test
  public void explictDefaultValue_stored(@TestParameter boolean frozen) {
    container.setAttributeValue(firstCustomAttrIndex, STRING_DEFAULT, /* explicit= */ true);

    if (frozen) {
      container = container.freeze(rule);
    }

    assertThat(container.getAttributeValue(firstCustomAttrIndex)).isNotNull();
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex)).isTrue();
  }

  @Test
  public void nonExplicitDefaultValue_mutable_stored() {
    container.setAttributeValue(firstCustomAttrIndex, STRING_DEFAULT, /* explicit= */ false);

    assertThat(container.getAttributeValue(firstCustomAttrIndex)).isNotNull();
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex)).isFalse();
  }

  @Test
  public void nonExplicitDefaultValue_frozen_notStored() {
    container.setAttributeValue(firstCustomAttrIndex, STRING_DEFAULT, /* explicit= */ false);

    container = container.freeze(rule);

    assertThat(container.getAttributeValue(firstCustomAttrIndex)).isNull();
    assertThat(container.isAttributeValueExplicitlySpecified(firstCustomAttrIndex)).isFalse();
  }

  @Test
  public void computedDefault_mutable_stored() {
    Attribute attr = rule.getRuleClassObject().getAttribute(computedDefaultIndex);
    var computedDefault = attr.getDefaultValue(rule);
    assertThat(attr.hasComputedDefault()).isTrue();
    assertThat(computedDefault).isInstanceOf(ComputedDefault.class);

    container.setAttributeValue(computedDefaultIndex, computedDefault, /* explicit= */ false);

    assertThat(container.getAttributeValue(computedDefaultIndex)).isEqualTo(computedDefault);
    assertThat(container.isAttributeValueExplicitlySpecified(computedDefaultIndex)).isFalse();
  }

  @Test
  public void computedDefault_frozen_notStored() {
    Attribute attr = rule.getRuleClassObject().getAttribute(computedDefaultIndex);
    var computedDefault = attr.getDefaultValue(rule);
    assertThat(attr.hasComputedDefault()).isTrue();
    assertThat(computedDefault).isInstanceOf(ComputedDefault.class);

    container.setAttributeValue(computedDefaultIndex, computedDefault, /* explicit= */ false);
    container = container.freeze(rule);

    assertThat(container.getAttributeValue(computedDefaultIndex)).isNull();
    assertThat(container.isAttributeValueExplicitlySpecified(computedDefaultIndex)).isFalse();
  }
}
