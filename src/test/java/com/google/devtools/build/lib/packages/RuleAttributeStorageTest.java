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
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.Collections;
import java.util.Iterator;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link Rule}'s attribute storage behavior. */
@RunWith(TestParameterInjector.class)
public final class RuleAttributeStorageTest extends BuildViewTestCase {

  private static final String STRING_DEFAULT = Type.STRING.getDefaultValue();
  private static final int COMPUTED_DEFAULT_OFFSET = 1;
  private static final int LATE_BOUND_DEFAULT_OFFSET = 2;

  private enum ContainerSize {
    SMALL(16),
    LARGE(128);

    private final int numAttrs;

    ContainerSize(int numAttrs) {
      this.numAttrs = numAttrs;
    }
  }

  @TestParameter private ContainerSize containerSize;

  private Rule rule;
  private int firstCustomAttrIndex;
  private Attribute firstCustomAttr;
  private int lastCustomAttrIndex;
  private Attribute lastCustomAttr;
  private int computedDefaultIndex;
  private Attribute computedDefaultAttr;
  private int lateBoundDefaultIndex;
  private Attribute lateBoundDefaultAttr;

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
                          // Make one attribute a computed default and one a late bound default.
                          if (i == COMPUTED_DEFAULT_OFFSET) {
                            return attr("attr" + i + "_computed_default", Type.STRING)
                                .value(
                                    new ComputedDefault() {
                                      @Override
                                      public Object getDefault(AttributeMap rule) {
                                        return "computed";
                                      }
                                    });
                          }
                          if (i == LATE_BOUND_DEFAULT_OFFSET) {
                            return attr(":attr" + i + "_late_bound_default", Type.STRING)
                                .value(
                                    new LateBoundDefault<>(Void.class, (rule) -> "late_bound") {
                                      @Override
                                      public String resolve(
                                          Rule rule, AttributeMap attributes, Void input) {
                                        return "late_bound";
                                      }
                                    });
                          }
                          return attr("attr" + i, Type.STRING);
                        })
                    .toArray(Attribute.Builder[]::new));
    var builder = new ConfiguredRuleClassProvider.Builder().addRuleDefinition(exampleRule);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Before
  public void setUpForRule() throws Exception {
    scratch.file("foo/BUILD", "example_rule(name = 'example')");

    // Make a mutable copy of the rule so we can set attributes.
    Rule actualRule = (Rule) getTarget("//foo:example");
    rule =
        new Rule(
            actualRule.getPackage(),
            actualRule.getLabel(),
            actualRule.getRuleClassObject(),
            actualRule.getLocation(),
            actualRule.getInteriorCallStack());

    firstCustomAttrIndex = rule.getRuleClassObject().getAttributeIndex("attr0");
    firstCustomAttr = attrAt(firstCustomAttrIndex);
    lastCustomAttrIndex = rule.getRuleClassObject().getAttributeCount() - 1;
    lastCustomAttr = attrAt(lastCustomAttrIndex);
    computedDefaultIndex = firstCustomAttrIndex + COMPUTED_DEFAULT_OFFSET;
    computedDefaultAttr = attrAt(computedDefaultIndex);
    lateBoundDefaultIndex = firstCustomAttrIndex + LATE_BOUND_DEFAULT_OFFSET;
    lateBoundDefaultAttr = attrAt(lateBoundDefaultIndex);
  }

  @Test
  public void attributeSettingAndRetrieval(@TestParameter boolean frozen) {
    rule.setAttributeValue(firstCustomAttr, "val1", /* explicit= */ true);
    rule.setAttributeValue(lastCustomAttr, "val2", /* explicit= */ true);

    if (frozen) {
      rule.freeze();
    }

    assertThat(rule.getAttrIfStored(firstCustomAttrIndex)).isEqualTo("val1");
    assertThat(rule.isAttributeValueExplicitlySpecified(firstCustomAttr)).isTrue();
    assertThat(rule.getAttrIfStored(lastCustomAttrIndex)).isEqualTo("val2");
    assertThat(rule.isAttributeValueExplicitlySpecified(lastCustomAttr)).isTrue();
  }

  @Test
  public void indexOutOfBounds_throws(@TestParameter boolean frozen) {
    if (frozen) {
      rule.freeze();
    }
    assertThrows(
        IndexOutOfBoundsException.class, () -> rule.getAttrIfStored(lastCustomAttrIndex + 1));
  }

  @Test
  public void testForOffByOneError(@TestParameter boolean frozen) {
    // Set an index explicitly and check neighbouring indices don't leak that.
    rule.setAttributeValue(firstCustomAttr, "val", true);

    if (frozen) {
      rule.freeze();
    }

    assertThat(rule.getAttrIfStored(firstCustomAttrIndex - 1)).isNull();
    assertThat(rule.isAttributeValueExplicitlySpecified(attrAt(firstCustomAttrIndex - 1)))
        .isFalse();
    assertThat(rule.getAttrIfStored(firstCustomAttrIndex + 1)).isNull();
    assertThat(rule.isAttributeValueExplicitlySpecified(attrAt(firstCustomAttrIndex + 1)))
        .isFalse();
  }

  @Test
  public void testFreezeWorks() {
    rule.setAttributeValue(firstCustomAttr, "val1", /* explicit= */ true);
    rule.setAttributeValue(lastCustomAttr, "val2", /* explicit= */ false);
    assertThat(rule.isFrozen()).isFalse();

    rule.freeze();

    assertThat(rule.isFrozen()).isTrue();
    // Double freezing is a no-op
    rule.freeze();
    // reads/explicit bits work as expected
    assertThat(rule.getAttrIfStored(firstCustomAttrIndex)).isEqualTo("val1");
    assertThat(rule.isAttributeValueExplicitlySpecified(firstCustomAttr)).isTrue();
    assertThat(rule.getAttrIfStored(lastCustomAttrIndex)).isEqualTo("val2");
    assertThat(rule.isAttributeValueExplicitlySpecified(lastCustomAttr)).isFalse();
    // writes no longer work.
    assertThrows(
        IllegalStateException.class,
        () -> rule.setAttributeValue(lastCustomAttr, "different", true));
  }

  @Test
  public void allAttributesSet(@TestParameter boolean frozen) {
    int size = rule.getRuleClassObject().getAttributeCount();
    rule.setAttributeValue(attrAt(0), rule.getName(), /* explicit= */ true);
    for (int i = 1; i < size; i++) {
      rule.setAttributeValue(attrAt(i), "value " + i, i % 2 == 0);
    }

    if (frozen) {
      rule.freeze();
    }

    for (int i = 1; i < size; i++) { // Skip attribute 0 (name) which is never stored.
      assertThat(rule.getAttrIfStored(i)).isEqualTo("value " + i);
      assertWithMessage("attribute " + i)
          .that(rule.isAttributeValueExplicitlySpecified(attrAt(i)))
          .isEqualTo(i % 2 == 0);
    }
  }

  @Test
  public void getRawAttrValues_mutable_nullSafe() {
    assertThat(rule.getRawAttrValues())
        .containsAtLeastElementsIn(
            Collections.nCopies(rule.getRuleClassObject().getAttributeCount(), null));
  }

  @Test
  public void getRawAttrValues_frozen_noNulls() {
    rule.setAttributeValue(firstCustomAttr, "hi", /* explicit= */ true);
    rule.setAttributeValue(lastCustomAttr, null, /* explicit= */ false);
    rule.freeze();
    assertThat(rule.getRawAttrValues()).containsExactly("hi");
  }

  @Test
  public void getRawAttrValues_unmodifiable(@TestParameter boolean frozen) {
    rule.setAttributeValue(firstCustomAttr, "hi", /* explicit= */ true);

    if (frozen) {
      rule.freeze();
    }

    Iterator<Object> it = rule.getRawAttrValues().iterator();
    it.next();
    assertThrows(UnsupportedOperationException.class, it::remove);
  }

  /** Regression test for b/269593252. */
  @Test
  public void boundaryOfFrozenContainer() {
    String ruleName = rule.getName();
    rule.setAttributeValue(attrAt(0), ruleName, /* explicit= */ true);
    rule.setAttributeValue(lastCustomAttr, "last", /* explicit= */ true);

    rule.freeze();

    assertThat(rule.getAttr("name")).isEqualTo(ruleName);
    assertThat(rule.isAttributeValueExplicitlySpecified("name")).isTrue();
    assertThat(rule.getAttrIfStored(lastCustomAttrIndex)).isEqualTo("last");
    assertThat(rule.isAttributeValueExplicitlySpecified(lastCustomAttr)).isTrue();
  }

  @Test
  public void nameNotStoredAsRawAttr(@TestParameter boolean frozen) {
    String ruleName = rule.getName();
    rule.setAttributeValue(attrAt(0), ruleName, /* explicit= */ true);

    if (frozen) {
      rule.freeze();
    }

    assertThat(rule.getAttrIfStored(0)).isNull();
    assertThat(rule.getRawAttrValues()).doesNotContain(ruleName);
    assertThat(rule.getAttr("name")).isEqualTo(ruleName);
    assertThat(rule.isAttributeValueExplicitlySpecified("name")).isTrue();
  }

  @Test
  public void explicitDefaultValue_stored(@TestParameter boolean frozen) {
    rule.setAttributeValue(firstCustomAttr, STRING_DEFAULT, /* explicit= */ true);

    if (frozen) {
      rule.freeze();
    }

    assertThat(rule.getAttrIfStored(firstCustomAttrIndex)).isNotNull();
    assertThat(rule.isAttributeValueExplicitlySpecified(firstCustomAttr)).isTrue();
  }

  @Test
  public void nonExplicitDefaultValue_mutable_stored() {
    rule.setAttributeValue(firstCustomAttr, STRING_DEFAULT, /* explicit= */ false);

    assertThat(rule.getAttrIfStored(firstCustomAttrIndex)).isNotNull();
    assertThat(rule.isAttributeValueExplicitlySpecified(firstCustomAttr)).isFalse();
  }

  @Test
  public void nonExplicitDefaultValue_frozen_notStored() {
    rule.setAttributeValue(firstCustomAttr, STRING_DEFAULT, /* explicit= */ false);

    rule.freeze();

    assertThat(rule.getAttrIfStored(firstCustomAttrIndex)).isNull();
    assertThat(rule.isAttributeValueExplicitlySpecified(firstCustomAttr)).isFalse();
  }

  @Test
  public void computedDefault_mutable_stored() {
    var computedDefault = computedDefaultAttr.getDefaultValue(null);
    assertThat(computedDefaultAttr.hasComputedDefault()).isTrue();
    assertThat(computedDefault).isInstanceOf(ComputedDefault.class);

    rule.setAttributeValue(computedDefaultAttr, computedDefault, /* explicit= */ false);

    assertThat(rule.getAttrIfStored(computedDefaultIndex)).isEqualTo(computedDefault);
    assertThat(rule.getAttr(computedDefaultAttr.getName())).isEqualTo(computedDefault);
    assertThat(rule.isAttributeValueExplicitlySpecified(computedDefaultAttr)).isFalse();
  }

  @Test
  public void computedDefault_frozen_notStored() {
    var computedDefault = computedDefaultAttr.getDefaultValue(null);
    assertThat(computedDefaultAttr.hasComputedDefault()).isTrue();
    assertThat(computedDefault).isInstanceOf(ComputedDefault.class);

    rule.setAttributeValue(computedDefaultAttr, computedDefault, /* explicit= */ false);
    rule.freeze();

    assertThat(rule.getAttrIfStored(computedDefaultIndex)).isNull();
    assertThat(rule.getAttr(computedDefaultAttr.getName())).isEqualTo(computedDefault);
    assertThat(rule.isAttributeValueExplicitlySpecified(computedDefaultAttr)).isFalse();
  }

  @Test
  public void lateBoundDefault_mutable_stored() {
    var lateBoundDefault = lateBoundDefaultAttr.getLateBoundDefault();

    rule.setAttributeValue(lateBoundDefaultAttr, lateBoundDefault, /* explicit= */ false);

    assertThat(rule.getAttrIfStored(lateBoundDefaultIndex)).isEqualTo(lateBoundDefault);
    assertThat(rule.getAttr(lateBoundDefaultAttr.getName())).isEqualTo(lateBoundDefault);
    assertThat(rule.isAttributeValueExplicitlySpecified(lateBoundDefaultAttr)).isFalse();
  }

  @Test
  public void lateBoundDefault_frozen_notStored() {
    var lateBoundDefault = lateBoundDefaultAttr.getLateBoundDefault();

    rule.setAttributeValue(lateBoundDefaultAttr, lateBoundDefault, /* explicit= */ false);
    rule.freeze();

    assertThat(rule.getAttrIfStored(lateBoundDefaultIndex)).isNull();
    assertThat(rule.getAttr(lateBoundDefaultAttr.getName())).isEqualTo(lateBoundDefault);
    assertThat(rule.isAttributeValueExplicitlySpecified(lateBoundDefaultAttr)).isFalse();
  }

  private Attribute attrAt(int attrIndex) {
    return rule.getRuleClassObject().getAttribute(attrIndex);
  }
}
