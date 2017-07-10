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
package com.google.devtools.build.docgen;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.testutil.TestData.BaseRule;
import com.google.devtools.build.docgen.testutil.TestData.IntermediateRule;
import com.google.devtools.build.docgen.testutil.TestData.TestRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.syntax.Type;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test class for RuleDocumentationAttribute.
 */
@RunWith(JUnit4.class)
public class RuleDocumentationAttributeTest {
  private static final ImmutableSet<String> NO_FLAGS = ImmutableSet.<String>of();

  @Test
  public void testDirectChild() {
    RuleDocumentationAttribute attr1 = RuleDocumentationAttribute.create(
        IntermediateRule.class, "", "", 0, "", NO_FLAGS);
    assertThat(attr1.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(1);
  }

  @Test
  public void testTransitiveChild() {
    RuleDocumentationAttribute attr2 = RuleDocumentationAttribute.create(
        BaseRule.class, "", "", 0, "", NO_FLAGS);
    assertThat(attr2.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(2);
  }

  @Test
  public void testClassIsNotChild() {
    RuleDocumentationAttribute attr2 = RuleDocumentationAttribute.create(
        IntermediateRule.class, "", "", 0, "", NO_FLAGS);
    assertThat(attr2.getDefinitionClassAncestryLevel(BaseRule.class, null)).isEqualTo(-1);
  }

  @Test
  public void testClassIsSame() {
    RuleDocumentationAttribute attr3 = RuleDocumentationAttribute.create(
        TestRule.class, "", "", 0, "", NO_FLAGS);
    assertThat(attr3.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(0);
  }

  @Test
  public void testHasFlags() {
    RuleDocumentationAttribute attr = RuleDocumentationAttribute.create(
        TestRule.class, "", "", 0, "", ImmutableSet.<String>of("SOME_FLAG"));
    assertThat(attr.hasFlag("SOME_FLAG")).isTrue();
  }

  @Test
  public void testCompareTo() {
    assertThat(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, "", NO_FLAGS)
                .compareTo(
                    RuleDocumentationAttribute.create(TestRule.class, "b", "", 0, "", NO_FLAGS)))
        .isEqualTo(-1);
  }

  @Test
  public void testCompareToWithPriorityAttributeName() {
    assertThat(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, "", NO_FLAGS)
                .compareTo(
                    RuleDocumentationAttribute.create(TestRule.class, "name", "", 0, "", NO_FLAGS)))
        .isEqualTo(1);
  }

  @Test
  public void testEquals() {
    assertThat(RuleDocumentationAttribute.create(IntermediateRule.class, "a", "", 0, "", NO_FLAGS))
        .isEqualTo(RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, "", NO_FLAGS));
  }

  @Test
  public void testHashCode() {
    assertThat(
            RuleDocumentationAttribute.create(IntermediateRule.class, "a", "", 0, "", NO_FLAGS)
                .hashCode())
        .isEqualTo(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, "", NO_FLAGS).hashCode());
  }

  @Test
  public void testSynopsisForStringAttribute() {
    final String defaultValue = "9";
    Attribute attribute = Attribute.attr("foo_version", Type.STRING)
        .value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, "", NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("String; optional; default is \"" + defaultValue + "\"");
  }

  @Test
  public void testSynopsisForIntegerAttribute() {
    final int defaultValue = 384;
    Attribute attribute = Attribute.attr("bar_limit", Type.INTEGER)
        .value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, "", NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("Integer; optional; default is " + defaultValue);
  }

  @Test
  public void testSynopsisForLabelListAttribute() {
    Attribute attribute = Attribute.attr("some_labels", BuildType.LABEL_LIST)
        .allowedRuleClasses("foo_rule")
        .allowedFileTypes()
        .build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, "", NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("List of <a href=\"../build-ref.html#labels\">labels</a>; optional");
  }

  @Test
  public void testSynopsisForMandatoryAttribute() {
    Attribute attribute = Attribute.attr("baz_labels", BuildType.LABEL)
        .mandatory()
        .allowedFileTypes(CppFileTypes.CPP_HEADER)
        .build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, "", NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("<a href=\"../build-ref.html#labels\">Label</a>; required");
  }
}
