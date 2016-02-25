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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        IntermediateRule.class, "", "", 0, NO_FLAGS);
    assertEquals(1, attr1.getDefinitionClassAncestryLevel(TestRule.class));
  }

  @Test
  public void testTransitiveChild() {
    RuleDocumentationAttribute attr2 = RuleDocumentationAttribute.create(
        BaseRule.class, "", "", 0, NO_FLAGS);
    assertEquals(2, attr2.getDefinitionClassAncestryLevel(TestRule.class));
  }

  @Test
  public void testClassIsNotChild() {
    RuleDocumentationAttribute attr2 = RuleDocumentationAttribute.create(
        IntermediateRule.class, "", "", 0, NO_FLAGS);
    assertEquals(-1, attr2.getDefinitionClassAncestryLevel(BaseRule.class));
  }

  @Test
  public void testClassIsSame() {
    RuleDocumentationAttribute attr3 = RuleDocumentationAttribute.create(
        TestRule.class, "", "", 0, NO_FLAGS);
    assertEquals(0, attr3.getDefinitionClassAncestryLevel(TestRule.class));
  }

  @Test
  public void testHasFlags() {
    RuleDocumentationAttribute attr = RuleDocumentationAttribute.create(
        TestRule.class, "", "", 0, ImmutableSet.<String>of("SOME_FLAG"));
    assertTrue(attr.hasFlag("SOME_FLAG"));
  }

  @Test
  public void testCompareTo() {
    assertTrue(
        RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, NO_FLAGS)
        .compareTo(
            RuleDocumentationAttribute.create(TestRule.class, "b", "", 0, NO_FLAGS))
            == -1);
  }

  @Test
  public void testCompareToWithPriorityAttributeName() {
    assertTrue(
        RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, NO_FLAGS)
        .compareTo(
            RuleDocumentationAttribute.create(TestRule.class, "name", "", 0, NO_FLAGS))
            == 1);
  }

  @Test
  public void testEquals() {
    assertEquals(
        RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, NO_FLAGS),
        RuleDocumentationAttribute.create(IntermediateRule.class, "a", "", 0, NO_FLAGS));
  }

  @Test
  public void testHashCode() {
    assertEquals(
        RuleDocumentationAttribute.create(TestRule.class, "a", "", 0, NO_FLAGS)
        .hashCode(),
        RuleDocumentationAttribute.create(IntermediateRule.class, "a", "", 0, NO_FLAGS)
        .hashCode());
  }

  @Test
  public void testSynopsisForStringAttribute() {
    final String defaultValue = "9";
    Attribute attribute = Attribute.attr("foo_version", Type.STRING)
        .value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertEquals("String; optional; default is \"" + defaultValue + "\"", doc);
  }

  @Test
  public void testSynopsisForIntegerAttribute() {
    final int defaultValue = 384;
    Attribute attribute = Attribute.attr("bar_limit", Type.INTEGER)
        .value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertEquals("Integer; optional; default is " + defaultValue, doc);
  }

  @Test
  public void testSynopsisForLabelListAttribute() {
    Attribute attribute = Attribute.attr("some_labels", BuildType.LABEL_LIST)
        .allowedRuleClasses("foo_rule")
        .allowedFileTypes()
        .build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertEquals("List of <a href=\"../build-ref.html#labels\">labels</a>; optional", doc);
  }

  @Test
  public void testSynopsisForMandatoryAttribute() {
    Attribute attribute = Attribute.attr("baz_labels", BuildType.LABEL)
        .mandatory()
        .allowedFileTypes(CppFileTypes.CPP_HEADER)
        .build();
    RuleDocumentationAttribute attributeDoc = RuleDocumentationAttribute.create(
        TestRule.class, "testrule", "", 0, NO_FLAGS);
    attributeDoc.setAttribute(attribute);
    String doc = attributeDoc.getSynopsis();
    assertEquals("<a href=\"../build-ref.html#labels\">Label</a>; required", doc);
  }
}
