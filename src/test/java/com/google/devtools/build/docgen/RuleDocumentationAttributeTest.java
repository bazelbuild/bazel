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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.testutil.TestData.BaseRule;
import com.google.devtools.build.docgen.testutil.TestData.IntermediateRule;
import com.google.devtools.build.docgen.testutil.TestData.TestRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
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
    RuleDocumentationAttribute attr1 =
        RuleDocumentationAttribute.create(IntermediateRule.class, "", "", "Test.java", 0, NO_FLAGS);
    assertThat(attr1.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(1);
  }

  @Test
  public void testTransitiveChild() {
    RuleDocumentationAttribute attr2 =
        RuleDocumentationAttribute.create(BaseRule.class, "", "", "Test.java", 0, NO_FLAGS);
    assertThat(attr2.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(2);
  }

  @Test
  public void testClassIsNotChild() {
    RuleDocumentationAttribute attr2 =
        RuleDocumentationAttribute.create(IntermediateRule.class, "", "", "Test.java", 0, NO_FLAGS);
    assertThat(attr2.getDefinitionClassAncestryLevel(BaseRule.class, null)).isEqualTo(-1);
  }

  @Test
  public void testClassIsSame() {
    RuleDocumentationAttribute attr3 =
        RuleDocumentationAttribute.create(TestRule.class, "", "", "Test.java", 0, NO_FLAGS);
    assertThat(attr3.getDefinitionClassAncestryLevel(TestRule.class, null)).isEqualTo(0);
  }

  @Test
  public void testHasFlags() {
    RuleDocumentationAttribute attr =
        RuleDocumentationAttribute.create(
            TestRule.class, "", "", "Test.java", 0, ImmutableSet.<String>of("SOME_FLAG"));
    assertThat(attr.hasFlag("SOME_FLAG")).isTrue();
  }

  @Test
  public void testCompareTo() {
    assertThat(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", "Test.java", 0, NO_FLAGS)
                .compareTo(
                    RuleDocumentationAttribute.create(
                        TestRule.class, "b", "", "Test.java", 0, NO_FLAGS)))
        .isEqualTo(-1);
  }

  @Test
  public void testCompareToWithPriorityAttributeName() {
    assertThat(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", "Test.java", 0, NO_FLAGS)
                .compareTo(
                    RuleDocumentationAttribute.create(
                        TestRule.class, "name", "", "Test.java", 0, NO_FLAGS)))
        .isEqualTo(1);
  }

  @Test
  public void testEquals() {
    assertThat(
            RuleDocumentationAttribute.create(
                IntermediateRule.class, "a", "", "Test.java", 0, NO_FLAGS))
        .isEqualTo(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", "Test.java", 0, NO_FLAGS));
  }

  @Test
  public void testHashCode() {
    assertThat(
            RuleDocumentationAttribute.create(
                    IntermediateRule.class, "a", "", "Test.java", 0, NO_FLAGS)
                .hashCode())
        .isEqualTo(
            RuleDocumentationAttribute.create(TestRule.class, "a", "", "Test.java", 0, NO_FLAGS)
                .hashCode());
  }

  @Test
  public void synopsis_stringAttribute() throws BuildEncyclopediaDocException {
    final String defaultValue = "9";
    Attribute attribute = Attribute.attr("foo_version", Type.STRING).value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(TestRule.class, "testrule", "", "Test.java", 0, NO_FLAGS)
            .copyAndUpdateFrom(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("String; default is <code>\"" + defaultValue + "\"</code>");
  }

  @Test
  public void synopsis_stringAttribute_fromProto() throws BuildEncyclopediaDocException {
    final String defaultValue = "9";
    AttributeInfo attributeInfo =
        AttributeInfo.newBuilder()
            .setName("foo_version")
            .setType(AttributeType.STRING)
            .setDefaultValue(Starlark.repr(defaultValue))
            .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.createFromAttributeInfo(attributeInfo, "//:test.bzl", NO_FLAGS);
    assertThat(attributeDoc.getSynopsis())
        .isEqualTo("String; default is <code>\"" + defaultValue + "\"</code>");
  }

  @Test
  public void synopsis_integerAttribute() throws BuildEncyclopediaDocException {
    StarlarkInt defaultValue = StarlarkInt.of(384);
    Attribute attribute = Attribute.attr("bar_limit", Type.INTEGER)
        .value(defaultValue).build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(TestRule.class, "testrule", "", "Test.java", 0, NO_FLAGS)
            .copyAndUpdateFrom(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("Integer; default is <code>" + defaultValue + "</code>");
  }

  @Test
  public void synopsis_integerAttribute_fromProto() throws BuildEncyclopediaDocException {
    int defaultValue = 384;
    AttributeInfo attributeInfo =
        AttributeInfo.newBuilder()
            .setName("bar_limit")
            .setType(AttributeType.INT)
            .setDefaultValue(Starlark.repr(defaultValue))
            .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.createFromAttributeInfo(attributeInfo, "//:test.bzl", NO_FLAGS);
    assertThat(attributeDoc.getSynopsis())
        .isEqualTo("Integer; default is <code>" + defaultValue + "</code>");
  }

  @Test
  public void synopsis_labelListAttribute() throws BuildEncyclopediaDocException {
    Attribute attribute = Attribute.attr("some_labels", BuildType.LABEL_LIST)
        .allowedRuleClasses("foo_rule")
        .allowedFileTypes()
        .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(TestRule.class, "testrule", "", "Test.java", 0, NO_FLAGS)
            .copyAndUpdateFrom(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc)
        .isEqualTo(
            "List of <a href=\"${link build-ref#labels}\">labels</a>; default is <code>[]</code>");
  }

  @Test
  public void synopsis_labelListAttribute_fromProto() throws BuildEncyclopediaDocException {
    AttributeInfo attributeInfo =
        AttributeInfo.newBuilder()
            .setName("some_labels")
            .setType(AttributeType.LABEL_LIST)
            .setDefaultValue(Starlark.repr(ImmutableList.of()))
            .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.createFromAttributeInfo(attributeInfo, "//:test.bzl", NO_FLAGS);
    assertThat(attributeDoc.getSynopsis())
        .isEqualTo(
            "List of <a href=\"${link build-ref#labels}\">labels</a>; default is <code>[]</code>");
  }

  @Test
  public void synopsis_mandatoryAttribute() throws BuildEncyclopediaDocException {
    Attribute attribute =
        Attribute.attr("baz_labels", BuildType.LABEL)
            .mandatory()
            .allowedFileTypes(CppFileTypes.CPP_HEADER)
            .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(TestRule.class, "testrule", "", "Test.java", 0, NO_FLAGS)
            .copyAndUpdateFrom(attribute);
    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("<a href=\"${link build-ref#labels}\">Label</a>; required");
  }

  @Test
  public void synopsis_mandatoryAttribute_fromProto() throws BuildEncyclopediaDocException {
    AttributeInfo attributeInfo =
        AttributeInfo.newBuilder()
            .setName("baz_labels")
            .setType(AttributeType.LABEL)
            .setMandatory(true)
            .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.createFromAttributeInfo(attributeInfo, "//:test.bzl", NO_FLAGS);
    assertThat(attributeDoc.getSynopsis())
        .isEqualTo("<a href=\"${link build-ref#labels}\">Label</a>; required");
  }

  @Test
  public void testSynopsisWithRuleLinkExpander() throws BuildEncyclopediaDocException {
    Attribute attribute = Attribute.attr("baz_labels", BuildType.LABEL)
        .mandatory()
        .allowedFileTypes(CppFileTypes.CPP_HEADER)
        .build();
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(TestRule.class, "testrule", "", "Test.java", 0, NO_FLAGS)
            .copyAndUpdateFrom(attribute);

    DocLinkMap linkMap =
        new DocLinkMap(
            "",
            ImmutableMap.of("build-ref", "THE_REF.html"),
            "https://example.com/",
            ImmutableMap.of());
    RuleLinkExpander expander = new RuleLinkExpander(false, linkMap);
    attributeDoc.setRuleLinkExpander(expander);

    String doc = attributeDoc.getSynopsis();
    assertThat(doc).isEqualTo("<a href=\"THE_REF.html#labels\">Label</a>; required");
  }
}
