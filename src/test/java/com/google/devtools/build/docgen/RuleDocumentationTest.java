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
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.DocgenConsts.RuleType;
import com.google.devtools.build.docgen.testutil.TestData.TestRule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test class for RuleDocumentation. */
@RunWith(JUnit4.class)
public class RuleDocumentationTest {

  private static final ImmutableSet<String> NO_FLAGS = ImmutableSet.<String>of();

  private static void assertContains(String base, String value) {
    assertWithMessage(base + " is expected to contain " + value)
        .that(base.contains(value))
        .isTrue();
  }

  private void checkAttributeForRule(
      RuleDocumentation rule, RuleDocumentationAttribute attr, boolean isCommonAttribute) {
    rule.addAttribute(attr);
    String signature = rule.getAttributeSignature();
    StringBuilder sb = new StringBuilder();
    if (isCommonAttribute) {
      sb.append("<a href=\"common-definitions.html#");
    } else {
      sb.append("<a href=\"#");
    }
    sb.append(attr.getGeneratedInRule(rule.getRuleName())).append(".");
    sb.append(attr.getAttributeName()).append("\">").append(attr.getAttributeName()).append("</a>");
    assertContains(signature, sb.toString());
  }

  private static RuleDocumentation createRuleDocumentation(
      String ruleName, String ruleType, String ruleFamily, String htmlDocumentation)
      throws BuildEncyclopediaDocException {
    return new RuleDocumentation(
        ruleName,
        ruleType,
        ruleFamily,
        htmlDocumentation,
        "Test.java",
        0,
        "https://example.com/src/Test.java",
        NO_FLAGS,
        "");
  }

  private static RuleDocumentation createRuleDocumentation(
      String ruleName, String ruleType, String ruleFamily) throws BuildEncyclopediaDocException {
    return createRuleDocumentation(ruleName, ruleType, ruleFamily, "");
  }

  @Test
  public void testVariableSubstitution() throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc =
        createRuleDocumentation(
            "rule", "OTHER", "FOO", Joiner.on("\n").join(new String[] {"x", "${VAR}", "z"}));
    ruleDoc.addDocVariable("VAR", "y");
    assertThat(ruleDoc.getHtmlDocumentation()).isEqualTo("x\ny\nz");
  }

  @Test
  public void testSignatureContainsCommonAttribute() throws Exception {
    RuleDocumentationAttribute licensesAttr =
        RuleDocumentationAttribute.createCommon("licenses", "common", "attribute doc");
    checkAttributeForRule(
        createRuleDocumentation("java_binary", "BINARY", "JAVA"), licensesAttr, true);
  }

  @Test
  public void testInheritedAttributeGeneratesSignature() throws Exception {
    RuleDocumentationAttribute runtimeDepsAttr =
        RuleDocumentationAttribute.create(
            TestRule.class, "runtime_deps", "attribute doc", "Test.java", 0, NO_FLAGS);
    checkAttributeForRule(
        createRuleDocumentation("java_binary", "BINARY", "JAVA"), runtimeDepsAttr, false);
    checkAttributeForRule(
        createRuleDocumentation("java_library", "LIBRARY", "JAVA"), runtimeDepsAttr, false);
  }

  @Test
  public void testRuleDocFlagSubstitution() throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc =
        new RuleDocumentation(
            "rule",
            "OTHER",
            "FOO",
            "x",
            "Test.java",
            0,
            "https://example.com/src/Test.java",
            ImmutableSet.of("DEPRECATED"),
            "");
    ruleDoc.addDocVariable("VAR", "y");
    assertThat(ruleDoc.getHtmlDocumentation()).isEqualTo("x");
  }

  @Test
  public void testCommandLineDocumentation() throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc =
        createRuleDocumentation(
            "foo_binary",
            "OTHER",
            "FOO",
            Joiner.on("\n").join(new String[] {"x", "y", "z", "${VAR}"}));
    ruleDoc.addDocVariable("VAR", "w");
    RuleDocumentationAttribute attributeDoc =
        RuleDocumentationAttribute.create(
            TestRule.class, "srcs", "attribute doc", "Test.java", 0, NO_FLAGS);
    ruleDoc.addAttribute(attributeDoc);
    assertThat(ruleDoc.getCommandLineDocumentation()).isEqualTo("\nx\ny\nz\n\n");
  }

  @Test
  public void testCreateExceptions() throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc =
        new RuleDocumentation(
            "foo_binary",
            "OTHER",
            "FOO",
            "",
            "Foo.java",
            10,
            "https://example.com/src/Foo.java",
            NO_FLAGS,
            "");
    BuildEncyclopediaDocException e = ruleDoc.createException("msg");
    assertThat(e).hasMessageThat().isEqualTo("Error in Foo.java:10: msg");
  }

  @Test
  public void getSourceUrl() throws BuildEncyclopediaDocException {
    RuleDocumentation ruleDoc =
        new RuleDocumentation(
            "foo_binary",
            "OTHER",
            "FOO",
            "",
            "Foo.java",
            10,
            "https://example.com/src/Foo.java",
            NO_FLAGS,
            "");
    assertThat(ruleDoc.getSourceUrl()).isEqualTo("https://example.com/src/Foo.java");
  }

  @Test
  public void testEquals() throws BuildEncyclopediaDocException {
    // Expect equality purely based on name
    assertThat(
            new RuleDocumentation(
                "rule",
                "BINARY",
                "FOO",
                "x",
                "Foo.java",
                0,
                "https://example.com/src/Foo.java",
                NO_FLAGS,
                ""))
        .isEqualTo(
            new RuleDocumentation(
                "rule",
                "OTHER",
                "BAR",
                "y",
                "Bar.java",
                10,
                "https://example.com/src/Bar.java",
                ImmutableSet.of("DEPRECATED"),
                "Blah blah blah"));
  }

  @Test
  public void testNotEquals() throws BuildEncyclopediaDocException {
    // Expect inequality purely based on name
    assertThat(
            createRuleDocumentation("rule1", "OTHER", "FOO", "x")
                .equals(createRuleDocumentation("rule2", "OTHER", "FOO", "x")))
        .isFalse();
  }

  @Test
  public void testCompareTo() throws BuildEncyclopediaDocException {
    assertThat(
            createRuleDocumentation("rule1", "OTHER", "FOO", "x")
                .compareTo(createRuleDocumentation("rule2", "OTHER", "FOO", "x")))
        .isEqualTo(-1);
  }

  @Test
  public void testHashCode() throws BuildEncyclopediaDocException {
    assertThat(createRuleDocumentation("rule", "OTHER", "FOO", "y").hashCode())
        .isEqualTo(createRuleDocumentation("rule", "OTHER", "FOO", "x").hashCode());
  }

  @Test
  public void testRuleTypeIsOtherForGenericRules() throws Exception {
    assertThat(
            new RuleDocumentation(
                    "rule",
                    "BINARY",
                    "FOO",
                    "y",
                    "Test.java",
                    0,
                    "https://example.com/src/Test.java",
                    ImmutableSet.of("GENERIC_RULE"),
                    "")
                .getRuleType())
        .isEqualTo(RuleType.OTHER);
    assertThat(
            new RuleDocumentation(
                    "rule",
                    null,
                    "FOO",
                    "y",
                    "Test.java",
                    0,
                    "https://example.com/src/Test.java",
                    ImmutableSet.of("GENERIC_RULE"),
                    "")
                .getRuleType())
        .isEqualTo(RuleType.OTHER);
  }
}
