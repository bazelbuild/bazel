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
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Tests of Attribute code.
 */
@RunWith(JUnit4.class)
public class AttributeTest {

  private void assertDefaultValue(Object expected, Attribute attr) {
    assertEquals(expected, attr.getDefaultValue(null));
  }

  private void assertType(Type<?> expectedType, Attribute attr) {
    assertEquals(expectedType, attr.getType());
  }

  @Test
  public void testBasics() throws Exception {
    Attribute attr = attr("foo", Type.INTEGER).mandatory().value(3).build();
    assertEquals("foo", attr.getName());
    assertEquals(3, attr.getDefaultValue(null));
    assertEquals(Type.INTEGER, attr.getType());
    assertTrue(attr.isMandatory());
    assertTrue(attr.isDocumented());
    attr = attr("$foo", Type.INTEGER).build();
    assertFalse(attr.isDocumented());
  }

  @Test
  public void testNonEmptyReqiresListType() throws Exception {
    try {
      attr("foo", Type.INTEGER).nonEmpty().value(3).build();
      fail();
    } catch (NullPointerException e) {
      assertThat(e).hasMessage("attribute 'foo' must be a list");
    }
  }

  @Test
  public void testNonEmpty() throws Exception {
    Attribute attr = attr("foo", BuildType.LABEL_LIST).nonEmpty().legacyAllowAnyFileType().build();
    assertEquals("foo", attr.getName());
    assertEquals(BuildType.LABEL_LIST, attr.getType());
    assertTrue(attr.isNonEmpty());
  }

  @Test
  public void testSingleArtifactReqiresLabelType() throws Exception {
    try {
      attr("foo", Type.INTEGER).singleArtifact().value(3).build();
      fail();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessage("attribute 'foo' must be a label-valued type");
    }
  }

  @Test
  public void testDoublePropertySet() {
    Attribute.Builder<String> builder = attr("x", STRING).mandatory().cfg(HOST).undocumented("")
        .value("y");
    try {
      builder.mandatory();
      fail();
    } catch (IllegalStateException expected) {
      // expected
    }
    try {
      builder.cfg(HOST);
      fail();
    } catch (IllegalStateException expected) {
      // expected
    }
    try {
      builder.undocumented("");
      fail();
    } catch (IllegalStateException expected) {
      // expected
    }
    try {
      builder.value("z");
      fail();
    } catch (IllegalStateException expected) {
      // expected
    }

    builder = attr("$x", STRING);
    try {
      builder.undocumented("");
      fail();
    } catch (IllegalStateException expected) {
      // expected
    }
  }

  /**
   *  Tests the "convenience factories" (string, label, etc) for default
   *  values.
   */
  @Test
  public void testConvenienceFactoriesDefaultValues() throws Exception {
    assertDefaultValue(0,
                       attr("x", INTEGER).build());
    assertDefaultValue(42,
                       attr("x", INTEGER).value(42).build());

    assertDefaultValue("",
                       attr("x", STRING).build());
    assertDefaultValue("foo",
                       attr("x", STRING).value("foo").build());

    Label label = Label.parseAbsolute("//foo:bar");
    assertDefaultValue(null,
                       attr("x", LABEL).legacyAllowAnyFileType().build());
    assertDefaultValue(label,
                       attr("x", LABEL).legacyAllowAnyFileType().value(label).build());

    List<String> slist = Arrays.asList("foo", "bar");
    assertDefaultValue(Collections.emptyList(),
                       attr("x", STRING_LIST).build());
    assertDefaultValue(slist,
                       attr("x", STRING_LIST).value(slist).build());

    List<Label> llist = Arrays.asList(Label.parseAbsolute("//foo:bar"),
                                      Label.parseAbsolute("//foo:wiz"));
    assertDefaultValue(Collections.emptyList(),
                       attr("x", LABEL_LIST).legacyAllowAnyFileType().build());
    assertDefaultValue(llist,
                       attr("x", LABEL_LIST).legacyAllowAnyFileType().value(llist).build());
  }

  /**
   *  Tests the "convenience factories" (string, label, etc) for types.
   */
  @Test
  public void testConvenienceFactoriesTypes() throws Exception {
    assertType(INTEGER,
               attr("x", INTEGER).build());
    assertType(INTEGER,
               attr("x", INTEGER).value(42).build());

    assertType(STRING,
               attr("x", STRING).build());
    assertType(STRING,
               attr("x", STRING).value("foo").build());

    Label label = Label.parseAbsolute("//foo:bar");
    assertType(LABEL,
                       attr("x", LABEL).legacyAllowAnyFileType().build());
    assertType(LABEL,
               attr("x", LABEL).legacyAllowAnyFileType().value(label).build());

    List<String> slist = Arrays.asList("foo", "bar");
    assertType(STRING_LIST,
               attr("x", STRING_LIST).build());
    assertType(STRING_LIST,
               attr("x", STRING_LIST).value(slist).build());

    List<Label> llist = Arrays.asList(Label.parseAbsolute("//foo:bar"),
                                      Label.parseAbsolute("//foo:wiz"));
    assertType(LABEL_LIST,
               attr("x", LABEL_LIST).legacyAllowAnyFileType().build());
    assertType(LABEL_LIST,
               attr("x", LABEL_LIST).legacyAllowAnyFileType().value(llist).build());
  }

  @Test
  public void testCloneBuilder() {
    FileTypeSet txtFiles = FileTypeSet.of(FileType.of("txt"));
    RuleClass.Builder.RuleClassNamePredicate ruleClasses =
        new RuleClass.Builder.RuleClassNamePredicate("mock_rule");

    Attribute parentAttr =
        attr("x", LABEL_LIST)
            .allowedFileTypes(txtFiles)
            .mandatory()
            .aspect(TestAspects.SimpleAspect.class)
            .build();

    {
      Attribute childAttr1 = parentAttr.cloneBuilder().build();
      assertEquals("x", childAttr1.getName());
      assertEquals(txtFiles, childAttr1.getAllowedFileTypesPredicate());
      assertEquals(Predicates.alwaysTrue(), childAttr1.getAllowedRuleClassesPredicate());
      assertTrue(childAttr1.isMandatory());
      assertFalse(childAttr1.isNonEmpty());
      assertThat(childAttr1.getAspects(null /* rule */)).hasSize(1);
    }

    {
      Attribute childAttr2 =
          parentAttr
              .cloneBuilder()
              .nonEmpty()
              .allowedRuleClasses(ruleClasses)
              .aspect(TestAspects.ErrorAspect.class)
              .build();
      assertEquals("x", childAttr2.getName());
      assertEquals(txtFiles, childAttr2.getAllowedFileTypesPredicate());
      assertEquals(ruleClasses, childAttr2.getAllowedRuleClassesPredicate());
      assertTrue(childAttr2.isMandatory());
      assertTrue(childAttr2.isNonEmpty());
      assertThat(childAttr2.getAspects(null /* rule */)).hasSize(2);
    }

    //Check if the parent attribute is unchanged
    assertFalse(parentAttr.isNonEmpty());
    assertEquals(Predicates.alwaysTrue(), parentAttr.getAllowedRuleClassesPredicate());
  }

  /**
   * Tests that configurability settings are properly received.
   */
  @Test
  public void testConfigurability() {
    assertTrue(attr("foo_configurable", BuildType.LABEL_LIST).legacyAllowAnyFileType().build()
        .isConfigurable());
    assertFalse(attr("foo_nonconfigurable", BuildType.LABEL_LIST).legacyAllowAnyFileType()
        .nonconfigurable("test").build().isConfigurable());
  }
}
