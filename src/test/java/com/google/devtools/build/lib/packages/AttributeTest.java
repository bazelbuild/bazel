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
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassNamePredicate;
import com.google.devtools.build.lib.testutil.FakeAttributeMapper;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.StarlarkInt;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Attribute}. */
@RunWith(JUnit4.class)
public final class AttributeTest {

  private static void assertDefaultValue(Object expected, Attribute attr) {
    assertThat(attr.getDefaultValue(null)).isEqualTo(expected);
  }

  private static void assertType(Type<?> expectedType, Attribute attr) {
    assertThat(attr.getType()).isEqualTo(expectedType);
  }

  @Test
  public void testBasics() {
    Attribute attr = attr("foo", Type.INTEGER).mandatory().value(StarlarkInt.of(3)).build();
    assertThat(attr.getName()).isEqualTo("foo");
    assertThat(attr.getDefaultValue(null)).isEqualTo(StarlarkInt.of(3));
    assertThat(attr.getType()).isEqualTo(Type.INTEGER);
    assertThat(attr.isMandatory()).isTrue();
    assertThat(attr.isDocumented()).isTrue();
    assertThat(attr.starlarkDefined()).isFalse();
    attr = attr("$foo", Type.INTEGER).build();
    assertThat(attr.isDocumented()).isFalse();
  }

  @Test
  public void testNonEmptyRequiresListType() {
    NullPointerException e =
        assertThrows(
            NullPointerException.class,
            () -> attr("foo", Type.INTEGER).nonEmpty().value(StarlarkInt.of(3)).build());
    assertThat(e).hasMessageThat().isEqualTo("attribute 'foo' must be a list");
  }

  @Test
  public void testNonEmpty() {
    Attribute attr = attr("foo", BuildType.LABEL_LIST).nonEmpty().legacyAllowAnyFileType().build();
    assertThat(attr.getName()).isEqualTo("foo");
    assertThat(attr.getType()).isEqualTo(BuildType.LABEL_LIST);
    assertThat(attr.isNonEmpty()).isTrue();
  }

  @Test
  public void testSingleArtifactRequiresLabelType() {
    IllegalStateException e =
        assertThrows(
            IllegalStateException.class,
            () -> attr("foo", Type.INTEGER).singleArtifact().value(StarlarkInt.of(3)).build());
    assertThat(e).hasMessageThat().isEqualTo("attribute 'foo' must be a label-valued type");
  }

  @Test
  public void testDoublePropertySet() {
    Attribute.Builder<String> builder =
        attr("x", STRING)
            .mandatory()
            .cfg(ExecutionTransitionFactory.createFactory())
            .undocumented("")
            .value("y");
    assertThrows(IllegalStateException.class, builder::mandatory);
    assertThrows(
        IllegalStateException.class, () -> builder.cfg(ExecutionTransitionFactory.createFactory()));
    assertThrows(IllegalStateException.class, () -> builder.undocumented(""));
    assertThrows(IllegalStateException.class, () -> builder.value("z"));

    Attribute.Builder<String> builder2 = attr("$x", STRING);
    assertThrows(IllegalStateException.class, () -> builder2.undocumented(""));
  }

  /**
   *  Tests the "convenience factories" (string, label, etc) for default
   *  values.
   */
  @Test
  public void testConvenienceFactoriesDefaultValues() throws Exception {
    assertDefaultValue(StarlarkInt.of(0), attr("x", INTEGER).build());
    assertDefaultValue(StarlarkInt.of(42), attr("x", INTEGER).value(StarlarkInt.of(42)).build());

    assertDefaultValue("",
                       attr("x", STRING).build());
    assertDefaultValue("foo",
                       attr("x", STRING).value("foo").build());

    Label label = Label.parseCanonical("//foo:bar");
    assertDefaultValue(null,
                       attr("x", LABEL).legacyAllowAnyFileType().build());
    assertDefaultValue(label,
                       attr("x", LABEL).legacyAllowAnyFileType().value(label).build());

    List<String> slist = Arrays.asList("foo", "bar");
    assertDefaultValue(Collections.emptyList(),
                       attr("x", STRING_LIST).build());
    assertDefaultValue(slist,
                       attr("x", STRING_LIST).value(slist).build());

    List<Label> llist =
        Arrays.asList(Label.parseCanonical("//foo:bar"), Label.parseCanonical("//foo:wiz"));
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
    assertType(INTEGER, attr("x", INTEGER).value(StarlarkInt.of(42)).build());

    assertType(STRING,
               attr("x", STRING).build());
    assertType(STRING,
               attr("x", STRING).value("foo").build());

    Label label = Label.parseCanonical("//foo:bar");
    assertType(LABEL,
                       attr("x", LABEL).legacyAllowAnyFileType().build());
    assertType(LABEL,
               attr("x", LABEL).legacyAllowAnyFileType().value(label).build());

    List<String> slist = Arrays.asList("foo", "bar");
    assertType(STRING_LIST,
               attr("x", STRING_LIST).build());
    assertType(STRING_LIST,
               attr("x", STRING_LIST).value(slist).build());

    List<Label> llist =
        Arrays.asList(Label.parseCanonical("//foo:bar"), Label.parseCanonical("//foo:wiz"));
    assertType(LABEL_LIST,
               attr("x", LABEL_LIST).legacyAllowAnyFileType().build());
    assertType(LABEL_LIST,
               attr("x", LABEL_LIST).legacyAllowAnyFileType().value(llist).build());
  }

  @Test
  public void testCloneBuilder() {
    FileTypeSet txtFiles = FileTypeSet.of(FileType.of("txt"));
    RuleClassNamePredicate ruleClasses = RuleClassNamePredicate.only("mock_rule");

    Attribute parentAttr =
        attr("x", LABEL_LIST)
            .allowedFileTypes(txtFiles)
            .mandatory()
            .aspect(TestAspects.SIMPLE_ASPECT)
            .build();

    {
      Attribute childAttr1 = parentAttr.cloneBuilder().build();
      assertThat(childAttr1.getName()).isEqualTo("x");
      assertThat(childAttr1.getAllowedFileTypesPredicate()).isEqualTo(txtFiles);
      assertThat(childAttr1.getAllowedRuleClassObjectPredicate())
          .isEqualTo(Predicates.alwaysTrue());
      assertThat(childAttr1.isMandatory()).isTrue();
      assertThat(childAttr1.isNonEmpty()).isFalse();
      assertThat(childAttr1.getAspects(/* rule= */ null)).hasSize(1);
    }

    {
      Attribute childAttr2 =
          parentAttr
              .cloneBuilder()
              .nonEmpty()
              .allowedRuleClasses(ruleClasses)
              .aspect(TestAspects.ERROR_ASPECT)
              .build();
      assertThat(childAttr2.getName()).isEqualTo("x");
      assertThat(childAttr2.getAllowedFileTypesPredicate()).isEqualTo(txtFiles);
      assertThat(childAttr2.getAllowedRuleClassObjectPredicate())
          .isEqualTo(ruleClasses.asPredicateOfRuleClassObject());
      assertThat(childAttr2.isMandatory()).isTrue();
      assertThat(childAttr2.isNonEmpty()).isTrue();
      assertThat(childAttr2.getAspects(/* rule= */ null)).hasSize(2);
    }

    // Check if the parent attribute is unchanged
    assertThat(parentAttr.isNonEmpty()).isFalse();
    assertThat(parentAttr.getAllowedRuleClassObjectPredicate()).isEqualTo(Predicates.alwaysTrue());
  }

  /**
   * Tests that configurability settings are properly received.
   */
  @Test
  public void testConfigurability() {
    assertThat(
            attr("foo_configurable", BuildType.LABEL_LIST)
                .legacyAllowAnyFileType()
                .build()
                .isConfigurable())
        .isTrue();
    assertThat(
            attr("foo_nonconfigurable", BuildType.LABEL_LIST)
                .legacyAllowAnyFileType()
                .nonconfigurable("test")
                .build()
                .isConfigurable())
        .isFalse();
  }

  @Test
  public void testSplitTransition() {
    TestSplitTransition splitTransition = new TestSplitTransition();
    Attribute attr =
        attr("foo", LABEL).cfg(TransitionFactories.of(splitTransition)).allowedFileTypes().build();
    assertThat(attr.getTransitionFactory().isSplit()).isTrue();
    ConfigurationTransition transition =
        attr.getTransitionFactory()
            .create(
                AttributeTransitionData.builder().attributes(FakeAttributeMapper.empty()).build());
    assertThat(transition).isEqualTo(splitTransition);
  }

  @Test
  public void testSplitTransitionProvider() {
    TestSplitTransitionProvider splitTransitionProvider = new TestSplitTransitionProvider();
    Attribute attr = attr("foo", LABEL).cfg(splitTransitionProvider).allowedFileTypes().build();
    assertThat(attr.getTransitionFactory().isSplit()).isTrue();
    ConfigurationTransition transition =
        attr.getTransitionFactory()
            .create(
                AttributeTransitionData.builder().attributes(FakeAttributeMapper.empty()).build());
    assertThat(transition).isInstanceOf(TestSplitTransition.class);
  }

  @Test
  public void testExecTransition() {
    Attribute attr =
        attr("foo", LABEL)
            .cfg(ExecutionTransitionFactory.createFactory())
            .allowedFileTypes()
            .build();
    assertThat(attr.getTransitionFactory().isTool()).isTrue();
    assertThat(attr.getTransitionFactory().isSplit()).isFalse();
  }

  private static class TestSplitTransition implements SplitTransition {
    @Override
    public Map<String, BuildOptions> split(
        BuildOptionsView buildOptions, EventHandler eventHandler) {
      return ImmutableMap.of(
          "test0", buildOptions.clone().underlying(), "test1", buildOptions.clone().underlying());
    }
  }

  private static class TestSplitTransitionProvider
      implements TransitionFactory<AttributeTransitionData> {
    @Override
    public SplitTransition create(AttributeTransitionData data) {
      return new TestSplitTransition();
    }

    @Override
    public TransitionType transitionType() {
      return TransitionType.ATTRIBUTE;
    }

    @Override
    public boolean isSplit() {
      return true;
    }
  }

  @Test
  public void allowedRuleClassesAndAllowedRuleClassesWithWarningsCannotOverlap() {
    IllegalStateException e =
        assertThrows(
            IllegalStateException.class,
            () ->
                attr("x", LABEL_LIST)
                    .allowedRuleClasses("foo", "bar", "baz")
                    .allowedRuleClassesWithWarning("bar")
                    .allowedFileTypes()
                    .build());
    assertThat(e).hasMessageThat().contains("may not contain the same rule classes");
  }
}
