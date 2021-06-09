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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.HostTransition;
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

/** Tests of Attribute code. */
@RunWith(JUnit4.class)
public class AttributeTest {

  private void assertDefaultValue(Object expected, Attribute attr) {
    assertThat(attr.getDefaultValue(null)).isEqualTo(expected);
  }

  private void assertType(Type<?> expectedType, Attribute attr) {
    assertThat(attr.getType()).isEqualTo(expectedType);
  }

  @Test
  public void testBasics() throws Exception {
    Attribute attr = attr("foo", Type.INTEGER).mandatory().value(StarlarkInt.of(3)).build();
    assertThat(attr.getName()).isEqualTo("foo");
    assertThat(attr.getDefaultValue(null)).isEqualTo(StarlarkInt.of(3));
    assertThat(attr.getType()).isEqualTo(Type.INTEGER);
    assertThat(attr.isMandatory()).isTrue();
    assertThat(attr.isDocumented()).isTrue();
    attr = attr("$foo", Type.INTEGER).build();
    assertThat(attr.isDocumented()).isFalse();
  }

  @Test
  public void testNonEmptyReqiresListType() throws Exception {
    NullPointerException e =
        assertThrows(
            NullPointerException.class,
            () -> attr("foo", Type.INTEGER).nonEmpty().value(StarlarkInt.of(3)).build());
    assertThat(e).hasMessageThat().isEqualTo("attribute 'foo' must be a list");
  }

  @Test
  public void testNonEmpty() throws Exception {
    Attribute attr = attr("foo", BuildType.LABEL_LIST).nonEmpty().legacyAllowAnyFileType().build();
    assertThat(attr.getName()).isEqualTo("foo");
    assertThat(attr.getType()).isEqualTo(BuildType.LABEL_LIST);
    assertThat(attr.isNonEmpty()).isTrue();
  }

  @Test
  public void testSingleArtifactReqiresLabelType() throws Exception {
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
            .cfg(HostTransition.createFactory())
            .undocumented("")
            .value("y");
    assertThrows(IllegalStateException.class, () -> builder.mandatory());
    assertThrows(IllegalStateException.class, () -> builder.cfg(HostTransition.createFactory()));
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

    Label label = Label.parseAbsolute("//foo:bar", ImmutableMap.of());
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
        Arrays.asList(
            Label.parseAbsolute("//foo:bar", ImmutableMap.of()),
            Label.parseAbsolute("//foo:wiz", ImmutableMap.of()));
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

    Label label = Label.parseAbsolute("//foo:bar", ImmutableMap.of());
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
        Arrays.asList(
            Label.parseAbsolute("//foo:bar", ImmutableMap.of()),
            Label.parseAbsolute("//foo:wiz", ImmutableMap.of()));
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
      assertThat(childAttr1.getAllowedRuleClassesPredicate()).isEqualTo(Predicates.alwaysTrue());
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
      assertThat(childAttr2.getAllowedRuleClassesPredicate())
          .isEqualTo(ruleClasses.asPredicateOfRuleClass());
      assertThat(childAttr2.isMandatory()).isTrue();
      assertThat(childAttr2.isNonEmpty()).isTrue();
      assertThat(childAttr2.getAspects(/* rule= */ null)).hasSize(2);
    }

    // Check if the parent attribute is unchanged
    assertThat(parentAttr.isNonEmpty()).isFalse();
    assertThat(parentAttr.getAllowedRuleClassesPredicate()).isEqualTo(Predicates.alwaysTrue());
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
  public void testSplitTransition() throws Exception {
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
  public void testSplitTransitionProvider() throws Exception {
    TestSplitTransitionProvider splitTransitionProvider = new TestSplitTransitionProvider();
    Attribute attr =
        attr("foo", LABEL).cfg(splitTransitionProvider).allowedFileTypes().build();
    assertThat(attr.getTransitionFactory().isSplit()).isTrue();
    ConfigurationTransition transition =
        attr.getTransitionFactory()
            .create(
                AttributeTransitionData.builder().attributes(FakeAttributeMapper.empty()).build());
    assertThat(transition).isInstanceOf(TestSplitTransition.class);
  }

  @Test
  public void testHostTransition() throws Exception {
    Attribute attr =
        attr("foo", LABEL).cfg(HostTransition.createFactory()).allowedFileTypes().build();
    assertThat(attr.getTransitionFactory().isHost()).isTrue();
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
    public boolean isSplit() {
      return true;
    }
  }

  @Test
  public void allowedRuleClassesAndAllowedRuleClassesWithWarningsCannotOverlap() throws Exception {
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

  private static final Label FAKE_LABEL = Label.parseAbsoluteUnchecked("//fake/label.bzl");

  private static final StarlarkProviderIdentifier STARLARK_P1 =
      StarlarkProviderIdentifier.forKey(new StarlarkProvider.Key(FAKE_LABEL, "STARLARK_P1"));

  private static final StarlarkProviderIdentifier STARLARK_P2 =
      StarlarkProviderIdentifier.forKey(new StarlarkProvider.Key(FAKE_LABEL, "STARLARK_P2"));

  private static final StarlarkProviderIdentifier STARLARK_P3 =
      StarlarkProviderIdentifier.forKey(new StarlarkProvider.Key(FAKE_LABEL, "STARLARK_P3"));

  private static final StarlarkProviderIdentifier STARLARK_P4 =
      StarlarkProviderIdentifier.forKey(new StarlarkProvider.Key(FAKE_LABEL, "STARLARK_P4"));

  @Test
  public void testAttrRequiredAspects_inheritAttrAspects() throws Exception {
    ImmutableList<String> inheritedAttributeAspects1 = ImmutableList.of("attr1", "attr2");
    ImmutableList<String> inheritedAttributeAspects2 = ImmutableList.of("attr3", "attr2");

    Attribute attr =
        attr("x", LABEL)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_1",
                /** inheritedRequiredProviders= */
                ImmutableList.of(),
                inheritedAttributeAspects1)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_2",
                /** inheritedRequiredProviders= */
                ImmutableList.of(),
                inheritedAttributeAspects2)
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);
    AspectDescriptor aspectDescriptor = aspects.get(0).getDescriptor();
    assertThat(aspectDescriptor.getInheritedAttributeAspects())
        .containsExactly("attr1", "attr2", "attr3");
  }

  @Test
  public void testAttrRequiredAspects_inheritRequiredProviders() throws Exception {
    ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders1 =
        ImmutableList.of(ImmutableSet.of(STARLARK_P1), ImmutableSet.of(STARLARK_P2, STARLARK_P3));
    ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders2 =
        ImmutableList.of(ImmutableSet.of(STARLARK_P4), ImmutableSet.of(STARLARK_P2, STARLARK_P3));

    Attribute attr =
        attr("x", LABEL)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_1",
                inheritedRequiredProviders1,
                /** inheritedAttributeAspects= */
                ImmutableList.of())
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_2",
                inheritedRequiredProviders2,
                /** inheritedAttributeAspects= */
                ImmutableList.of())
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);

    RequiredProviders actualInheritedRequiredProviders =
        aspects.get(0).getDescriptor().getInheritedRequiredProviders();
    AdvertisedProviderSet expectedOkSet1 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P1).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet1)).isTrue();

    AdvertisedProviderSet expectedOkSet2 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P4).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet2)).isTrue();

    AdvertisedProviderSet expectedOkSet3 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P2).addStarlark(STARLARK_P3).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet3)).isTrue();

    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(AdvertisedProviderSet.ANY)).isTrue();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(AdvertisedProviderSet.EMPTY))
        .isFalse();
  }

  @Test
  public void testAttrRequiredAspects_aspectAlreadyExists_inheritAttrAspects() throws Exception {
    ImmutableList<String> inheritedAttributeAspects = ImmutableList.of("attr1", "attr2");

    Attribute attr =
        attr("x", LABEL)
            .aspect(TestAspects.SIMPLE_ASPECT)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect",
                /** inheritedRequiredProviders = */
                ImmutableList.of(),
                inheritedAttributeAspects)
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);
    AspectDescriptor aspectDescriptor = aspects.get(0).getDescriptor();
    assertThat(aspectDescriptor.getInheritedAttributeAspects()).containsExactly("attr1", "attr2");
  }

  @Test
  public void testAttrRequiredAspects_aspectAlreadyExists_inheritRequiredProviders()
      throws Exception {
    ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders =
        ImmutableList.of(ImmutableSet.of(STARLARK_P1), ImmutableSet.of(STARLARK_P2, STARLARK_P3));

    Attribute attr =
        attr("x", LABEL)
            .aspect(TestAspects.SIMPLE_ASPECT)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect",
                inheritedRequiredProviders,
                /** inheritedAttributeAspects= */
                ImmutableList.of())
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);

    RequiredProviders actualInheritedRequiredProviders =
        aspects.get(0).getDescriptor().getInheritedRequiredProviders();
    AdvertisedProviderSet expectedOkSet1 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P1).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet1)).isTrue();

    AdvertisedProviderSet expectedOkSet2 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P4).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet2)).isFalse();

    AdvertisedProviderSet expectedOkSet3 =
        AdvertisedProviderSet.builder().addStarlark(STARLARK_P2).addStarlark(STARLARK_P3).build();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(expectedOkSet3)).isTrue();

    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(AdvertisedProviderSet.ANY)).isTrue();
    assertThat(actualInheritedRequiredProviders.isSatisfiedBy(AdvertisedProviderSet.EMPTY))
        .isFalse();
  }

  @Test
  public void testAttrRequiredAspects_inheritAllAttrAspects() throws Exception {
    ImmutableList<String> inheritedAttributeAspects1 = ImmutableList.of("attr1", "attr2");
    ImmutableList<String> inheritedAttributeAspects2 = ImmutableList.of("*");

    Attribute attr =
        attr("x", LABEL)
            .aspect(TestAspects.SIMPLE_ASPECT)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_1",
                /** inheritedRequiredProviders = */
                ImmutableList.of(),
                inheritedAttributeAspects1)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_2",
                /** inheritedRequiredProviders = */
                ImmutableList.of(),
                inheritedAttributeAspects2)
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);
    AspectDescriptor aspectDescriptor = aspects.get(0).getDescriptor();
    assertThat(aspectDescriptor.getInheritedAttributeAspects()).isNull();
  }

  @Test
  public void testAttrRequiredAspects_inheritAllRequiredProviders() throws Exception {
    ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders1 =
        ImmutableList.of();
    ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders2 =
        ImmutableList.of(ImmutableSet.of(STARLARK_P4), ImmutableSet.of(STARLARK_P2, STARLARK_P3));

    Attribute attr =
        attr("x", LABEL)
            .aspect(TestAspects.SIMPLE_ASPECT)
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_1",
                inheritedRequiredProviders1,
                /** inheritedAttributeAspects= */
                ImmutableList.of())
            .aspect(
                TestAspects.SIMPLE_ASPECT,
                "base_aspect_2",
                inheritedRequiredProviders2,
                /** inheritedAttributeAspects= */
                ImmutableList.of())
            .allowedFileTypes()
            .build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);
    AspectDescriptor aspectDescriptor = aspects.get(0).getDescriptor();
    assertThat(aspectDescriptor.getInheritedRequiredProviders())
        .isEqualTo(RequiredProviders.acceptAnyBuilder().build());
  }

  @Test
  public void testAttrRequiredAspects_defaultInheritedRequiredProvidersAndAttrAspects()
      throws Exception {
    Attribute attr = attr("x", LABEL).aspect(TestAspects.SIMPLE_ASPECT).allowedFileTypes().build();

    ImmutableList<Aspect> aspects = attr.getAspects(null);
    assertThat(aspects).hasSize(1);
    AspectDescriptor aspectDescriptor = aspects.get(0).getDescriptor();
    assertThat(aspectDescriptor.getInheritedAttributeAspects()).isEmpty();
    assertThat(aspectDescriptor.getInheritedRequiredProviders()).isNull();
  }
}
