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
package com.google.devtools.build.lib.analysis.util;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredNativeAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.List;

/**
 * Various rule and aspect classes that aid in testing the aspect machinery.
 *
 * <p>These are mostly used in {@link com.google.devtools.build.lib.analysis.DependencyResolverTest}
 * and {@link com.google.devtools.build.lib.analysis.AspectTest}.
 */
public class TestAspects {

  public static final LateBoundLabel EMPTY_LATE_BOUND_LABEL = new LateBoundLabel<Object>() {
    @Override
    public Label getDefault(Rule rule, Object configuration) {
      return null;
    }
  };

  /**
   * A transitive info provider for collecting aspects in the transitive closure. Created by
   * aspects.
   */
  @Immutable
  public static final class AspectInfo implements TransitiveInfoProvider {
    private final NestedSet<String> data;

    public AspectInfo(NestedSet<String> data) {
      this.data = data;
    }

    public NestedSet<String> getData() {
      return data;
    }
  }

  /**
   * A transitive info provider for collecting aspects in the transitive closure. Created by
   * rules.
   */
  @Immutable
  public static final class RuleInfo implements TransitiveInfoProvider {
    private final NestedSet<String> data;

    public RuleInfo(NestedSet<String> data) {
      this.data = data;
    }

    public NestedSet<String> getData() {
      return data;
    }
  }

  /**
   * A very simple provider used in tests that check whether the logic that attaches aspects
   * depending on whether a configured target has a provider works or not.
   */
  @Immutable
  public static final class RequiredProvider implements TransitiveInfoProvider {
  }

  private static NestedSet<String> collectAspectData(String me, RuleContext ruleContext) {
    NestedSetBuilder<String> result = new NestedSetBuilder<>(Order.STABLE_ORDER);
    result.add(me);
    for (AspectInfo dep : ruleContext.getPrerequisites("foo", Mode.TARGET, AspectInfo.class)) {
      result.addTransitive(dep.getData());
    }

    return result.build();
  }

  /**
   * A simple rule configured target factory that is used in all the mock rules in this class.
   */
  public static class DummyRuleFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {

      RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RuleInfo.class,
              new RuleInfo(collectAspectData("rule " + ruleContext.getLabel(), ruleContext)))
          .setFilesToBuild(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER))
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

      if (ruleContext.getRule().getRuleClassObject().getName().equals("honest")) {
        builder.addProvider(RequiredProvider.class, new RequiredProvider());
      }

      return builder.build();
    }
  }

  /**
   * A base class for mock aspects to reduce boilerplate.
   */
  public abstract static class BaseAspect implements ConfiguredNativeAspectFactory {
    @Override
    public ConfiguredAspect create(
        ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters) {
      String information = parameters.isEmpty()
          ? ""
          : " data " + Iterables.getFirst(parameters.getAttribute("baz"), null);
      return new ConfiguredAspect.Builder(getClass().getName(), ruleContext)
          .addProvider(
              AspectInfo.class,
              new AspectInfo(
                  collectAspectData("aspect " + ruleContext.getLabel() + information, ruleContext)))
          .build();
    }
  }

  private static final AspectDefinition SIMPLE_ASPECT =
      new AspectDefinition.Builder("simple").build();

  /**
   * A very simple aspect.
   */
  public static class SimpleAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return SIMPLE_ASPECT;
    }
  }

  private static final AspectDefinition EXTRA_ATTRIBUTE_ASPECT =
      new AspectDefinition.Builder("extra_attribute")
          .add(attr("$dep", LABEL).value(Label.parseAbsoluteUnchecked("//extra:extra")))
          .build();

  private static final AspectDefinition EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER =
      new AspectDefinition.Builder("extra_attribute_with_provider")
          .add(attr("$dep", LABEL).value(Label.parseAbsoluteUnchecked("//extra:extra")))
          .requireProvider(RequiredProvider.class)
          .build();

  /**
   * An aspect that defines its own implicit attribute.
   */
  public static class ExtraAttributeAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return EXTRA_ATTRIBUTE_ASPECT;
    }
  }

  private static final AspectDefinition ATTRIBUTE_ASPECT = new AspectDefinition.Builder("attribute")
      .attributeAspect("foo", AttributeAspect.class)
      .build();

  /**
   * An aspect that requires aspects on the attributes of rules it attaches to.
   */
  public static class AttributeAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ATTRIBUTE_ASPECT;
    }
  }

  /**
   * An aspect that defines its own implicit attribute and requires provider.
   */
  public static class ExtraAttributeAspectRequiringProvider extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER;
    }
  }

  public static class AspectRequiringProvider extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ASPECT_REQUIRING_PROVIDER;
    }
  }

  /**
   * An aspect that has a definition depending on parameters provided by originating rule.
   */
  public static class ParametrizedDefinitionAspect implements ConfiguredNativeAspectFactory {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder builder =
          new AspectDefinition.Builder("parametrized_definition_aspect")
              .attributeAspect("foo", ParametrizedDefinitionAspect.class);
      ImmutableCollection<String> baz = aspectParameters.getAttribute("baz");
      if (baz != null) {
        try {
          builder.add(attr("$dep", LABEL).value(Label.parseAbsolute(baz.iterator().next())));
        } catch (LabelSyntaxException e) {
          throw new IllegalStateException();
        }
      }
      return builder.build();
    }

    @Override
    public ConfiguredAspect create(
        ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters) {
      StringBuilder information = new StringBuilder("aspect " + ruleContext.getLabel());
      if (!parameters.isEmpty()) {
        information.append(" data " + Iterables.getFirst(parameters.getAttribute("baz"), null));
        information.append(" ");
      }
      List<? extends TransitiveInfoCollection> deps =
          ruleContext.getPrerequisites("$dep", Mode.TARGET);
      information.append("$dep:[");
      for (TransitiveInfoCollection dep : deps) {
        information.append(" ");
        information.append(dep.getLabel());
      }
      information.append("]");
      return new ConfiguredAspect.Builder(getClass().getName(), ruleContext)
          .addProvider(
              AspectInfo.class,
              new AspectInfo(collectAspectData(information.toString(), ruleContext)))
          .build();
    }
  }


  private static final AspectDefinition ASPECT_REQUIRING_PROVIDER =
      new AspectDefinition.Builder("requiring_provider")
          .requireProvider(RequiredProvider.class)
          .build();

  /**
   * An aspect that raises an error.
   */
  public static class ErrorAspect implements ConfiguredNativeAspectFactory {
    @Override
    public ConfiguredAspect create(
        ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters) {
      ruleContext.ruleError("Aspect error");
      return null;
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ERROR_ASPECT;
    }
  }

  private static final AspectDefinition ERROR_ASPECT = new AspectDefinition.Builder("error")
      .attributeAspect("bar", ErrorAspect.class)
      .build();

  /**
   * A common base rule for mock rules in this class to reduce boilerplate.
   *
   * <p>It has a few common attributes because internal Blaze machinery assumes the presence of
   * these.
   */
  public static class BaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("testonly", BOOLEAN).nonconfigurable("test").value(false))
          .add(attr("deprecation", STRING).nonconfigurable("test"))
          .add(attr("tags", STRING_LIST))
          .add(attr("visibility", NODEP_LABEL_LIST).orderIndependent().cfg(HOST)
              .nonconfigurable("test"))
          .add(attr(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
              .allowedFileTypes(FileTypeSet.NO_FILE))
          .add(attr(RuleClass.RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
              .allowedFileTypes(FileTypeSet.NO_FILE))
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("base")
          .factoryClass(DummyRuleFactory.class)
          .build();
    }
  }

  /**
   * A rule that defines an aspect on one of its attributes.
   */
  public static class AspectRequiringRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(SimpleAspect.class))
          .add(attr("bar", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(SimpleAspect.class))
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("aspect")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that defines an {@link AspectRequiringProvider} on one of its attributes.
   */
  public static class AspectRequiringProviderRule implements RuleDefinition {

    private static final class TestAspectParametersExtractor implements 
        Function<Rule, AspectParameters> {
      @Override
      public AspectParameters apply(Rule rule) {
        if (rule.isAttrDefined("baz", STRING)) {
          String value = rule.getAttributeContainer().getAttr("baz").toString();
          if (!value.equals("")) {
            return new AspectParameters.Builder().addAttribute("baz", value).build();
          }
        }
        return AspectParameters.EMPTY;
      }
    }

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(AspectRequiringProvider.class, new TestAspectParametersExtractor()))
          .add(attr("baz", STRING))
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("aspect_requiring_provider")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that defines an {@link ParametrizedDefinitionAspect} on one of its attributes.
   */
  public static class ParametrizedDefinitionAspectRule implements RuleDefinition {

    private static final class TestAspectParametersExtractor
        implements Function<Rule, AspectParameters> {
      @Override
      public AspectParameters apply(Rule rule) {
        if (rule.isAttrDefined("baz", STRING)) {
          String value = rule.getAttributeContainer().getAttr("baz").toString();
          if (!value.equals("")) {
            return new AspectParameters.Builder().addAttribute("baz", value).build();
          }
        }
        return AspectParameters.EMPTY;
      }
    }

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(
              attr("foo", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .aspect(ParametrizedDefinitionAspect.class, new TestAspectParametersExtractor()))
          .add(attr("baz", STRING))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("parametrized_definition_aspect")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that defines an {@link ExtraAttributeAspectRequiringProvider} on one of its attributes.
   */
  public static class ExtraAttributeAspectRequiringProviderRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(ExtraAttributeAspectRequiringProvider.class))
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("extra_attribute_aspect_requiring_provider")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that defines an {@link AspectRequiringProvider} on one of its attributes.
   */
  public static class ErrorAspectRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(ErrorAspect.class))
          .add(attr("bar", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("error_aspect")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A simple rule that has an attribute.
   */
  public static class SimpleRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("simple")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that advertises a provider but doesn't implement it.
   */
  public static class LiarRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
          .advertiseProvider(RequiredProvider.class)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("liar")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }

  /**
   * A rule that advertises a provider and implements it.
   */
  public static class HonestRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
          .advertiseProvider(RequiredProvider.class)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("honest")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(BaseRule.class)
          .build();
    }
  }
}
