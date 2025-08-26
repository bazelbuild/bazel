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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AspectContext;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.AspectPropagationEdgesSupplier;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.Attribute.LabelListLateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkNativeAspect;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.io.Serializable;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.syntax.Location;

/**
 * Various rule and aspect classes that aid in testing the aspect machinery.
 *
 * <p>These are mostly used in {@link com.google.devtools.build.lib.analysis.DependencyResolverTest}
 * and {@link com.google.devtools.build.lib.analysis.AspectTest}.
 */
public class TestAspects {

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
   * A transitive info provider used as sentinel. Created by aspects.
   */
  @Immutable
  public static final class FooProvider implements TransitiveInfoProvider {
  }

  /**
   * A transitive info provider used as sentinel. Created by aspects.
   */
  @Immutable
  public static final class BarProvider implements TransitiveInfoProvider {
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

  private static final Label FAKE_LABEL = Label.parseCanonicalUnchecked("//fake/label.bzl");

  public static final StarlarkProvider.Key REQUIRED_PROVIDER_KEY =
      new StarlarkProvider.Key(keyForBuild(FAKE_LABEL), "RequiredProvider");
  public static final StarlarkProvider.Key REQUIRED_PROVIDER2_KEY =
      new StarlarkProvider.Key(keyForBuild(FAKE_LABEL), "RequiredProvider2");

  /**
   * A very simple provider used in tests that check whether the logic that attaches aspects
   * depending on whether a configured target has a provider works or not.
   */
  @SerializationConstant @VisibleForSerialization
  static final StarlarkProvider REQUIRED_PROVIDER =
      StarlarkProvider.builder(Location.BUILTIN).buildExported(REQUIRED_PROVIDER_KEY);

  /**
   * Another very simple provider used in tests that check whether the logic that attaches aspects
   * depending on whether a configured target has a provider works or not.
   */
  @SerializationConstant @VisibleForSerialization
  static final StarlarkProvider REQUIRED_PROVIDER2 =
      StarlarkProvider.builder(Location.BUILTIN).buildExported(REQUIRED_PROVIDER2_KEY);

  private static NestedSet<String> collectAspectData(String me, RuleContext ruleContext) {
    NestedSetBuilder<String> result = NestedSetBuilder.newBuilder(Order.STABLE_ORDER);
    result.add(me);

    Iterable<String> attributeNames = ruleContext.attributes().getAttributeNames();
    for (String attributeName : attributeNames) {
      Type<?> attributeType = ruleContext.attributes().getAttributeType(attributeName);
      if (!LABEL.equals(attributeType) && !LABEL_LIST.equals(attributeType)) {
        continue;
      }
      Iterable<AspectInfo> prerequisites =
          ruleContext
              .getRulePrerequisitesCollection()
              .getPrerequisites(attributeName, AspectInfo.class);
      for (AspectInfo prerequisite : prerequisites) {
        result.addTransitive(prerequisite.getData());
      }
    }
    return result.build();
  }

  public static class FileProviderForwardingRuleFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(ruleContext.getPrerequisite("dep", FileProvider.class).getFilesToBuild())
          .setRunfilesSupport(null, null)
          .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }

  /**
   * A simple rule configured target factory that is used in all the mock rules in this class.
   */
  public static class DummyRuleFactory implements RuleConfiguredTargetFactory {
    @Override
    @Nullable
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {

      RuleConfiguredTargetBuilder builder =
          new RuleConfiguredTargetBuilder(ruleContext)
              .addProvider(
                  new RuleInfo(collectAspectData("rule " + ruleContext.getLabel(), ruleContext)))
              .setFilesToBuild(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER))
              .setRunfilesSupport(null, null)
              .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

      if (ruleContext.getRule().getRuleClassObject().getName().equals("honest")) {
        builder.addStarlarkDeclaredProvider(
            StarlarkInfo.create(REQUIRED_PROVIDER, ImmutableMap.of(), null));
      }

      return builder.build();
    }
  }

  /** A simple rule configured target factory that exports provider {@link REQUIRED_PROVIDER2}. */
  public static class DummyRuleFactory2 implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(
              new RuleInfo(collectAspectData("rule " + ruleContext.getLabel(), ruleContext)))
          .setFilesToBuild(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER))
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .addStarlarkDeclaredProvider(
              StarlarkInfo.create(REQUIRED_PROVIDER, ImmutableMap.of(), null))
          .addStarlarkDeclaredProvider(
              StarlarkInfo.create(REQUIRED_PROVIDER2, ImmutableMap.of(), null))
          .build();
    }
  }

  /**
   * A simple rule configured target factory that expects different providers added through
   * different aspects.
   */
  public static class MultiAspectRuleFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      TransitiveInfoCollection fooAttribute =
          ruleContext.getRulePrerequisitesCollection().getPrerequisite("foo");
      TransitiveInfoCollection barAttribute =
          ruleContext.getRulePrerequisitesCollection().getPrerequisite("bar");

      NestedSetBuilder<String> infoBuilder = NestedSetBuilder.<String>stableOrder();

      if (fooAttribute.getProvider(FooProvider.class) != null) {
        infoBuilder.add("foo");
      }
      if (barAttribute.getProvider(BarProvider.class) != null) {
        infoBuilder.add("bar");
      }

      RuleConfiguredTargetBuilder builder =
          new RuleConfiguredTargetBuilder(ruleContext)
              .addProvider(
                  new RuleInfo(infoBuilder.build()))
              .setFilesToBuild(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER))
              .setRunfilesSupport(null, null)
              .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

      return builder.build();
    }
  }

  /**
   * A base class for mock aspects to reduce boilerplate.
   */
  public abstract static class BaseAspect extends NativeAspectClass
    implements ConfiguredAspectFactory {
    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      String information = parameters.isEmpty()
          ? ""
          : " data " + Iterables.getFirst(parameters.getAttribute("baz"), null);
      return new ConfiguredAspect.Builder(ruleContext)
          .addProvider(
              new AspectInfo(
                  collectAspectData("aspect " + ruleContext.getLabel() + information, ruleContext)))
          .build();
    }
  }

  public static class FileProviderAspect extends BaseAspect {
    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      Artifact artifact = ruleContext.getBinArtifact("file_provider_aspect_file");
      ruleContext.registerAction(FileWriteAction.create(ruleContext, artifact, "empty", false));
      return new ConfiguredAspect.Builder(ruleContext)
          .addProvider(FileProvider.of(NestedSetBuilder.create(Order.STABLE_ORDER, artifact)))
          .build();
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return SIMPLE_ASPECT_DEFINITION;
    }
  }

  public static final SimpleAspect SIMPLE_ASPECT = new SimpleAspect();
  public static final FileProviderAspect FILE_PROVIDER_ASPECT = new FileProviderAspect();
  public static final FooProviderAspect FOO_PROVIDER_ASPECT = new FooProviderAspect();
  public static final BarProviderAspect BAR_PROVIDER_ASPECT = new BarProviderAspect();
  public static final SimpleStarlarkNativeAspect SIMPLE_STARLARK_NATIVE_ASPECT =
      new SimpleStarlarkNativeAspect();
  public static final ParametrizedAspectWithProvider
      PARAMETRIZED_STARLARK_NATIVE_ASPECT_WITH_PROVIDER = new ParametrizedAspectWithProvider();
  public static final StarlarkNativeAspectWithProvider STARLARK_NATIVE_ASPECT_WITH_PROVIDER =
      new StarlarkNativeAspectWithProvider();

  private static final AspectDefinition SIMPLE_ASPECT_DEFINITION =
      new AspectDefinition.Builder(SIMPLE_ASPECT).build();
  private static final AspectDefinition FOO_PROVIDER_ASPECT_DEFINITION =
      new AspectDefinition.Builder(FOO_PROVIDER_ASPECT).build();
  private static final AspectDefinition BAR_PROVIDER_ASPECT_DEFINITION =
      new AspectDefinition.Builder(BAR_PROVIDER_ASPECT).build();
  private static final AspectDefinition SIMPLE_STARLARK_NATIVE_ASPECT_DEFINITION =
      new AspectDefinition.Builder(SIMPLE_STARLARK_NATIVE_ASPECT).build();

  /** Simple StarlarkNativeAspect */
  public static class SimpleStarlarkNativeAspect extends StarlarkNativeAspect
      implements ConfiguredAspectFactory {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return SIMPLE_STARLARK_NATIVE_ASPECT_DEFINITION;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      return new ConfiguredAspect.Builder(ruleContext).addProvider(new FooProvider()).build();
    }
  }

  /**
   * A very simple aspect.
   */
  public static class SimpleAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return SIMPLE_ASPECT_DEFINITION;
    }
  }

  /**
   * A simple aspect that propagates a FooProvider provider.
   */
  public static class FooProviderAspect extends NativeAspectClass
      implements ConfiguredAspectFactory {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return FOO_PROVIDER_ASPECT_DEFINITION;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      return new ConfiguredAspect.Builder(ruleContext).addProvider(new FooProvider()).build();
    }
  }

  /** A simple aspect that propagates a BarProvider provider. */
  public static class BarProviderAspect extends NativeAspectClass
      implements ConfiguredAspectFactory {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return BAR_PROVIDER_ASPECT_DEFINITION;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      return new ConfiguredAspect.Builder(ruleContext).addProvider(new BarProvider()).build();
    }
  }

  public static final ExtraAttributeAspect EXTRA_ATTRIBUTE_ASPECT =
      new ExtraAttributeAspect(/*depLabel=*/ "//extra", /*applyToFiles=*/ false);

  private static final ExtraAttributeAspectRequiringProvider
    EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER = new ExtraAttributeAspectRequiringProvider();
  private static final AspectDefinition EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER_DEFINITION =
      new AspectDefinition.Builder(EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER)
          .add(attr("$dep", LABEL).value(Label.parseCanonicalUnchecked("//extra:extra")))
          .requireStarlarkProviders(StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY))
          .build();

  /** An aspect that defines its own implicit attribute. */
  public static class ExtraAttributeAspect extends BaseAspect {

    /** Test provider which includes the {@code dep} label. */
    @SerializationConstant
    public static final StarlarkProvider PROVIDER =
        StarlarkProvider.builder(Location.BUILTIN)
            .buildExported(new StarlarkProvider.Key(keyForBuild(FAKE_LABEL), "Provider"));

    private final Label depLabel;
    private final boolean applyToFiles;
    private final StarlarkProviderIdentifier[] requiredAspectProviders;

    public ExtraAttributeAspect(
        String depLabel,
        boolean applyToFiles,
        StarlarkProviderIdentifier... requiredAspectProviders) {
      this.depLabel = Label.parseCanonicalUnchecked(depLabel);
      this.applyToFiles = applyToFiles;
      this.requiredAspectProviders = requiredAspectProviders;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      TransitiveInfoCollection dep =
          ((AspectContext) ruleContext)
              .getMainAspectPrerequisitesCollection()
              .getPrerequisite("$dep");
      if (dep == null) {
        ruleContext.attributeError("$dep", "$dep attribute not resolved");
        return ConfiguredAspect.builder(ruleContext).build();
      }
      try {
        return ConfiguredAspect.builder(ruleContext)
            .addStarlarkDeclaredProvider(
                StarlarkInfo.create(
                    PROVIDER, ImmutableMap.of("label", dep.getLabel().getCanonicalForm()), null))
            .build();
      } catch (EvalException e) {
        throw new IllegalStateException(e);
      }
    }

    @Override
    public String getName() {
      return String.format("%s_%s_%s", super.getName(), depLabel.getCanonicalForm(), applyToFiles);
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder aspectDefinition =
          new AspectDefinition.Builder(this)
              .add(attr("$dep", LABEL).value(depLabel))
              .applyToFiles(applyToFiles)
              .advertiseProvider(
                  ImmutableList.of(StarlarkProviderIdentifier.forKey(PROVIDER.getKey())));

      if (requiredAspectProviders.length > 0) {
        aspectDefinition.requireAspectsWithProviders(
            ImmutableList.of(ImmutableSet.copyOf(requiredAspectProviders)));
      }

      return aspectDefinition.build();
    }
  }

  /**
   * An aspect that applies to output files and propagates to toolchain dependencies and attribute
   * dependencies.
   */
  public static class DepsVisitingFileAspect extends BaseAspect {

    /** Test provider which includes the base target label. */
    @SerializationConstant
    public static final StarlarkProvider PROVIDER =
        StarlarkProvider.builder(Location.BUILTIN)
            .buildExported(new StarlarkProvider.Key(keyForBuild(FAKE_LABEL), "AspectProvider"));

    private String depAttr;
    private String toolchainType;

    public DepsVisitingFileAspect(String depAttr, String toolchainType) {
      this.depAttr = depAttr;
      this.toolchainType = toolchainType;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      try {
        return ConfiguredAspect.builder(ruleContext)
            .addStarlarkDeclaredProvider(
                StarlarkInfo.create(
                    PROVIDER, ImmutableMap.of("val", ct.getLabel().getCanonicalForm()), null))
            .build();
      } catch (EvalException e) {
        throw new IllegalStateException(e);
      }
    }

    @Override
    public String getName() {
      return String.format("%s_%s_%s", super.getName(), depAttr, toolchainType);
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder aspectDefinition =
          new AspectDefinition.Builder(this)
              .applyToFiles(true)
              .propagateToAttributes(createAttrAspects(ImmutableList.of(depAttr)))
              .propagateToToolchainsTypes(createToolchainsAspects(ImmutableList.of(toolchainType)));

      return aspectDefinition.build();
    }
  }

  /** An aspect that defines its own implicit attribute, requiring PackageSpecificationProvider. */
  public static class PackageGroupAttributeAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return PACKAGE_GROUP_ATTRIBUTE_ASPECT_DEFINITION;
    }
  }

  public static final PackageGroupAttributeAspect PACKAGE_GROUP_ATTRIBUTE_ASPECT =
      new PackageGroupAttributeAspect();
  private static final AspectDefinition PACKAGE_GROUP_ATTRIBUTE_ASPECT_DEFINITION =
      new AspectDefinition.Builder(PACKAGE_GROUP_ATTRIBUTE_ASPECT)
          .add(
              attr("$dep", LABEL)
                  .value(Label.parseCanonicalUnchecked("//extra:extra"))
                  .mandatoryBuiltinProviders(ImmutableList.of(PackageSpecificationProvider.class)))
          .build();

  public static final ComputedAttributeAspect COMPUTED_ATTRIBUTE_ASPECT =
      new ComputedAttributeAspect();
  private static final AspectDefinition COMPUTED_ATTRIBUTE_ASPECT_DEFINITION =
      new AspectDefinition.Builder(COMPUTED_ATTRIBUTE_ASPECT).build();

  /** An aspect that defines its own computed default attribute. */
  public static class ComputedAttributeAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return COMPUTED_ATTRIBUTE_ASPECT_DEFINITION;
    }
  }

  public static final AttributeAspect ATTRIBUTE_ASPECT = new AttributeAspect();
  private static final AspectDefinition ATTRIBUTE_ASPECT_DEFINITION =
      new AspectDefinition.Builder(ATTRIBUTE_ASPECT)
          .propagateToAttributes(createAttrAspects(ImmutableList.of("foo")))
          .build();

  /**
   * An aspect that propagates along all attributes.
   */
  public static class AllAttributesAspect extends BaseAspect {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ALL_ATTRIBUTES_ASPECT_DEFINITION;
    }
  }
  public static final NativeAspectClass ALL_ATTRIBUTES_ASPECT = new AllAttributesAspect();
  private static final AspectDefinition ALL_ATTRIBUTES_ASPECT_DEFINITION =
      new AspectDefinition.Builder(ALL_ATTRIBUTES_ASPECT)
          .propagateToAttributes(createAttrAspects(ImmutableList.of("*")))
          .build();

  /** An aspect that propagates along all attributes and has a tool dependency. */
  public static class AllAttributesWithToolAspect extends BaseAspect {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ALL_ATTRIBUTES_WITH_TOOL_ASPECT_DEFINITION;
    }
  }

  public static final NativeAspectClass ALL_ATTRIBUTES_WITH_TOOL_ASPECT =
      new AllAttributesWithToolAspect();
  private static final AspectDefinition ALL_ATTRIBUTES_WITH_TOOL_ASPECT_DEFINITION =
      new AspectDefinition.Builder(ALL_ATTRIBUTES_WITH_TOOL_ASPECT)
          .propagateToAttributes(createAttrAspects(ImmutableList.of("*")))
          .add(
              attr("$tool", BuildType.LABEL)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .value(Label.parseCanonicalUnchecked("//a:tool")))
          .build();

  /**
   * An aspect that requires aspects on the attributes of rules it attaches to.
   */
  public static class AttributeAspect extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ATTRIBUTE_ASPECT_DEFINITION;
    }
  }

  /**
   * An aspect that defines its own implicit attribute and requires provider.
   */
  public static class ExtraAttributeAspectRequiringProvider extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER_DEFINITION;
    }
  }

  public static class AspectRequiringProvider extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ASPECT_REQUIRING_PROVIDER_DEFINITION;
    }
  }

  /**
   * An aspect that requires provider sets {{@link RequiredProvider}} and
   * {{@link RequiredProvider2}}.
   */
  public static class AspectRequiringProviderSets extends BaseAspect {
    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ASPECT_REQUIRING_PROVIDER_SETS_DEFINITION;
    }
  }

  /** A native aspect exposed to Starlark and advertises a simple provider. */
  public static class StarlarkNativeAspectWithProvider extends StarlarkNativeAspect
      implements ConfiguredAspectFactory {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder builder =
          new AspectDefinition.Builder(STARLARK_NATIVE_ASPECT_WITH_PROVIDER);
      builder.requireStarlarkProviders(StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY));
      return builder.build();
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      return new ConfiguredAspect.Builder(ruleContext)
          .addStarlarkTransitiveInfo("native_aspect_prov", "native_aspect_val")
          .build();
    }
  }

  /**
   * An aspect that has a definition depending on parameters provided by originating rule and
   * advertises a simple provider.
   */
  public static class ParametrizedAspectWithProvider extends StarlarkNativeAspect
      implements ConfiguredAspectFactory {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder builder =
          new AspectDefinition.Builder(PARAMETRIZED_STARLARK_NATIVE_ASPECT_WITH_PROVIDER);
      ImmutableCollection<String> aspectAttr = aspectParameters.getAttribute("aspect_attr");
      if (aspectAttr != null) {
        builder.add(
            attr("aspect_attr", Type.STRING)
                .allowedValues(new AllowedValueSet("v1", "v2"))
                .value(aspectAttr.iterator().next()));
      }
      return builder.build();
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      return new ConfiguredAspect.Builder(ruleContext).addProvider(new FooProvider()).build();
    }

    @Override
    public Function<Rule, AspectParameters> getDefaultParametersExtractor() {
      return (Function<Rule, AspectParameters> & Serializable)
          (@Nullable Rule rule) -> {
            AttributeMap attributes = RawAttributeMapper.of(rule);
            return new AspectParameters.Builder()
                .addAttribute("aspect_attr", attributes.get("aspect_attr", Type.STRING))
                .build();
          };
    }

    @Override
    public ImmutableSet<String> getParamAttributes() {
      return ImmutableSet.of("aspect_attr");
    }
  }

  /**
   * An aspect that has a definition depending on parameters provided by originating rule.
   */
  public static class ParametrizedDefinitionAspect extends NativeAspectClass
    implements ConfiguredAspectFactory {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      AspectDefinition.Builder builder =
          new AspectDefinition.Builder(PARAMETRIZED_DEFINITION_ASPECT)
              .propagateToAttributes(createAttrAspects(ImmutableList.of("foo")));
      ImmutableCollection<String> baz = aspectParameters.getAttribute("baz");
      if (baz != null) {
        try {
          builder.add(attr("$dep", LABEL).value(Label.parseCanonical(baz.iterator().next())));
        } catch (LabelSyntaxException e) {
          throw new IllegalStateException(e);
        }
      }
      return builder.build();
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      StringBuilder information = new StringBuilder("aspect " + ruleContext.getLabel());
      if (!parameters.isEmpty()) {
        information.append(" data " + Iterables.getFirst(parameters.getAttribute("baz"), null));
        information.append(" ");
      }
      List<? extends TransitiveInfoCollection> deps =
          ((AspectContext) ruleContext)
              .getMainAspectPrerequisitesCollection()
              .getPrerequisites("$dep");
      information.append("$dep:[");
      for (TransitiveInfoCollection dep : deps) {
        information.append(" ");
        information.append(dep.getLabel());
      }
      information.append("]");
      return new ConfiguredAspect.Builder(ruleContext)
          .addProvider(new AspectInfo(collectAspectData(information.toString(), ruleContext)))
          .build();
    }
  }

  static final ParametrizedDefinitionAspect PARAMETRIZED_DEFINITION_ASPECT =
      new ParametrizedDefinitionAspect();

  static final AspectRequiringProvider ASPECT_REQUIRING_PROVIDER = new AspectRequiringProvider();
  static final AspectRequiringProviderSets ASPECT_REQUIRING_PROVIDER_SETS =
      new AspectRequiringProviderSets();
  private static final AspectDefinition ASPECT_REQUIRING_PROVIDER_DEFINITION =
      new AspectDefinition.Builder(ASPECT_REQUIRING_PROVIDER)
          .requireStarlarkProviders(StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY))
          .propagateToAttributes(createAttrAspects(ImmutableList.of("foo")))
          .build();
  private static final AspectDefinition ASPECT_REQUIRING_PROVIDER_SETS_DEFINITION =
      new AspectDefinition.Builder(ASPECT_REQUIRING_PROVIDER_SETS)
          .requireStarlarkProviderSets(
              ImmutableList.of(
                  ImmutableSet.of(StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY)),
                  ImmutableSet.of(StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER2_KEY))))
          .build();

  /**
   * An aspect that prints a warning.
   */
  public static class WarningAspect extends NativeAspectClass
    implements ConfiguredAspectFactory {

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      ruleContext.ruleWarning("Aspect warning on " + targetLabel);
      return new ConfiguredAspect.Builder(ruleContext).build();
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return WARNING_ASPECT_DEFINITION;
    }
  }

  public static final WarningAspect WARNING_ASPECT = new WarningAspect();
  private static final AspectDefinition WARNING_ASPECT_DEFINITION =
      new AspectDefinition.Builder(WARNING_ASPECT)
          .propagateToAttributes(createAttrAspects(ImmutableList.of("bar")))
          .build();

  /**
   * An aspect that raises an error.
   */
  public static class ErrorAspect extends NativeAspectClass
    implements ConfiguredAspectFactory {

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext ruleContext,
        AspectParameters parameters,
        RepositoryName toolsRepository) {
      ruleContext.ruleError("Aspect error");
      return null;
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return ERROR_ASPECT_DEFINITION;
    }
  }

  public static final ErrorAspect ERROR_ASPECT = new ErrorAspect();
  private static final AspectDefinition ERROR_ASPECT_DEFINITION =
      new AspectDefinition.Builder(ERROR_ASPECT)
          .propagateToAttributes(createAttrAspects(ImmutableList.of("bar")))
          .build();

  private static AspectPropagationEdgesSupplier<String> createAttrAspects(
      ImmutableList<String> attrAspects) {
    try {
      return AspectPropagationEdgesSupplier.createForAttrAspects(
          StarlarkList.immutableCopyOf(attrAspects), /* thread= */ null);
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  private static AspectPropagationEdgesSupplier<Label> createToolchainsAspects(
      ImmutableList<String> toolchainsAspects) {
    try {
      return AspectPropagationEdgesSupplier.createForToolchainsAspects(
          StarlarkList.immutableCopyOf(toolchainsAspects),
          /* thread= */ null,
          new LabelConverter(PackageIdentifier.createInMainRepo("quux"), RepositoryMapping.EMPTY));
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * An aspect that advertises but fails to provide providers.
   */
  public static class FalseAdvertisementAspect extends NativeAspectClass
    implements ConfiguredAspectFactory {

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return FALSE_ADVERTISEMENT_DEFINITION;
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext context,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws InterruptedException, ActionConflictException {
      return new ConfiguredAspect.Builder(context).build();
    }
  }
  public static final FalseAdvertisementAspect FALSE_ADVERTISEMENT_ASPECT
      = new FalseAdvertisementAspect();
  private static final AspectDefinition FALSE_ADVERTISEMENT_DEFINITION =
      new AspectDefinition.Builder(FALSE_ADVERTISEMENT_ASPECT)
          .advertiseProvider(
              ImmutableList.of(
                  StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY)))
          .build();

  /**
   * A common base rule for mock rules in this class to reduce boilerplate.
   *
   * <p>It has a few common attributes because internal Blaze machinery assumes the presence of
   * these.
   */
  public static final MockRule BASE_RULE = () ->
      MockRule.factory(DummyRuleFactory.class).define("base");

  public static final MockRule FILE_PROVIDER_ASPECT_REQUIRING_RULE =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(FileProviderForwardingRuleFactory.class)
              .define(
                  "file_provider_aspect",
                  attr("dep", LABEL)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .aspect(FILE_PROVIDER_ASPECT));

  /**
   * A rule that defines an aspect on one of its attributes.
   */
  public static final MockRule ASPECT_REQUIRING_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "aspect",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(SIMPLE_ASPECT),
          attr("bar", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(SIMPLE_ASPECT));

  /**
   * A rule that defines different aspects on different attributes.
   */
  public static final MockRule MULTI_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(MultiAspectRuleFactory.class).define(
          "multi_aspect",
          attr("foo", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE)
              .mandatory()
              .aspect(FOO_PROVIDER_ASPECT),
          attr("bar", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE)
              .mandatory()
              .aspect(BAR_PROVIDER_ASPECT));

  private static final Function<Rule, AspectParameters> TEST_ASPECT_PARAMETERS_EXTRACTOR =
      (rule) -> {
        if (rule.isAttrDefined("baz", STRING)) {
          String value = rule.getAttr("baz").toString();
          if (!value.equals("")) {
            return new AspectParameters.Builder().addAttribute("baz", value).build();
          }
        }
        return AspectParameters.EMPTY;
      };

  /**
   * A rule that defines an {@link AspectRequiringProvider} on one of its attributes.
   */
  public static final MockRule ASPECT_REQUIRING_PROVIDER_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "aspect_requiring_provider",
          (builder, env) ->
              builder
                  .add(
                      attr("foo", LABEL_LIST)
                          .allowedFileTypes(FileTypeSet.ANY_FILE)
                          .aspect(ASPECT_REQUIRING_PROVIDER, TEST_ASPECT_PARAMETERS_EXTRACTOR))
                  .add(attr("baz", STRING)));

  /**
   * A rule that defines an {@link AspectRequiringProviderSets} on one of its attributes.
   */
  public static final MockRule ASPECT_REQUIRING_PROVIDER_SETS_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "aspect_requiring_provider_sets",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(ASPECT_REQUIRING_PROVIDER_SETS),
          attr("baz", STRING));

  /**
   * A rule that defines an {@link ExtraAttributeAspect} on one of its attributes.
   */
  public static final MockRule EXTRA_ATTRIBUTE_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "rule_with_extra_deps_aspect",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(EXTRA_ATTRIBUTE_ASPECT));

  /** A rule that defines an {@link PackageGroupAttributeAspect} on one of its attributes. */
  public static final MockRule PACKAGE_GROUP_ATTRIBUTE_ASPECT_RULE =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "rule_with_package_group_deps_aspect",
                  attr("foo", LABEL_LIST)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .aspect(PACKAGE_GROUP_ATTRIBUTE_ASPECT));

  /** A rule that defines an {@link ComputedAttributeAspect} on one of its attributes. */
  public static final MockRule COMPUTED_ATTRIBUTE_ASPECT_RULE =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "rule_with_computed_deps_aspect",
                  attr("foo", LABEL_LIST)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .aspect(COMPUTED_ATTRIBUTE_ASPECT));

  /**
   * A rule that defines an {@link ParametrizedDefinitionAspect} on one of its attributes.
   */
  public static final MockRule PARAMETERIZED_DEFINITION_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "parametrized_definition_aspect",
          (builder, env) ->
              builder
                  .add(
                      attr("foo", LABEL_LIST)
                          .allowedFileTypes(FileTypeSet.ANY_FILE)
                          .aspect(PARAMETRIZED_DEFINITION_ASPECT, TEST_ASPECT_PARAMETERS_EXTRACTOR))
                  .add(attr("baz", STRING)));


  /**
   * A rule that defines an {@link ExtraAttributeAspectRequiringProvider} on one of its attributes.
   */
  public static final MockRule EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "extra_attribute_aspect_requiring_provider",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(EXTRA_ATTRIBUTE_ASPECT_REQUIRING_PROVIDER));

  /**
   * A rule that defines an {@link AllAttributesAspect} on one of its attributes.
   */
  public static final MockRule ALL_ATTRIBUTES_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "all_attributes_aspect",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(ALL_ATTRIBUTES_ASPECT));

  /** A rule that defines an {@link AllAttributesWithToolAspect} on one of its attributes. */
  public static final MockRule ALL_ATTRIBUTES_WITH_TOOL_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "all_attributes_with_tool_aspect",
           attr("foo", LABEL_LIST)
               .allowedFileTypes(FileTypeSet.ANY_FILE)
               .aspect(ALL_ATTRIBUTES_WITH_TOOL_ASPECT));

  /**
   * A rule that defines a {@link WarningAspect} on one of its attributes.
   */
  public static final MockRule WARNING_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "warning_aspect",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(WARNING_ASPECT),
          attr("bar", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE));

  /**
   * A rule that defines an {@link ErrorAspect} on one of its attributes.
   */
  public static final MockRule ERROR_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "error_aspect",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)
              .aspect(ERROR_ASPECT),
          attr("bar", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE));

  /**
   * A simple rule that has an attribute.
   */
  public static final MockRule SIMPLE_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "simple",
          attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE),
          attr("foo1", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
          attr("txt", STRING));


  /** A rule that advertises a provider and implements it. */
  public static final MockRule HONEST_RULE =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "honest",
                  (builder, env) ->
                      builder
                          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
                          .advertiseStarlarkProvider(
                              StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER_KEY)));

  /** A rule that advertises another, different provider and implements it. */
  public static final MockRule HONEST_RULE_2 =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(DummyRuleFactory2.class)
              .define(
                  "honest2",
                  (builder, env) ->
                      builder
                          .add(attr("foo", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
                          .advertiseStarlarkProvider(
                              StarlarkProviderIdentifier.forKey(REQUIRED_PROVIDER2_KEY)));

  /** Rule with an implicit dependency. */
  public static final MockRule IMPLICIT_DEP_RULE =
      () ->
          MockRule.ancestor(BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "implicit_dep",
                  attr("$dep", LABEL).value(Label.parseCanonicalUnchecked("//extra:extra")));

  // TODO(b/65746853): provide a way to do this without passing the entire configuration
  private static final LabelListLateBoundDefault<?> PLUGINS_LABEL_LIST =
      LabelListLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class, (rule, attributes, javaConfig) -> javaConfig.getPlugins());

  public static final MockRule LATE_BOUND_DEP_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "late_bound_dep",
          attr(":plugins", LABEL_LIST).value(PLUGINS_LABEL_LIST));

  /**
   * Rule with {@link FalseAdvertisementAspect}
   */
  public static final MockRule FALSE_ADVERTISEMENT_ASPECT_RULE = () ->
      MockRule.ancestor(BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "false_advertisement_aspect",
          attr("deps", LABEL_LIST).allowedFileTypes().aspect(FALSE_ADVERTISEMENT_ASPECT));

  /** Aspect that propagates over rule outputs. */
  public static class AspectApplyingToFiles extends NativeAspectClass
      implements ConfiguredAspectFactory {

    /** Simple provider for testing */
    @Immutable
    public static final class Provider implements TransitiveInfoProvider {
      private final Label label;

      private Provider(Label label) {
        this.label = label;
      }

      public Label getLabel() {
        return label;
      }
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return AspectDefinition.builder(this).applyToFiles(true).build();
    }

    @Override
    public ConfiguredAspect create(
        Label targetLabel,
        ConfiguredTarget ct,
        RuleContext context,
        AspectParameters parameters,
        RepositoryName toolsRepository)
        throws InterruptedException, ActionConflictException {
      return ConfiguredAspect.builder(context)
          .addProvider(Provider.class, new Provider(ct.getLabel()))
          .build();
    }
  }
}
