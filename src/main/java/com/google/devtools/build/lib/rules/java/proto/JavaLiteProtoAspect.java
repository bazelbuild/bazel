// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.cmdline.Label.parseAbsoluteUnchecked;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.java.proto.JplCcLinkParams.createCcLinkParamsStore;
import static com.google.devtools.build.lib.rules.java.proto.StrictDepsUtils.createNonStrictCompilationArgsProvider;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.analysis.WrappingProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
import javax.annotation.Nullable;

/** An Aspect which JavaLiteProtoLibrary injects to build Java Lite protos. */
public class JavaLiteProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  public static final String PROTO_TOOLCHAIN_ATTR = ":aspect_proto_toolchain_for_javalite";

  public static Attribute.LateBoundLabel<BuildConfiguration> getProtoToolchainLabel(
      String defaultValue) {
    return new Attribute.LateBoundLabel<BuildConfiguration>(
        defaultValue, ProtoConfiguration.class) {
      @Override
      public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
        return configuration.getFragment(ProtoConfiguration.class).protoToolchainForJavaLite();
      }
    };
  }

  private final JavaSemantics javaSemantics;

  @Nullable private final String jacocoLabel;
  private final String defaultProtoToolchainLabel;
  private final LateBoundLabel<BuildConfiguration> hostJdkAttribute;

  public JavaLiteProtoAspect(
      JavaSemantics javaSemantics,
      @Nullable String jacocoLabel,
      String defaultProtoToolchainLabel,
      LateBoundLabel<BuildConfiguration> hostJdkAttribute) {
    this.javaSemantics = javaSemantics;
    this.jacocoLabel = jacocoLabel;
    this.defaultProtoToolchainLabel = defaultProtoToolchainLabel;
    this.hostJdkAttribute = hostJdkAttribute;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder aspect = new ConfiguredAspect.Builder(this, parameters, ruleContext);

    // Get SupportData, which is provided by the proto_library rule we attach to.
    SupportData supportData =
        checkNotNull(base.getProvider(ProtoSupportDataProvider.class)).getSupportData();

    Impl impl = new Impl(ruleContext, supportData, javaSemantics);
    impl.addProviders(aspect);

    return aspect.build();
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(this)
            .propagateAlongAttribute("deps")
            .requiresConfigurationFragments(JavaConfiguration.class, ProtoConfiguration.class)
            .requireProviders(ProtoSourcesProvider.class)
            .advertiseProvider(JavaProtoLibraryAspectProvider.class)
            .advertiseProvider(ImmutableList.of(JavaSkylarkApiProvider.PROTO_NAME))
            .add(
                attr(PROTO_TOOLCHAIN_ATTR, LABEL)
                    .mandatoryNativeProviders(
                        ImmutableList.<Class<? extends TransitiveInfoProvider>>of(
                            ProtoLangToolchainProvider.class))
                    .value(getProtoToolchainLabel(defaultProtoToolchainLabel)))
            .add(attr(":host_jdk", LABEL).cfg(HOST).value(hostJdkAttribute))
            .add(
                attr(":java_toolchain", LABEL)
                    .useOutputLicenses()
                    .allowedRuleClasses("java_toolchain")
                    .value(JavaSemantics.JAVA_TOOLCHAIN));

    Attribute.Builder<Label> jacocoAttr = attr("$jacoco_instrumentation", LABEL).cfg(HOST);

    if (jacocoLabel != null) {
      jacocoAttr.value(parseAbsoluteUnchecked(jacocoLabel));
    }
    return result.add(jacocoAttr).build();
  }

  private static class Impl {

    private final RuleContext ruleContext;
    private final SupportData supportData;

    /**
     * Compilation-args from all dependencies, merged together. This is typically the input to a
     * Java compilation action.
     */
    private final JavaCompilationArgsProvider dependencyCompilationArgs;

    private final JavaSemantics javaSemantics;
    private final Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders;

    Impl(
        final RuleContext ruleContext, final SupportData supportData, JavaSemantics javaSemantics) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.javaSemantics = javaSemantics;
      this.javaProtoLibraryAspectProviders =
          ruleContext.getPrerequisites(
              "deps", RuleConfiguredTarget.Mode.TARGET, JavaProtoLibraryAspectProvider.class);

      dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              WrappingProvider.Helper.unwrapProviders(
                  javaProtoLibraryAspectProviders, JavaCompilationArgsProvider.class));
    }

    void addProviders(ConfiguredAspect.Builder aspect) {
      // Represents the result of compiling the code generated for this proto, including all of its
      // dependencies.
      JavaCompilationArgsProvider generatedCompilationArgsProvider;

      // The jars that this proto and its dependencies produce. Used to roll-up jars up to the
      // java_proto_library, to be put into filesToBuild.
      NestedSetBuilder<Artifact> transitiveOutputJars = NestedSetBuilder.stableOrder();
      for (JavaProtoLibraryAspectProvider provider : javaProtoLibraryAspectProviders) {
        transitiveOutputJars.addTransitive(provider.getJars());
      }

      TransitiveInfoProviderMapBuilder javaProvidersBuilder =
          new TransitiveInfoProviderMapBuilder();

      if (supportData.hasProtoSources()) {
        Artifact sourceJar = getSourceJarArtifact();
        createProtoCompileAction(sourceJar);
        Artifact outputJar = getOutputJarArtifact();

        generatedCompilationArgsProvider = createJavaCompileAction(sourceJar, outputJar);

        NestedSet<Artifact> javaSourceJars =
            NestedSetBuilder.<Artifact>stableOrder().add(sourceJar).build();
        transitiveOutputJars.add(outputJar);

        Artifact compileTimeJar =
            getOnlyElement(
                generatedCompilationArgsProvider.getJavaCompilationArgs().getCompileTimeJars());
        // TODO(carmi): Expose to native rules
        JavaRuleOutputJarsProvider ruleOutputJarsProvider =
            JavaRuleOutputJarsProvider.builder()
                .addOutputJar(outputJar, compileTimeJar, ImmutableList.of(sourceJar))
                .build();
        JavaSourceJarsProvider sourceJarsProvider =
            JavaSourceJarsProvider.create(
                NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER), javaSourceJars);

        javaProvidersBuilder.add(ruleOutputJarsProvider).add(sourceJarsProvider);
      } else {
        // No sources - this proto_library is an alias library, which exports its dependencies.
        // Simply propagate the compilation-args from its dependencies.
        generatedCompilationArgsProvider = dependencyCompilationArgs;
        javaProvidersBuilder.add(JavaRuleOutputJarsProvider.EMPTY);
      }

      javaProvidersBuilder.add(generatedCompilationArgsProvider);
      javaProvidersBuilder.add(createCcLinkParamsStore(ruleContext, getProtoRuntimeDeps()));
      TransitiveInfoProviderMap javaProviders = javaProvidersBuilder.build();

      aspect
          .addSkylarkTransitiveInfo(
              JavaSkylarkApiProvider.PROTO_NAME.getLegacyId(),
              JavaSkylarkApiProvider.fromProviderMap(javaProviders))
          .addProvider(
              new JavaProtoLibraryAspectProvider(
                  javaProviders,
                  transitiveOutputJars.build(),
                  createNonStrictCompilationArgsProvider(
                      javaProtoLibraryAspectProviders,
                      generatedCompilationArgsProvider.getJavaCompilationArgs(),
                      getProtoRuntimeDeps())));
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ProtoCompileActionBuilder.registerActions(
          ruleContext,
          ImmutableList.of(
              new ProtoCompileActionBuilder.ToolchainInvocation(
                  "javalite", getProtoToolchainProvider(), sourceJar.getExecPathString())),
          supportData.getDirectProtoSources(),
          supportData.getTransitiveImports(),
          supportData.getProtosInDirectDeps(),
          ruleContext.getLabel(),
          ImmutableList.of(sourceJar),
          "JavaLite",
          true /* allowServices */);
    }

    private JavaCompilationArgsProvider createJavaCompileAction(
        Artifact sourceJar, Artifact outputJar) {
      JavaLibraryHelper helper =
          new JavaLibraryHelper(ruleContext)
              .setOutput(outputJar)
              .addSourceJars(sourceJar)
              .setJavacOpts(ProtoJavacOpts.constructJavacOpts(ruleContext));
      helper.addDep(dependencyCompilationArgs).setCompilationStrictDepsMode(StrictDepsMode.OFF);
      for (TransitiveInfoCollection t : getProtoRuntimeDeps()) {
        JavaCompilationArgsProvider provider = t.getProvider(JavaCompilationArgsProvider.class);
        if (provider != null) {
          helper.addDep(provider);
        }
      }

      JavaCompilationArtifacts artifacts =
          helper.build(
              javaSemantics,
              JavaCompilationHelper.getJavaToolchainProvider(ruleContext),
              JavaHelper.getHostJavabaseInputs(ruleContext),
              JavaCompilationHelper.getInstrumentationJars(ruleContext));
      return helper.buildCompilationArgsProvider(artifacts, true /* isReportedAsStrict */);
    }

    private ImmutableList<TransitiveInfoCollection> getProtoRuntimeDeps() {
      TransitiveInfoCollection runtime = getProtoToolchainProvider().runtime();
      return runtime != null ? ImmutableList.of(runtime) : ImmutableList.of();
    }

    private ProtoLangToolchainProvider getProtoToolchainProvider() {
      return checkNotNull(
          ruleContext.getPrerequisite(
              PROTO_TOOLCHAIN_ATTR, TARGET, ProtoLangToolchainProvider.class));
    }

    private Artifact getSourceJarArtifact() {
      return ruleContext.getGenfilesArtifact(ruleContext.getLabel().getName() + "-lite-src.jar");
    }

    private Artifact getOutputJarArtifact() {
      return ruleContext.getBinArtifact("lib" + ruleContext.getLabel().getName() + "-lite.jar");
    }
  }
}
