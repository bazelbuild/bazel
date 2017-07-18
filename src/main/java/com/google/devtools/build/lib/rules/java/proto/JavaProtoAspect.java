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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
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
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
import javax.annotation.Nullable;

/** An Aspect which JavaProtoLibrary injects to build Java SPEED protos. */
public class JavaProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  private static final String SPEED_PROTO_TOOLCHAIN_ATTR = ":aspect_java_proto_toolchain";
  private final LateBoundLabel<BuildConfiguration> hostJdkAttribute;

  private static Attribute.LateBoundLabel<BuildConfiguration> getSpeedProtoToolchainLabel(
      String defaultValue) {
    return new Attribute.LateBoundLabel<BuildConfiguration>(
        defaultValue, ProtoConfiguration.class) {
      @Override
      public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
        return configuration.getFragment(ProtoConfiguration.class).protoToolchainForJava();
      }
    };
  }

  private final JavaSemantics javaSemantics;

  @Nullable private final String jacocoLabel;
  private final RpcSupport rpcSupport;
  private final String defaultSpeedProtoToolchainLabel;

  protected JavaProtoAspect(
      JavaSemantics javaSemantics,
      @Nullable String jacocoLabel,
      RpcSupport rpcSupport,
      String defaultSpeedProtoToolchainLabel,
      LateBoundLabel<BuildConfiguration> hostJdkAttribute) {
    this.javaSemantics = javaSemantics;
    this.jacocoLabel = jacocoLabel;
    this.rpcSupport = rpcSupport;
    this.defaultSpeedProtoToolchainLabel = defaultSpeedProtoToolchainLabel;
    this.hostJdkAttribute = hostJdkAttribute;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder aspect = new ConfiguredAspect.Builder(this, parameters, ruleContext);

    if (!rpcSupport.checkAttributes(ruleContext, parameters)) {
      return aspect.build();
    }

    // Get SupportData, which is provided by the proto_library rule we attach to.
    SupportData supportData =
        checkNotNull(base.getProvider(ProtoSupportDataProvider.class)).getSupportData();

    Impl impl = new Impl(ruleContext, supportData, javaSemantics, rpcSupport);
    if (impl.shouldGenerateCode() && ActionReuser.reuseExistingActions(base, ruleContext, aspect)) {
      return aspect.build();
    }
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
                attr(SPEED_PROTO_TOOLCHAIN_ATTR, LABEL)
                    // TODO(carmi): reinstate mandatoryNativeProviders(ProtoLangToolchainProvider)
                    // once it's in a Bazel release.
                    .legacyAllowAnyFileType()
                    .value(getSpeedProtoToolchainLabel(defaultSpeedProtoToolchainLabel)))
            .add(attr(":host_jdk", LABEL).cfg(HOST).value(hostJdkAttribute))
            .add(
                attr(":java_toolchain", LABEL)
                    .useOutputLicenses()
                    .allowedRuleClasses("java_toolchain")
                    .value(JavaSemantics.JAVA_TOOLCHAIN));

    rpcSupport.mutateAspectDefinition(result, aspectParameters);

    Attribute.Builder<Label> jacocoAttr = attr("$jacoco_instrumentation", LABEL).cfg(HOST);

    if (jacocoLabel != null) {
      jacocoAttr.value(parseAbsoluteUnchecked(jacocoLabel));
    }
    return result.add(jacocoAttr).build();
  }

  private static class Impl {

    private final RuleContext ruleContext;
    private final SupportData supportData;

    private final RpcSupport rpcSupport;
    private final JavaSemantics javaSemantics;

    /**
     * Compilation-args from all dependencies, merged together. This is typically the input to a
     * Java compilation action.
     */
    private final JavaCompilationArgsProvider dependencyCompilationArgs;

    private final Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders;

    Impl(
        final RuleContext ruleContext,
        final SupportData supportData,
        JavaSemantics javaSemantics,
        RpcSupport rpcSupport) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.javaSemantics = javaSemantics;
      this.rpcSupport = rpcSupport;
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

      if (shouldGenerateCode()) {
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
      TransitiveInfoProviderMap javaProviders = javaProvidersBuilder.build();
      aspect
          .addSkylarkTransitiveInfo(
              JavaSkylarkApiProvider.PROTO_NAME.getLegacyId(),
              JavaSkylarkApiProvider.fromProviderMap(javaProviders))
          .addProvider(
              new JavaProtoLibraryAspectProvider(javaProviders, transitiveOutputJars.build()));
    }

    /**
     * Decides whether code should be generated for the .proto files in the currently-processed
     * proto_library.
     */
    private boolean shouldGenerateCode() {
      if (!supportData.hasProtoSources()) {
        return false;
      }

      final ProtoSourceFileBlacklist protoBlackList;
      NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
      blacklistedProtos.addTransitive(getProtoToolchainProvider().blacklistedProtos());
      blacklistedProtos.addTransitive(rpcSupport.getBlacklist(ruleContext));

      protoBlackList = new ProtoSourceFileBlacklist(ruleContext, blacklistedProtos.build());

      return protoBlackList.checkSrcs(supportData.getDirectProtoSources(), "java_proto_library");
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ImmutableList.Builder<ToolchainInvocation> invocations = ImmutableList.builder();
      invocations.add(
          new ToolchainInvocation(
              "java", checkNotNull(getProtoToolchainProvider()), sourceJar.getExecPathString()));
      invocations.addAll(rpcSupport.getToolchainInvocation(ruleContext, sourceJar));
      ProtoCompileActionBuilder.registerActions(
          ruleContext,
          invocations.build(),
          supportData.getDirectProtoSources(),
          supportData.getTransitiveImports(),
          supportData.getProtosInDirectDeps(),
          ruleContext.getLabel().getCanonicalForm(),
          ImmutableList.of(sourceJar),
          "Java (Immutable)",
          rpcSupport.allowServices(ruleContext));
    }

    private JavaCompilationArgsProvider createJavaCompileAction(
        Artifact sourceJar, Artifact outputJar) {
      JavaLibraryHelper helper =
          new JavaLibraryHelper(ruleContext)
              .setOutput(outputJar)
              .addSourceJars(sourceJar)
              .setJavacOpts(ProtoJavacOpts.constructJavacOpts(ruleContext));
      helper.addDep(dependencyCompilationArgs).setCompilationStrictDepsMode(StrictDepsMode.OFF);
      TransitiveInfoCollection runtime = getProtoToolchainProvider().runtime();
      if (runtime != null) {
        helper.addDep(runtime.getProvider(JavaCompilationArgsProvider.class));
      }

      rpcSupport.mutateJavaCompileAction(ruleContext, helper);
      return helper.buildCompilationArgsProvider(
          helper.build(
              javaSemantics,
              JavaCompilationHelper.getJavaToolchainProvider(ruleContext),
              JavaHelper.getHostJavabaseInputs(ruleContext),
              JavaCompilationHelper.getInstrumentationJars(ruleContext)),
          true /* isReportedAsStrict */);
    }

    private ProtoLangToolchainProvider getProtoToolchainProvider() {
      return ruleContext.getPrerequisite(
          SPEED_PROTO_TOOLCHAIN_ATTR, TARGET, ProtoLangToolchainProvider.class);
    }

    private Artifact getSourceJarArtifact() {
      return ruleContext.getGenfilesArtifact(ruleContext.getLabel().getName() + "-speed-src.jar");
    }

    private Artifact getOutputJarArtifact() {
      return ruleContext.getBinArtifact("lib" + ruleContext.getLabel().getName() + "-speed.jar");
    }
  }
}
