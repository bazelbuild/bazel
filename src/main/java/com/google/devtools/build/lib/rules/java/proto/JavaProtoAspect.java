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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.devtools.build.lib.cmdline.Label.parseAbsoluteUnchecked;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.java.proto.JplCcLinkParams.createCcLinkingInfo;
import static com.google.devtools.build.lib.rules.java.proto.StrictDepsUtils.createNonStrictCompilationArgsProvider;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import javax.annotation.Nullable;

/** An Aspect which JavaProtoLibrary injects to build Java SPEED protos. */
public class JavaProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  private final LabelLateBoundDefault<JavaConfiguration> hostJdkAttribute;
  private final LabelLateBoundDefault<JavaConfiguration> javaToolchainAttribute;

  private static LabelLateBoundDefault<?> getSpeedProtoToolchainLabel(String defaultValue) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        ProtoConfiguration.class,
        Label.parseAbsoluteUnchecked(defaultValue),
        (rule, attributes, protoConfig) -> protoConfig.protoToolchainForJava());
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
      LabelLateBoundDefault<JavaConfiguration> hostJdkAttribute,
      LabelLateBoundDefault<JavaConfiguration> javaToolchainAttribute) {
    this.javaSemantics = Preconditions.checkNotNull(javaSemantics);
    this.jacocoLabel = jacocoLabel;
    this.rpcSupport = Preconditions.checkNotNull(rpcSupport);
    this.defaultSpeedProtoToolchainLabel =
        Preconditions.checkNotNull(defaultSpeedProtoToolchainLabel);
    this.hostJdkAttribute = Preconditions.checkNotNull(hostJdkAttribute);
    this.javaToolchainAttribute = Preconditions.checkNotNull(javaToolchainAttribute);
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder aspect = new ConfiguredAspect.Builder(this, parameters, ruleContext);

    if (!rpcSupport.checkAttributes(ruleContext, parameters)) {
      return aspect.build();
    }

    ProtoInfo protoInfo = ctadBase.getConfiguredTarget().get(ProtoInfo.PROVIDER);

    JavaProtoAspectCommon aspectCommon =
        JavaProtoAspectCommon.getSpeedInstance(ruleContext, javaSemantics, rpcSupport);
    Impl impl = new Impl(ruleContext, protoInfo, aspectCommon, rpcSupport);
    impl.addProviders(aspect);
    return aspect.build();
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(this)
            .propagateAlongAttribute("deps")
            .propagateAlongAttribute("exports")
            .requiresConfigurationFragments(JavaConfiguration.class, ProtoConfiguration.class)
            .requireSkylarkProviders(ProtoInfo.PROVIDER.id())
            .advertiseProvider(JavaProtoLibraryAspectProvider.class)
            .advertiseProvider(ImmutableList.of(JavaSkylarkApiProvider.SKYLARK_NAME))
            .add(
                attr(JavaProtoAspectCommon.SPEED_PROTO_TOOLCHAIN_ATTR, LABEL)
                    // TODO(carmi): reinstate mandatoryNativeProviders(ProtoLangToolchainProvider)
                    // once it's in a Bazel release.
                    .legacyAllowAnyFileType()
                    .value(getSpeedProtoToolchainLabel(defaultSpeedProtoToolchainLabel)))
            .add(attr(":host_jdk", LABEL).cfg(HostTransition.INSTANCE).value(hostJdkAttribute))
            .add(
                attr(":java_toolchain", LABEL)
                    .useOutputLicenses()
                    .allowedRuleClasses("java_toolchain")
                    .value(javaToolchainAttribute));

    rpcSupport.mutateAspectDefinition(result, aspectParameters);

    Attribute.Builder<Label> jacocoAttr =
        attr("$jacoco_instrumentation", LABEL).cfg(HostTransition.INSTANCE);

    if (jacocoLabel != null) {
      jacocoAttr.value(parseAbsoluteUnchecked(jacocoLabel));
    }
    return result.add(jacocoAttr).build();
  }

  private static class Impl {

    private final RuleContext ruleContext;
    private final ProtoInfo protoInfo;

    private final RpcSupport rpcSupport;
    private final JavaProtoAspectCommon aspectCommon;

    /**
     * Compilation-args from all dependencies, merged together. This is typically the input to a
     * Java compilation action.
     */
    private final JavaCompilationArgsProvider dependencyCompilationArgs;

    // Compilation-args from all exports, merged together.
    private final JavaCompilationArgsProvider exportsCompilationArgs;

    private final Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders;

    Impl(
        RuleContext ruleContext,
        ProtoInfo protoInfo,
        JavaProtoAspectCommon aspectCommon,
        RpcSupport rpcSupport) {
      this.ruleContext = ruleContext;
      this.protoInfo = protoInfo;
      this.rpcSupport = rpcSupport;
      this.aspectCommon = aspectCommon;
      this.javaProtoLibraryAspectProviders =
          ruleContext.getPrerequisites(
              "deps", RuleConfiguredTarget.Mode.TARGET, JavaProtoLibraryAspectProvider.class);

      this.dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              ruleContext.getPrerequisites(
                  "deps", RuleConfiguredTarget.Mode.TARGET, JavaCompilationArgsProvider.class));

        this.exportsCompilationArgs =
            JavaCompilationArgsProvider.merge(
                ruleContext.getPrerequisites(
                    "exports",
                    RuleConfiguredTarget.Mode.TARGET,
                    JavaCompilationArgsProvider.class));
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

      if (shouldGenerateCode()) {
        Artifact sourceJar = aspectCommon.getSourceJarArtifact();
        createProtoCompileAction(sourceJar);
        Artifact outputJar = aspectCommon.getOutputJarArtifact();

        generatedCompilationArgsProvider =
            aspectCommon.createJavaCompileAction(
                "java_proto_library", sourceJar, outputJar, dependencyCompilationArgs);

        NestedSet<Artifact> javaSourceJars =
            NestedSetBuilder.<Artifact>stableOrder().add(sourceJar).build();
        transitiveOutputJars.add(outputJar);

        Artifact compileTimeJar =
            getOnlyElement(generatedCompilationArgsProvider.getDirectCompileTimeJars());
        // TODO(carmi): Expose to native rules
        JavaRuleOutputJarsProvider ruleOutputJarsProvider =
            JavaRuleOutputJarsProvider.builder()
                .addOutputJar(
                    outputJar,
                    compileTimeJar,
                    null /* manifestProto */,
                    ImmutableList.of(sourceJar))
                .build();
        JavaSourceJarsProvider sourceJarsProvider =
            JavaSourceJarsProvider.create(
                NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER), javaSourceJars);

        aspect.addProvider(ruleOutputJarsProvider).addProvider(sourceJarsProvider);
      } else {
        // No sources - this proto_library is an alias library, which exports its dependencies.
        // Simply propagate the compilation-args from its dependencies.
        generatedCompilationArgsProvider = dependencyCompilationArgs;
        aspect.addProvider(JavaRuleOutputJarsProvider.EMPTY);
      }

      generatedCompilationArgsProvider =
          JavaCompilationArgsProvider.merge(
              ImmutableList.of(generatedCompilationArgsProvider, exportsCompilationArgs));

      aspect.addProvider(generatedCompilationArgsProvider);
      aspect.addNativeDeclaredProvider(
          createCcLinkingInfo(ruleContext, aspectCommon.getProtoRuntimeDeps()));
      JavaSkylarkApiProvider javaSkylarkApiProvider = JavaSkylarkApiProvider.fromRuleContext();
      aspect
          .addSkylarkTransitiveInfo(JavaSkylarkApiProvider.NAME, javaSkylarkApiProvider)
          .addProvider(
              new JavaProtoLibraryAspectProvider(
                  transitiveOutputJars.build(),
                  createNonStrictCompilationArgsProvider(
                      javaProtoLibraryAspectProviders,
                      generatedCompilationArgsProvider,
                      aspectCommon.getProtoRuntimeDeps())));
    }

    /**
     * Decides whether code should be generated for the .proto files in the currently-processed
     * proto_library.
     */
    private boolean shouldGenerateCode() {
      if (protoInfo.getDirectProtoSources().isEmpty()) {
        return false;
      }

      final ProtoSourceFileBlacklist protoBlackList;
      NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
      blacklistedProtos.addTransitive(aspectCommon.getProtoToolchainProvider().blacklistedProtos());
      blacklistedProtos.addTransitive(rpcSupport.getBlacklist(ruleContext));

      protoBlackList = new ProtoSourceFileBlacklist(ruleContext, blacklistedProtos.build());

      return protoBlackList.checkSrcs(protoInfo.getDirectProtoSources(), "java_proto_library");
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ImmutableList.Builder<ToolchainInvocation> invocations = ImmutableList.builder();
      invocations.add(
          new ToolchainInvocation(
              "java", aspectCommon.getProtoToolchainProvider(), sourceJar.getExecPathString()));
      invocations.addAll(rpcSupport.getToolchainInvocation(ruleContext, sourceJar));
      ProtoCompileActionBuilder.registerActions(
          ruleContext,
          invocations.build(),
          protoInfo,
          ruleContext.getLabel(),
          ImmutableList.of(sourceJar),
          "Java (Immutable)",
          Exports.USE,
          rpcSupport.allowServices(ruleContext) ? Services.ALLOW : Services.DISALLOW);
    }
  }
}
