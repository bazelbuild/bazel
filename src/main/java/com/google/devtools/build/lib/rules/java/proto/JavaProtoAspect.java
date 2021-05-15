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
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.rules.java.JavaCcInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;

/** An Aspect which JavaProtoLibrary injects to build Java SPEED protos. */
public class JavaProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  private final Label javaToolchainAttribute;

  private static LabelLateBoundDefault<?> getSpeedProtoToolchainLabel(String defaultValue) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        ProtoConfiguration.class,
        Label.parseAbsoluteUnchecked(defaultValue),
        (rule, attributes, protoConfig) -> protoConfig.protoToolchainForJava());
  }

  private final JavaSemantics javaSemantics;

  private final RpcSupport rpcSupport;
  private final String defaultSpeedProtoToolchainLabel;

  protected JavaProtoAspect(
      JavaSemantics javaSemantics,
      RpcSupport rpcSupport,
      String defaultSpeedProtoToolchainLabel,
      RuleDefinitionEnvironment env) {
    this.javaSemantics = Preconditions.checkNotNull(javaSemantics);
    this.rpcSupport = Preconditions.checkNotNull(rpcSupport);
    this.defaultSpeedProtoToolchainLabel =
        Preconditions.checkNotNull(defaultSpeedProtoToolchainLabel);
    this.javaToolchainAttribute = JavaSemantics.javaToolchainAttribute(env);
  }

  protected ConfiguredAspect createWithProtocOpts(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters parameters,
      String toolsRepository,
      Iterable<String> additionalProtocOpts)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder aspect = new ConfiguredAspect.Builder(ruleContext);

    if (!rpcSupport.checkAttributes(ruleContext, parameters)) {
      return aspect.build();
    }

    ProtoInfo protoInfo = ctadBase.getConfiguredTarget().get(ProtoInfo.PROVIDER);

    JavaProtoAspectCommon aspectCommon =
        JavaProtoAspectCommon.getSpeedInstance(ruleContext, javaSemantics, rpcSupport);
    Impl impl = new Impl(ruleContext, protoInfo, aspectCommon, rpcSupport, additionalProtocOpts);
    impl.addProviders(aspect);
    return aspect.build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters parameters,
      String toolsRepository)
      throws InterruptedException, ActionConflictException {
    return createWithProtocOpts(
        ctadBase, ruleContext, parameters, toolsRepository, ImmutableList.of());
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(this)
            .propagateAlongAttribute("deps")
            .propagateAlongAttribute("exports")
            .requiresConfigurationFragments(
                JavaConfiguration.class, ProtoConfiguration.class, PlatformConfiguration.class)
            .requireStarlarkProviders(ProtoInfo.PROVIDER.id())
            .advertiseProvider(JavaProtoLibraryAspectProvider.class)
            .advertiseProvider(
                ImmutableList.of(StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey())))
            .add(
                attr(JavaProtoAspectCommon.SPEED_PROTO_TOOLCHAIN_ATTR, LABEL)
                    // TODO(carmi): reinstate mandatoryBuiltinProviders(ProtoLangToolchainProvider)
                    // once it's in a Bazel release.
                    .legacyAllowAnyFileType()
                    .value(getSpeedProtoToolchainLabel(defaultSpeedProtoToolchainLabel)))
            .add(
                attr(JavaRuleClasses.JAVA_TOOLCHAIN_ATTRIBUTE_NAME, LABEL)
                    .useOutputLicenses()
                    .value(javaToolchainAttribute)
                    .mandatoryProviders(ToolchainInfo.PROVIDER.id()));

    rpcSupport.mutateAspectDefinition(result, aspectParameters);

    return result.build();
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

    private final ImmutableList<String> additionalProtocOpts;

    Impl(
        RuleContext ruleContext,
        ProtoInfo protoInfo,
        JavaProtoAspectCommon aspectCommon,
        RpcSupport rpcSupport,
        Iterable<String> additionalProtocOpts) {
      this.ruleContext = ruleContext;
      this.protoInfo = protoInfo;
      this.rpcSupport = rpcSupport;
      this.aspectCommon = aspectCommon;
      this.additionalProtocOpts = ImmutableList.copyOf(additionalProtocOpts);
      this.javaProtoLibraryAspectProviders =
          ruleContext.getPrerequisites("deps", JavaProtoLibraryAspectProvider.class);

      this.dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              ruleContext.getPrerequisites("deps", JavaCompilationArgsProvider.class));

      this.exportsCompilationArgs =
          JavaCompilationArgsProvider.merge(
              ruleContext.getPrerequisites("exports", JavaCompilationArgsProvider.class));
    }

    void addProviders(ConfiguredAspect.Builder aspect) throws InterruptedException {
      // Represents the result of compiling the code generated for this proto, including all of its
      // dependencies.
      JavaInfo.Builder javaInfo = JavaInfo.Builder.create();
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

        transitiveOutputJars.add(outputJar);

        Artifact compileTimeJar =
            generatedCompilationArgsProvider.getDirectCompileTimeJars().getSingleton();
        // TODO(carmi): Expose to native rules
        JavaRuleOutputJarsProvider ruleOutputJarsProvider =
            JavaRuleOutputJarsProvider.builder()
                .addJavaOutput(
                    JavaOutput.builder()
                        .setClassJar(outputJar)
                        .setCompileJar(compileTimeJar)
                        .addSourceJar(sourceJar)
                        .build())
                .build();
        JavaSourceJarsProvider sourceJarsProvider =
            JavaSourceJarsProvider.create(
                NestedSetBuilder.create(Order.STABLE_ORDER, sourceJar),
                ImmutableList.of(sourceJar));

        aspect.addProvider(ruleOutputJarsProvider).addProvider(sourceJarsProvider);
        javaInfo.addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider);
        javaInfo.addProvider(JavaSourceJarsProvider.class, sourceJarsProvider);
      } else {
        // No sources - this proto_library is an alias library, which exports its dependencies.
        // Simply propagate the compilation-args from its dependencies.
        generatedCompilationArgsProvider = dependencyCompilationArgs;
        aspect.addProvider(JavaRuleOutputJarsProvider.EMPTY);
        javaInfo.addProvider(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.EMPTY);
      }

      generatedCompilationArgsProvider =
          JavaCompilationArgsProvider.merge(
              ImmutableList.of(generatedCompilationArgsProvider, exportsCompilationArgs));

      aspect.addProvider(generatedCompilationArgsProvider);
      javaInfo.addProvider(JavaCompilationArgsProvider.class, generatedCompilationArgsProvider);

      javaInfo.addProvider(
          JavaCcInfoProvider.class,
          createCcLinkingInfo(ruleContext, aspectCommon.getProtoRuntimeDeps()));

      aspect
          .addNativeDeclaredProvider(javaInfo.build())
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
      if (protoInfo.getOriginalDirectProtoSources().isEmpty()) {
        return false;
      }

      final ProtoSourceFileBlacklist protoBlackList;
      NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
      blacklistedProtos.addTransitive(aspectCommon.getProtoToolchainProvider().blacklistedProtos());
      blacklistedProtos.addTransitive(rpcSupport.getBlacklist(ruleContext));

      protoBlackList = new ProtoSourceFileBlacklist(ruleContext, blacklistedProtos.build());

      return protoBlackList.checkSrcs(
          protoInfo.getOriginalDirectProtoSources(), "java_proto_library");
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ImmutableList.Builder<ToolchainInvocation> invocations = ImmutableList.builder();
      invocations.add(
          new ToolchainInvocation(
              "java",
              aspectCommon.getProtoToolchainProvider(),
              sourceJar.getExecPathString(),
              additionalProtocOpts));
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
