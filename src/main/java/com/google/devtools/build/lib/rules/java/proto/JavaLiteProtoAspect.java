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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
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
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;

/** An Aspect which JavaLiteProtoLibrary injects to build Java Lite protos. */
public class JavaLiteProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  public static LabelLateBoundDefault<?> getProtoToolchainLabel(String defaultValue) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        ProtoConfiguration.class,
        Label.parseAbsoluteUnchecked(defaultValue),
        (rule, attributes, protoConfig) -> protoConfig.protoToolchainForJavaLite());
  }

  private final JavaSemantics javaSemantics;

  private final String defaultProtoToolchainLabel;
  private final Label javaToolchainAttribute;

  public JavaLiteProtoAspect(
      JavaSemantics javaSemantics,
      String defaultProtoToolchainLabel,
      RuleDefinitionEnvironment env) {
    this.javaSemantics = javaSemantics;
    this.defaultProtoToolchainLabel = defaultProtoToolchainLabel;
    this.javaToolchainAttribute = JavaSemantics.javaToolchainAttribute(env);
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTargetAndData ctadBase,
      RuleContext ruleContext,
      AspectParameters parameters,
      String toolsRepository)
      throws InterruptedException, ActionConflictException {
    ConfiguredAspect.Builder aspect = new ConfiguredAspect.Builder(ruleContext);

    ProtoInfo protoInfo = ctadBase.getConfiguredTarget().get(ProtoInfo.PROVIDER);

    JavaProtoAspectCommon aspectCommon =
        JavaProtoAspectCommon.getLiteInstance(ruleContext, javaSemantics);
    Impl impl = new Impl(ruleContext, protoInfo, aspectCommon);
    impl.addProviders(aspect);

    return aspect.build();
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
                attr(JavaProtoAspectCommon.LITE_PROTO_TOOLCHAIN_ATTR, LABEL)
                    .mandatoryBuiltinProviders(
                        ImmutableList.<Class<? extends TransitiveInfoProvider>>of(
                            ProtoLangToolchainProvider.class))
                    .value(getProtoToolchainLabel(defaultProtoToolchainLabel)))
            .add(
                attr(JavaRuleClasses.JAVA_TOOLCHAIN_ATTRIBUTE_NAME, LABEL)
                    .useOutputLicenses()
                    .value(javaToolchainAttribute)
                    .mandatoryProviders(ToolchainInfo.PROVIDER.id()));

    return result.build();
  }

  private static class Impl {

    private final RuleContext ruleContext;
    private final ProtoInfo protoInfo;

    /**
     * Compilation-args from all dependencies, merged together. This is typically the input to a
     * Java compilation action.
     */
    private final JavaCompilationArgsProvider dependencyCompilationArgs;

    // Compilation-args from all exports, merged together.
    private final JavaCompilationArgsProvider exportsCompilationArgs;

    private final JavaProtoAspectCommon aspectCommon;
    private final Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders;

    Impl(RuleContext ruleContext, ProtoInfo protoInfo, JavaProtoAspectCommon aspectCommon) {
      this.ruleContext = ruleContext;
      this.protoInfo = protoInfo;
      this.aspectCommon = aspectCommon;
      this.javaProtoLibraryAspectProviders =
          ruleContext.getPrerequisites("deps", JavaProtoLibraryAspectProvider.class);

      dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              ruleContext.getPrerequisites("deps", JavaCompilationArgsProvider.class));

      this.exportsCompilationArgs =
          JavaCompilationArgsProvider.merge(
              ruleContext.getPrerequisites("exports", JavaCompilationArgsProvider.class));
    }

    void addProviders(ConfiguredAspect.Builder aspect) throws InterruptedException {
      JavaInfo.Builder javaInfo = JavaInfo.Builder.create();
      // Represents the result of compiling the code generated for this proto, including all of its
      // dependencies.
      JavaCompilationArgsProvider generatedCompilationArgsProvider;

      // The jars that this proto and its dependencies produce. Used to roll-up jars up to the
      // java_proto_library, to be put into filesToBuild.
      NestedSetBuilder<Artifact> transitiveOutputJars = NestedSetBuilder.stableOrder();
      for (JavaProtoLibraryAspectProvider provider : javaProtoLibraryAspectProviders) {
        transitiveOutputJars.addTransitive(provider.getJars());
      }

      if (!protoInfo.getDirectProtoSources().isEmpty()) {
        Artifact sourceJar = aspectCommon.getSourceJarArtifact();
        createProtoCompileAction(sourceJar);
        Artifact outputJar = aspectCommon.getOutputJarArtifact();

        generatedCompilationArgsProvider =
            aspectCommon.createJavaCompileAction(
                "java_lite_proto_library", sourceJar, outputJar, dependencyCompilationArgs);

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
                        .setCompileJdeps(
                            generatedCompilationArgsProvider
                                .getCompileTimeJavaDependencyArtifacts()
                                .getSingleton())
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

    private void createProtoCompileAction(Artifact sourceJar) {
      ProtoCompileActionBuilder.registerActions(
          ruleContext,
          ImmutableList.of(
              new ProtoCompileActionBuilder.ToolchainInvocation(
                  "javalite",
                  aspectCommon.getProtoToolchainProvider(),
                  sourceJar.getExecPathString())),
          protoInfo,
          ruleContext.getLabel(),
          ImmutableList.of(sourceJar),
          "JavaLite",
          Exports.USE,
          Services.ALLOW);
    }
  }
}
