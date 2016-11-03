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
import static com.google.common.collect.Iterables.transform;
import static com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.cmdline.Label.parseAbsoluteUnchecked;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.java.proto.JavaCompilationArgsAspectProvider.GET_PROVIDER;
import static com.google.devtools.build.lib.rules.java.proto.JavaProtoLibraryTransitiveFilesToBuildProvider.GET_JARS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
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

  public static final Attribute.LateBoundLabel<BuildConfiguration> PROTO_TOOLCHAIN_LABEL =
      new Attribute.LateBoundLabel<BuildConfiguration>(
          "//tools/proto/toolchains:javalite", ProtoConfiguration.class) {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          return configuration.getFragment(ProtoConfiguration.class).protoToolchainForJavaLite();
        }
      };

  private final JavaSemantics javaSemantics;

  @Nullable private final String jacocoLabel;

  public JavaLiteProtoAspect(JavaSemantics javaSemantics, @Nullable String jacocoLabel) {
    this.javaSemantics = javaSemantics;
    this.jacocoLabel = jacocoLabel;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder aspect =
        new ConfiguredAspect.Builder(getClass().getSimpleName(), ruleContext);

    // Get SupportData, which is provided by the proto_library rule we attach to.
    SupportData supportData =
        checkNotNull(base.getProvider(ProtoSupportDataProvider.class)).getSupportData();

    aspect.addProviders(new Impl(ruleContext, supportData, javaSemantics).createProviders());

    return aspect.build();
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    AspectDefinition.Builder result =
        new AspectDefinition.Builder(getClass().getSimpleName())
            .attributeAspect("deps", this)
            .requiresConfigurationFragments(JavaConfiguration.class, ProtoConfiguration.class)
            .requireProvider(ProtoSourcesProvider.class)
            .add(
                attr(PROTO_TOOLCHAIN_ATTR, LABEL)
                    .mandatoryNativeProviders(
                        ImmutableList.<Class<? extends TransitiveInfoProvider>>of(
                            ProtoLangToolchainProvider.class))
                    .value(PROTO_TOOLCHAIN_LABEL))
            .add(attr(":host_jdk", LABEL).cfg(HOST).value(JavaSemantics.HOST_JDK))
            .add(
                attr(":java_toolchain", LABEL)
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

    Impl(
        final RuleContext ruleContext,
        final SupportData supportData,
        JavaSemantics javaSemantics) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.javaSemantics = javaSemantics;

      dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              Iterables.<JavaCompilationArgsAspectProvider, JavaCompilationArgsProvider>transform(
                  this.<JavaCompilationArgsAspectProvider>getDeps(
                      JavaCompilationArgsAspectProvider.class),
                  GET_PROVIDER));
    }

    TransitiveInfoProviderMap createProviders() {
      TransitiveInfoProviderMap.Builder result = TransitiveInfoProviderMap.builder();

      // Represents the result of compiling the code generated for this proto, including all of its
      // dependencies.
      JavaCompilationArgsProvider generatedCompilationArgsProvider;

      // The jars that this proto and its dependencies produce. Used to roll-up jars up to the
      // java_proto_library, to be put into filesToBuild.
      NestedSetBuilder<Artifact> transitiveOutputJars =
          NestedSetBuilder.fromNestedSets(
              transform(getDeps(JavaProtoLibraryTransitiveFilesToBuildProvider.class), GET_JARS));

      if (supportData.hasProtoSources()) {
        Artifact sourceJar = getSourceJarArtifact();
        createProtoCompileAction(sourceJar);
        Artifact outputJar = getOutputJarArtifact();

        generatedCompilationArgsProvider = createJavaCompileAction(sourceJar, outputJar);

        NestedSet<Artifact> javaSourceJars =
            NestedSetBuilder.<Artifact>stableOrder().add(sourceJar).build();
        transitiveOutputJars.add(outputJar);

        result.add(
            new JavaSourceJarsAspectProvider(
                JavaSourceJarsProvider.create(
                    NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER), javaSourceJars)));
      } else {
        // No sources - this proto_library is an alias library, which exports its dependencies.
        // Simply propagate the compilation-args from its dependencies.
        generatedCompilationArgsProvider = dependencyCompilationArgs;
      }

      return result
          .add(
              new JavaProtoLibraryTransitiveFilesToBuildProvider(transitiveOutputJars.build()),
              new JavaCompilationArgsAspectProvider(generatedCompilationArgsProvider))
          .build();
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ProtoCompileActionBuilder.registerActions(
          ruleContext,
          ImmutableList.of(
              new ProtoCompileActionBuilder.ToolchainInvocation(
                  "javalite", getProtoToolchainProvider(), sourceJar.getExecPathString())),
          supportData,
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
              .setJavacOpts(getAndroidCompatibleJavacOpts());
      helper.addDep(dependencyCompilationArgs);
      TransitiveInfoCollection runtime = getProtoToolchainProvider().runtime();
      if (runtime != null) {
        helper.addDep(runtime.getProvider(JavaCompilationArgsProvider.class));
      }
      helper.setCompilationStrictDepsMode(StrictDepsMode.OFF);
      JavaCompilationArgs artifacts = helper.build(javaSemantics);
      return helper.buildCompilationArgsProvider(artifacts, true /* isReportedAsStrict */);
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

    /**
     * Returns javacopts for compiling the Java source files generated by the proto compiler.
     * Ensures that they are compiled so that they can be used by Android targets.
     *
     * <p>See java_toolchain.compatible_javacopts for the javacopts required for android.
     */
    private ImmutableList<String> getAndroidCompatibleJavacOpts() {
      JavaToolchainProvider toolchain = JavaToolchainProvider.fromRuleContext(ruleContext);
      ImmutableList.Builder<String> listBuilder = ImmutableList.builder();
      listBuilder.addAll(toolchain.getJavacOptions());
      // TODO(b/30890416): Get this from AndroidSemantics.getJavacArguments()
      listBuilder.addAll(
          toolchain.getCompatibleJavacOptions("android"));
      return listBuilder.build();
    }

    private <C extends TransitiveInfoProvider> Iterable<C> getDeps(Class<C> clazz) {
      return ruleContext.getPrerequisites("deps", TARGET, clazz);
    }
  }
}
