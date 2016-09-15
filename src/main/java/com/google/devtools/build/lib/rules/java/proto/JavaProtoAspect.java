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
import static com.google.devtools.build.lib.cmdline.Label.PARSE_ABSOLUTE_UNCHECKED;
import static com.google.devtools.build.lib.cmdline.Label.parseAbsoluteUnchecked;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.java.proto.JavaCompilationArgsAspectProvider.GET_PROVIDER;
import static com.google.devtools.build.lib.rules.java.proto.JavaProtoLibraryTransitiveFilesToBuildProvider.GET_JARS;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** An Aspect which JavaProtoLibrary injects to build Java SPEED protos. */
public class JavaProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  private static final String SPEED_PROTO_RUNTIME_ATTR = "$aspect_java_lib";
  private static final String SPEED_PROTO_RUNTIME_LABEL = "//external:protobuf/java_runtime";

  /**
   * The attribute name for holding a list of protos for which no code should be generated because
   * the proto-runtime already contains them.
   */
  private static final String PROTO_SOURCE_FILE_BLACKLIST_ATTR = "$proto_source_file_blacklist";

  private final JavaSemantics javaSemantics;
  private final ImmutableList<String> protoSourceFileBlacklistLabels;

  @Nullable private final String jacocoLabel;
  private final ImmutableList<String> protoCompilerPluginOptions;
  private final RpcSupport rpcSupport;

  protected JavaProtoAspect(
      JavaSemantics javaSemantics,
      ImmutableList<String> protoSourceFileBlacklistLabels,
      @Nullable String jacocoLabel,
      ImmutableList<String> protoCompilerPluginOptions,
      RpcSupport rpcSupport) {
    this.javaSemantics = javaSemantics;
    this.protoSourceFileBlacklistLabels = protoSourceFileBlacklistLabels;
    this.jacocoLabel = jacocoLabel;
    this.protoCompilerPluginOptions = protoCompilerPluginOptions;
    this.rpcSupport = rpcSupport;
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder aspect =
        new ConfiguredAspect.Builder(getClass().getSimpleName(), ruleContext);

    if (!rpcSupport.checkAttributes(ruleContext, parameters)) {
      return aspect.build();
    }

    // Get SupportData, which is provided by the proto_library rule we attach to.
    SupportData supportData =
        checkNotNull(base.getProvider(ProtoSupportDataProvider.class)).getSupportData();

    aspect.addProviders(
        new Impl(
                ruleContext,
                supportData,
                protoCompilerPluginOptions,
                javaSemantics,
            rpcSupport)
            .createProviders());

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
                attr(SPEED_PROTO_RUNTIME_ATTR, LABEL)
                    .legacyAllowAnyFileType()
                    .value(parseAbsoluteUnchecked(SPEED_PROTO_RUNTIME_LABEL)))
            .add(ProtoSourceFileBlacklist.blacklistFilegroupAttribute(
                    PROTO_SOURCE_FILE_BLACKLIST_ATTR,
                    transformToList(protoSourceFileBlacklistLabels, PARSE_ABSOLUTE_UNCHECKED)))
            .add(attr(":host_jdk", LABEL).cfg(HOST).value(JavaSemantics.HOST_JDK))
            .add(
                attr(":java_toolchain", LABEL)
                    .allowedRuleClasses("java_toolchain")
                    .value(JavaSemantics.JAVA_TOOLCHAIN));

    rpcSupport.mutateAspectDefinition(result, aspectParameters);

    Attribute.Builder<Label> jacocoAttr = attr("$jacoco_instrumentation", LABEL).cfg(HOST);

    if (jacocoLabel != null) {
      jacocoAttr.value(parseAbsoluteUnchecked(jacocoLabel));
    }
    return result.add(jacocoAttr).build();
  }

  /** Like Iterables.transform(), except it returns a List<> instead of an Iterable<>. */
  private static <F, T> List<T> transformToList(
      Iterable<F> fromIterable, Function<? super F, ? extends T> function) {
    return Lists.<T>newArrayList(transform(fromIterable, function));
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
    private final ImmutableList<String> protoCompilerPluginOptions;

    Impl(
        final RuleContext ruleContext,
        final SupportData supportData,
        ImmutableList<String> protoCompilerPluginOptions,
        JavaSemantics javaSemantics,
        RpcSupport rpcSupport) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.protoCompilerPluginOptions = protoCompilerPluginOptions;
      this.javaSemantics = javaSemantics;
      this.rpcSupport = rpcSupport;

      dependencyCompilationArgs =
          JavaCompilationArgsProvider.merge(
              Iterables.<JavaCompilationArgsAspectProvider, JavaCompilationArgsProvider>transform(
                  this.<JavaCompilationArgsAspectProvider>getDeps(
                      JavaCompilationArgsAspectProvider.class),
                  GET_PROVIDER));
    }

    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> createProviders() {
      ImmutableMap.Builder<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> result =
          ImmutableMap.builder();

      // Represents the result of compiling the code generated for this proto, including all of its
      // dependencies.
      JavaCompilationArgsProvider generatedCompilationArgsProvider;

      // The jars that this proto and its dependencies produce. Used to roll-up jars up to the
      // java_proto_library, to be put into filesToBuild.
      NestedSetBuilder<Artifact> transitiveOutputJars =
          NestedSetBuilder.fromNestedSets(
              transform(getDeps(JavaProtoLibraryTransitiveFilesToBuildProvider.class), GET_JARS));

      if (shouldGenerateCode()) {
        Artifact sourceJar = getSourceJarArtifact();
        createProtoCompileAction(sourceJar);
        Artifact outputJar = getOutputJarArtifact();

        generatedCompilationArgsProvider = createJavaCompileAction(sourceJar, outputJar);

        NestedSet<Artifact> javaSourceJars =
            NestedSetBuilder.<Artifact>stableOrder().add(sourceJar).build();
        transitiveOutputJars.add(outputJar);

        result
            .put(
                JavaRuntimeJarAspectProvider.class,
                new JavaRuntimeJarAspectProvider(
                    new JavaRuntimeJarProvider(ImmutableList.of(outputJar))))
            .put(
                JavaSourceJarsAspectProvider.class,
                new JavaSourceJarsAspectProvider(
                    JavaSourceJarsProvider.create(
                        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER), javaSourceJars)));
      } else {
        // No sources - this proto_library is an alias library, which exports its dependencies.
        // Simply propagate the compilation-args from its dependencies.
        generatedCompilationArgsProvider = dependencyCompilationArgs;
      }

      return result
          .put(
              JavaProtoLibraryTransitiveFilesToBuildProvider.class,
              new JavaProtoLibraryTransitiveFilesToBuildProvider(transitiveOutputJars.build()))
          .put(
              JavaCompilationArgsAspectProvider.class,
              new JavaCompilationArgsAspectProvider(generatedCompilationArgsProvider))
          .build();
    }

    /**
     * Decides whether code should be generated for the .proto files in the currently-processed
     * proto_library.
     */
    private boolean shouldGenerateCode() {
      if (!supportData.hasProtoSources()) {
        return false;
      }

      ProtoSourceFileBlacklist protoBlackList =
          new ProtoSourceFileBlacklist(ruleContext, PROTO_SOURCE_FILE_BLACKLIST_ATTR);
      return protoBlackList.checkSrcs(supportData.getDirectProtoSources(), "java_proto_library");
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      ProtoCompileActionBuilder actionBuilder =
          new ProtoCompileActionBuilder(
                  ruleContext, supportData, "Java", "java", ImmutableList.of(sourceJar))
              .allowServices(true)
              .setLangParameter(
                  ProtoCompileActionBuilder.buildProtoArg(
                      "java_out", sourceJar.getExecPathString(), protoCompilerPluginOptions));
      rpcSupport.mutateProtoCompileAction(
          ruleContext, sourceJar, actionBuilder);
      ruleContext.registerAction(actionBuilder.build());
    }

    private JavaCompilationArgsProvider createJavaCompileAction(
        Artifact sourceJar, Artifact outputJar) {
      JavaLibraryHelper helper =
          new JavaLibraryHelper(ruleContext)
              .setOutput(outputJar)
              .addSourceJars(sourceJar)
              .setJavacOpts(constructJavacOpts());
      helper
          .addDep(dependencyCompilationArgs)
          .addDep(
              ruleContext.getPrerequisite(
                  SPEED_PROTO_RUNTIME_ATTR, Mode.TARGET, JavaCompilationArgsProvider.class))
          .setCompilationStrictDepsMode(StrictDepsMode.OFF);
      rpcSupport.mutateJavaCompileAction(ruleContext, helper);
      return helper.buildCompilationArgsProvider(
          helper.build(javaSemantics), true /* isReportedAsStrict */);
    }

    private Artifact getSourceJarArtifact() {
      return ruleContext.getGenfilesArtifact(ruleContext.getLabel().getName() + "-speed-src.jar");
    }

    private Artifact getOutputJarArtifact() {
      return ruleContext.getBinArtifact("lib" + ruleContext.getLabel().getName() + "-speed.jar");
    }

    /**
     * Returns javacopts for compiling the Java source files generated by the proto compiler.
     * Ensures that they are compiled so that they can be used by App Engine targets.
     */
    private ImmutableList<String> constructJavacOpts() {
      JavaToolchainProvider toolchain = JavaToolchainProvider.fromRuleContext(ruleContext);
      ImmutableList.Builder<String> listBuilder = ImmutableList.builder();
      listBuilder.addAll(toolchain.getJavacOptions());
      listBuilder.addAll(toolchain.getCompatibleJavacOptions(JavaSemantics.JAVA7_JAVACOPTS_KEY));
      return listBuilder.build();
    }

    private <C extends TransitiveInfoProvider> Iterable<C> getDeps(Class<C> clazz) {
      return ruleContext.getPrerequisites("deps", TARGET, clazz);
    }
  }
}
