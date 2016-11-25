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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.rules.java.proto.JavaCompilationArgsAspectProvider.GET_PROVIDER;
import static com.google.devtools.build.lib.rules.java.proto.JavaProtoLibraryTransitiveFilesToBuildProvider.GET_JARS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
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
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.rules.proto.ProtoSupportDataProvider;
import com.google.devtools.build.lib.rules.proto.SupportData;
import java.util.List;
import javax.annotation.Nullable;

/** An Aspect which JavaProtoLibrary injects to build Java SPEED protos. */
public class JavaProtoAspect extends NativeAspectClass implements ConfiguredAspectFactory {

  private static final String SPEED_PROTO_TOOLCHAIN_ATTR = ":aspect_java_proto_toolchain";

  private static final String SPEED_PROTO_RUNTIME_ATTR = "$aspect_java_lib";
  private static final String SPEED_PROTO_RUNTIME_LABEL = "//external:protobuf/java_runtime";

  /**
   * The attribute name for holding a list of protos for which no code should be generated because
   * the proto-runtime already contains them.
   */
  private static final String PROTO_SOURCE_FILE_BLACKLIST_ATTR = ":proto_source_file_blacklist";

  private static final Attribute.LateBoundLabelList<BuildConfiguration> BLACKLISTED_PROTOS =
      new Attribute.LateBoundLabelList<BuildConfiguration>(
          ImmutableList.<Label>of(), ProtoConfiguration.class) {
        @Override
        public List<Label> resolve(
            Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          return configuration
              .getFragment(ProtoConfiguration.class)
              .protoCompilerJavaBlacklistedProtos();
        }

        @Override
        public boolean useHostConfiguration() {
          return true;
        }
      };

  private static final Attribute.LateBoundLabel<BuildConfiguration> SPEED_PROTO_TOOLCHAIN_LABEL =
      new Attribute.LateBoundLabel<BuildConfiguration>(
          "//tools/proto/toolchains:java", ProtoConfiguration.class) {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          return configuration.getFragment(ProtoConfiguration.class).protoToolchainForJava();
        }
      };

  private final JavaSemantics javaSemantics;

  @Nullable private final String jacocoLabel;
  private final RpcSupport rpcSupport;

  protected JavaProtoAspect(
      JavaSemantics javaSemantics,
      @Nullable String jacocoLabel,
      RpcSupport rpcSupport) {
    this.javaSemantics = javaSemantics;
    this.jacocoLabel = jacocoLabel;
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
                ruleContext
                    .getFragment(ProtoConfiguration.class, ConfigurationTransition.HOST)
                    .protoCompilerJavaFlags(),
                javaSemantics,
                rpcSupport,
                ruleContext.getFragment(ProtoConfiguration.class).useToolchainForJavaProto())
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
            .add(
                attr(PROTO_SOURCE_FILE_BLACKLIST_ATTR, LABEL_LIST)
                    .cfg(HOST)
                    .value(BLACKLISTED_PROTOS))
            .add(
                attr(SPEED_PROTO_TOOLCHAIN_ATTR, LABEL)
                    // TODO(carmi): reinstate mandatoryNativeProviders(ProtoLangToolchainProvider)
                    // once it's in a Bazel release.
                    .legacyAllowAnyFileType()
                    .value(SPEED_PROTO_TOOLCHAIN_LABEL))
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
    private final String protoCompilerPluginOptions;
    private final boolean useToolchainForJavaProto;

    Impl(
        final RuleContext ruleContext,
        final SupportData supportData,
        String protoCompilerPluginOptions,
        JavaSemantics javaSemantics,
        RpcSupport rpcSupport,
        boolean useToolchainForJavaProto) {
      this.ruleContext = ruleContext;
      this.supportData = supportData;
      this.protoCompilerPluginOptions = protoCompilerPluginOptions;
      this.javaSemantics = javaSemantics;
      this.rpcSupport = rpcSupport;
      this.useToolchainForJavaProto = useToolchainForJavaProto;

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

      if (shouldGenerateCode()) {
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

    /**
     * Decides whether code should be generated for the .proto files in the currently-processed
     * proto_library.
     */
    private boolean shouldGenerateCode() {
      if (!supportData.hasProtoSources()) {
        return false;
      }

      final ProtoSourceFileBlacklist protoBlackList;
      if (useToolchainForJavaProto) {
        NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
        blacklistedProtos.addTransitive(getProtoToolchainProvider().blacklistedProtos());
        blacklistedProtos.addTransitive(rpcSupport.getBlacklist(ruleContext));

        protoBlackList = new ProtoSourceFileBlacklist(ruleContext, blacklistedProtos.build());
      } else {
        protoBlackList =
            new ProtoSourceFileBlacklist(
                ruleContext,
                ruleContext
                    .getPrerequisiteArtifacts(PROTO_SOURCE_FILE_BLACKLIST_ATTR, Mode.HOST)
                    .list());
      }

      return protoBlackList.checkSrcs(supportData.getDirectProtoSources(), "java_proto_library");
    }

    private void createProtoCompileAction(Artifact sourceJar) {
      if (useToolchainForJavaProto) {
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
            ImmutableList.of(sourceJar),
            "Java (Immutable)",
            rpcSupport.allowServices(ruleContext));
      } else {
        ProtoCompileActionBuilder actionBuilder =
            new ProtoCompileActionBuilder(
                    ruleContext, supportData, "Java", "java", ImmutableList.of(sourceJar))
                .allowServices(true)
                .setLangParameter(
                    String.format(protoCompilerPluginOptions, sourceJar.getExecPathString()));
        rpcSupport.mutateProtoCompileAction(ruleContext, sourceJar, actionBuilder);
        ruleContext.registerAction(actionBuilder.build());
      }
    }

    private JavaCompilationArgsProvider createJavaCompileAction(
        Artifact sourceJar, Artifact outputJar) {
      JavaLibraryHelper helper =
          new JavaLibraryHelper(ruleContext)
              .setOutput(outputJar)
              .addSourceJars(sourceJar)
              .setJavacOpts(ProtoJavacOpts.constructJavacOpts(ruleContext));
      helper
          .addDep(dependencyCompilationArgs)
          .setCompilationStrictDepsMode(StrictDepsMode.OFF);
      if (useToolchainForJavaProto) {
        TransitiveInfoCollection runtime = getProtoToolchainProvider().runtime();
        if (runtime != null) {
          helper.addDep(runtime.getProvider(JavaCompilationArgsProvider.class));
        }
      } else {
        helper.addDep(
            ruleContext.getPrerequisite(
                SPEED_PROTO_RUNTIME_ATTR, Mode.TARGET, JavaCompilationArgsProvider.class));
      }

      rpcSupport.mutateJavaCompileAction(ruleContext, helper);
      return helper.buildCompilationArgsProvider(
          helper.build(
              javaSemantics,
              JavaCompilationHelper.getJavaToolchainProvider(ruleContext),
              JavaCompilationHelper.getHostJavabaseInputsNonStatic(ruleContext),
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

    private <C extends TransitiveInfoProvider> Iterable<C> getDeps(Class<C> clazz) {
      return ruleContext.getPrerequisites("deps", TARGET, clazz);
    }
  }
}
