// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;

/** Implementation for the java_library rule. */
public class JavaLibrary implements RuleConfiguredTargetFactory {
  private final JavaSemantics semantics;

  protected JavaLibrary(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    JavaCommon common = new JavaCommon(ruleContext, semantics);
    return init(
        ruleContext,
        common,
        /* includeGeneratedExtensionRegistry = */ false,
        /* isJavaPluginRule = */ false);
  }

  final ConfiguredTarget init(
      RuleContext ruleContext,
      final JavaCommon common,
      boolean includeGeneratedExtensionRegistry,
      boolean isJavaPluginRule)
      throws InterruptedException, ActionConflictException {
    semantics.checkDependencyRuleKinds(ruleContext);
    JavaTargetAttributes.Builder attributesBuilder = common.initCommon();

    // Collect the transitive dependencies.
    JavaCompilationHelper helper =
        new JavaCompilationHelper(ruleContext, semantics, common.getJavacOpts(), attributesBuilder);
    helper.addLibrariesToAttributes(common.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));

    if (ruleContext.hasErrors()) {
      return null;
    }

    semantics.checkRule(ruleContext, common);

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();

    if (ruleContext.hasErrors()) {
      common.setJavaCompilationArtifacts(JavaCompilationArtifacts.EMPTY);
      return null;
    }

    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();

    JavaTargetAttributes attributes = attributesBuilder.build();
    if (attributes.hasMessages()) {
      helper.setTranslations(
          semantics.translate(ruleContext, javaConfig, attributes.getMessages()));
    }

    ruleContext.checkSrcsSamePackage(true);

    Artifact jar = null;

    Artifact srcJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_LIBRARY_SOURCE_JAR);

    NestedSet<Artifact> transitiveSourceJars = common.collectTransitiveSourceJars(srcJar);
    JavaSourceJarsProvider.Builder sourceJarsProviderBuilder =
        JavaSourceJarsProvider.builder()
            .addSourceJar(srcJar)
            .addAllTransitiveSourceJars(transitiveSourceJars);

    Artifact classJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_LIBRARY_CLASS_JAR);

    if (attributes.hasSources() || attributes.hasResources()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
      jar = classJar;
    }

    filesBuilder.add(classJar);

    JavaCompileOutputs<Artifact> outputs = helper.createOutputs(classJar);
    javaArtifactsBuilder.setCompileTimeDependencies(outputs.depsProto());
    helper.createCompileAction(outputs);
    helper.createSourceJarAction(srcJar, outputs.genSource());

    Artifact iJar = null;
    if (attributes.hasSources() && jar != null) {
      iJar = helper.createCompileTimeJarAction(jar, javaArtifactsBuilder);
    }
    JavaRuleOutputJarsProvider.Builder ruleOutputJarsProviderBuilder =
        JavaRuleOutputJarsProvider.builder()
            .addOutputJar(classJar, iJar, outputs.manifestProto(), ImmutableList.of(srcJar))
            .setJdeps(outputs.depsProto())
            .setNativeHeaders(outputs.nativeHeader());

    GeneratedExtensionRegistryProvider generatedExtensionRegistryProvider = null;
    if (includeGeneratedExtensionRegistry) {
      generatedExtensionRegistryProvider =
          semantics.createGeneratedExtensionRegistry(
              ruleContext,
              common,
              filesBuilder,
              javaArtifactsBuilder,
              ruleOutputJarsProviderBuilder,
              sourceJarsProviderBuilder);
    }

    boolean neverLink = JavaCommon.isNeverLink(ruleContext);
    JavaCompilationArtifacts javaArtifacts = javaArtifactsBuilder.build();
    common.setJavaCompilationArtifacts(javaArtifacts);
    common.setClassPathFragment(
        new ClasspathConfiguredFragment(
            javaArtifacts, attributes, neverLink, helper.getBootclasspathOrDefault()));

    JavaCompilationArgsProvider javaCompilationArgs =
        common.collectJavaCompilationArgs(
            neverLink, /* srcLessDepsExport= */ false, /* javaProtoLibraryStrictDeps= */ false);
    JavaStrictCompilationArgsProvider strictJavaCompilationArgs =
        new JavaStrictCompilationArgsProvider(
            common.collectJavaCompilationArgs(
                neverLink, /* srcLessDepsExport= */ false, /* javaProtoLibraryStrictDeps= */ true));
    NestedSet<LibraryToLink> transitiveJavaNativeLibraries =
        common.collectTransitiveJavaNativeLibraries();

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);

    semantics.addProviders(ruleContext, common, outputs.genSource(), builder);
    if (generatedExtensionRegistryProvider != null) {
      builder.addNativeDeclaredProvider(generatedExtensionRegistryProvider);
    }

    JavaCompilationArgsProvider compilationArgsProvider = javaCompilationArgs;
    JavaSourceJarsProvider sourceJarsProvider = sourceJarsProviderBuilder.build();
    JavaRuleOutputJarsProvider ruleOutputJarsProvider = ruleOutputJarsProviderBuilder.build();

    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();

    common.addTransitiveInfoProviders(builder, javaInfoBuilder, filesToBuild, classJar);
    common.addGenJarsProvider(builder, javaInfoBuilder, outputs.genClass(), outputs.genSource());

    NestedSet<Artifact> proguardSpecs = new ProguardLibrary(ruleContext).collectProguardSpecs();

    JavaPluginInfoProvider pluginInfoProvider =
        isJavaPluginRule
            // For java_plugin we create the provider with content retrieved from the rule
            // attributes.
            ? common.getJavaPluginInfoProvider(ruleContext)
            // For java_library we add the transitive plugins from plugins and exported_plugins
            // attrs.
            : JavaCommon.getTransitivePlugins(ruleContext);

    // java_library doesn't need to return JavaRunfilesProvider
    JavaInfo javaInfo =
        javaInfoBuilder
            .addProvider(JavaCompilationArgsProvider.class, compilationArgsProvider)
            .addProvider(JavaStrictCompilationArgsProvider.class, strictJavaCompilationArgs)
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider)
            // TODO(bazel-team): this should only happen for java_plugin
            .addProvider(JavaPluginInfoProvider.class, pluginInfoProvider)
            .maybeTransitiveOnlyRuntimeJarsToJavaInfo(common.getDependencies(), true)
            .setRuntimeJars(javaArtifacts.getRuntimeJars())
            .setJavaConstraints(JavaCommon.getConstraints(ruleContext))
            .setNeverlink(neverLink)
            .build();

    builder
        .addProvider(
            RunfilesProvider.simple(
                JavaCommon.getRunfiles(ruleContext, semantics, javaArtifacts, neverLink)))
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(new JavaNativeLibraryInfo(transitiveJavaNativeLibraries))
        .addProvider(JavaSourceInfoProvider.fromJavaTargetAttributes(attributes, semantics))
        .addNativeDeclaredProvider(new ProguardSpecProvider(proguardSpecs))
        .addNativeDeclaredProvider(javaInfo)
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, proguardSpecs);

    if (ruleContext.hasErrors()) {
      return null;
    }

    return builder.build();
  }
}
