// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Implementation for the java_library rule.
 */
public class JavaLibrary implements RuleConfiguredTargetFactory {
  private final JavaSemantics semantics;

  protected JavaLibrary(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    JavaCommon common = new JavaCommon(ruleContext, semantics);
    RuleConfiguredTargetBuilder builder = init(ruleContext, common);
    return builder != null ? builder.build() : null;
  }

  public RuleConfiguredTargetBuilder init(RuleContext ruleContext, final JavaCommon common)
      throws InterruptedException {
    common.initializeJavacOpts();
    JavaTargetAttributes.Builder attributesBuilder = common.initCommon();

    // Collect the transitive dependencies.
    JavaCompilationHelper helper = new JavaCompilationHelper(
        ruleContext, semantics, common.getJavacOpts(), attributesBuilder);
    helper.addLibrariesToAttributes(common.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));
    helper.addProvidersToAttributes(common.compilationArgsFromSources(),
        JavaCommon.isNeverLink(ruleContext));

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

    JavaTargetAttributes attributes = helper.getAttributes();
    if (attributes.hasJarFiles()) {
      // This rule is repackaging some source jars as a java library.
      Set<Artifact> jarFiles = attributes.getJarFiles();
      javaArtifactsBuilder.addRuntimeJars(jarFiles);
      javaArtifactsBuilder.addCompileTimeJars(attributes.getCompileTimeJarFiles());

      filesBuilder.addAll(jarFiles);
    }
    if (attributes.hasMessages()) {
      helper.addTranslations(semantics.translate(ruleContext, javaConfig,
          attributes.getMessages()));
    }

    ruleContext.checkSrcsSamePackage(true);

    Artifact jar = null;

    Artifact srcJar = ruleContext.getImplicitOutputArtifact(
        JavaSemantics.JAVA_LIBRARY_SOURCE_JAR);

    Artifact classJar = ruleContext.getImplicitOutputArtifact(
        JavaSemantics.JAVA_LIBRARY_CLASS_JAR);

    if (attributes.hasSourceFiles() || attributes.hasSourceJars() || attributes.hasResources()
        || attributes.hasMessages()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
      jar = classJar;
    }

    filesBuilder.add(classJar);

    Artifact manifestProtoOutput = helper.createManifestProtoOutput(classJar);

    // The gensrc jar is created only if the target uses annotation processing.
    // Otherwise, it is null, and the source jar action will not depend on the compile action.
    Artifact genSourceJar = null;
    Artifact genClassJar = null;
    if (helper.usesAnnotationProcessing()) {
      genClassJar = helper.createGenJar(classJar);
      genSourceJar = helper.createGensrcJar(classJar);
      helper.createGenJarAction(classJar, manifestProtoOutput, genClassJar);
    }

    Artifact outputDepsProto = helper.createOutputDepsProtoArtifact(classJar, javaArtifactsBuilder);

    helper.createCompileActionWithInstrumentation(classJar, manifestProtoOutput, genSourceJar,
        outputDepsProto, javaArtifactsBuilder);
    helper.createSourceJarAction(srcJar, genSourceJar);

    if ((attributes.hasSourceFiles() || attributes.hasSourceJars()) && jar != null) {
      helper.createCompileTimeJarAction(jar, outputDepsProto,
          javaArtifactsBuilder);
    }

    boolean neverLink = JavaCommon.isNeverLink(ruleContext);
    common.setJavaCompilationArtifacts(javaArtifactsBuilder.build());
    common.setClassPathFragment(new ClasspathConfiguredFragment(
        common.getJavaCompilationArtifacts(), attributes, neverLink));
    CppCompilationContext transitiveCppDeps = common.collectTransitiveCppDeps();

    NestedSet<Artifact> transitiveSourceJars = common.collectTransitiveSourceJars(srcJar);

    // If sources are empty, treat this library as a forwarding node for dependencies.
    JavaCompilationArgs javaCompilationArgs = common.collectJavaCompilationArgs(
        false, neverLink, common.compilationArgsFromSources());
    JavaCompilationArgs recursiveJavaCompilationArgs = common.collectJavaCompilationArgs(
        true, neverLink, common.compilationArgsFromSources());
    NestedSet<Artifact> compileTimeJavaDepArtifacts = common.collectCompileTimeDependencyArtifacts(
        common.getJavaCompilationArtifacts().getCompileTimeDependencyArtifact());
    NestedSet<Artifact> runTimeJavaDepArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    NestedSet<LinkerInput> transitiveJavaNativeLibraries =
        common.collectTransitiveJavaNativeLibraries();

    ImmutableList<String> exportedProcessorClasses = ImmutableList.of();
    NestedSet<Artifact> exportedProcessorClasspath =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    ImmutableList.Builder<String> processorClasses = ImmutableList.builder();
    NestedSetBuilder<Artifact> processorClasspath = NestedSetBuilder.naiveLinkOrder();
    for (JavaPluginInfoProvider provider : Iterables.concat(
        common.getPluginInfoProvidersForAttribute("exported_plugins", Mode.HOST),
        common.getPluginInfoProvidersForAttribute("exports", Mode.TARGET))) {
      processorClasses.addAll(provider.getProcessorClasses());
      processorClasspath.addTransitive(provider.getProcessorClasspath());
    }
    exportedProcessorClasses = processorClasses.build();
    exportedProcessorClasspath = processorClasspath.build();

    CcLinkParamsStore ccLinkParamsStore = new CcLinkParamsStore() {
      @Override
      protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                             boolean linkShared) {
        builder.addTransitiveTargets(common.targetsTreatedAsDeps(ClasspathType.BOTH),
            JavaCcLinkParamsProvider.TO_LINK_PARAMS, CcLinkParamsProvider.TO_LINK_PARAMS);
      }
    };

    // The "neverlink" attribute is transitive, so we don't add any
    // runfiles from this target or its dependencies.
    Runfiles runfiles = Runfiles.EMPTY;
    if (!neverLink) {
      Runfiles.Builder runfilesBuilder = new Runfiles.Builder(ruleContext.getWorkspaceName())
          .addArtifacts(common.getJavaCompilationArtifacts().getRuntimeJars());
      runfilesBuilder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
      runfilesBuilder.add(ruleContext, JavaRunfilesProvider.TO_RUNFILES);

      List<TransitiveInfoCollection> depsForRunfiles = new ArrayList<>();
      if (ruleContext.getRule().isAttrDefined("runtime_deps", BuildType.LABEL_LIST)) {
        depsForRunfiles.addAll(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET));
      }
      if (ruleContext.getRule().isAttrDefined("exports", BuildType.LABEL_LIST)) {
        depsForRunfiles.addAll(ruleContext.getPrerequisites("exports", Mode.TARGET));
      }

      runfilesBuilder.addTargets(depsForRunfiles, RunfilesProvider.DEFAULT_RUNFILES);
      runfilesBuilder.addTargets(depsForRunfiles, JavaRunfilesProvider.TO_RUNFILES);

      TransitiveInfoCollection launcher = JavaHelper.launcherForTarget(semantics, ruleContext);
      if (launcher != null) {
        runfilesBuilder.addTarget(launcher, RunfilesProvider.DATA_RUNFILES);
      }

      semantics.addRunfilesForLibrary(ruleContext, runfilesBuilder);
      runfiles = runfilesBuilder.build();
    }

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext);

    semantics.addProviders(
        ruleContext, common, ImmutableList.<String>of(), classJar, srcJar, 
        genClassJar, genSourceJar, ImmutableMap.<Artifact, Artifact>of(), 
        helper, filesBuilder, builder);

    NestedSet<Artifact> filesToBuild = filesBuilder.build();
    common.addTransitiveInfoProviders(builder, filesToBuild, classJar);
    common.addGenJarsProvider(builder, genClassJar, genSourceJar);

    builder
        .add(JavaRuleOutputJarsProvider.class, new JavaRuleOutputJarsProvider(classJar, srcJar))
        .add(JavaRuntimeJarProvider.class,
            new JavaRuntimeJarProvider(common.getJavaCompilationArtifacts().getRuntimeJars()))
        .add(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setFilesToBuild(filesToBuild)
        .addSkylarkTransitiveInfo(JavaSkylarkApiProvider.NAME, new JavaSkylarkApiProvider())
        .add(JavaNeverlinkInfoProvider.class, new JavaNeverlinkInfoProvider(neverLink))
        .add(CppCompilationContext.class, transitiveCppDeps)
        .add(JavaCompilationArgsProvider.class, new JavaCompilationArgsProvider(
            javaCompilationArgs, recursiveJavaCompilationArgs,
            compileTimeJavaDepArtifacts, runTimeJavaDepArtifacts))
        .add(CcLinkParamsProvider.class, new CcLinkParamsProvider(ccLinkParamsStore))
        .add(JavaNativeLibraryProvider.class, new JavaNativeLibraryProvider(
            transitiveJavaNativeLibraries))
        .add(JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(attributes, semantics))
        .add(JavaSourceJarsProvider.class, new JavaSourceJarsProvider(
            transitiveSourceJars, ImmutableList.of(srcJar)))
        // TODO(bazel-team): this should only happen for java_plugin
        .add(JavaPluginInfoProvider.class, new JavaPluginInfoProvider(
            exportedProcessorClasses, exportedProcessorClasspath))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars);

    if (ruleContext.hasErrors()) {
      return null;
    }

    return builder;
  }
}
