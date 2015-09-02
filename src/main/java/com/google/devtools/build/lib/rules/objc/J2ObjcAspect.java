// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * J2ObjC transpilation aspect for Java rules.
 */
public class J2ObjcAspect implements ConfiguredAspectFactory {
  public static final String NAME = "J2ObjcAspect";
  /**
   * Adds the attribute aspect args to the given AspectDefinition.Builder.
   */
  protected AspectDefinition.Builder addAttributeAspects(AspectDefinition.Builder builder) {
    return builder.attributeAspect("deps", J2ObjcAspect.class)
        .attributeAspect("exports", J2ObjcAspect.class)
        .attributeAspect("runtime_deps", J2ObjcAspect.class);
  }

  @Override
  public AspectDefinition getDefinition() {
    return addAttributeAspects(new AspectDefinition.Builder("J2ObjCAspect"))
        .requireProvider(JavaSourceInfoProvider.class)
        .requireProvider(JavaCompilationArgsProvider.class)
        .add(attr("$j2objc", LABEL).cfg(HOST).exec()
            .value(parseLabel("//tools/j2objc:j2objc_deploy.jar")))
        .add(attr("$j2objc_wrapper", LABEL)
            .allowedFileTypes(FileType.of(".py"))
            .cfg(HOST)
            .exec()
            .singleArtifact()
            .value(parseLabel("//tools/j2objc:j2objc_wrapper")))
        .build();
  }

  private static Label parseLabel(String from) {
    try {
      return Label.parseAbsolute(from);
    } catch (SyntaxException e) {
      throw new IllegalArgumentException(from);
    }
  }

  @Override
  public Aspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters parameters) {
    Aspect.Builder builder = new Aspect.Builder(NAME);

    JavaCompilationArgsProvider compilationArgsProvider =
        base.getProvider(JavaCompilationArgsProvider.class);
    JavaSourceInfoProvider sourceInfoProvider =
        base.getProvider(JavaSourceInfoProvider.class);

    ImmutableSet<Artifact> javaInputFiles = ImmutableSet.<Artifact>builder()
        .addAll(sourceInfoProvider.getSourceFiles())
        .addAll(sourceInfoProvider.getSourceJars())
        .addAll(sourceInfoProvider.getSourceJarsForJarFiles())
        .build();

    NestedSetBuilder<Artifact> depsHeaderMappingsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> depsClassMappingsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> depsDependencyMappingsBuilder = NestedSetBuilder.stableOrder();

    for (J2ObjcMappingFileProvider provider : getJ2ObjCMappings(ruleContext)) {
      depsHeaderMappingsBuilder.addTransitive(provider.getHeaderMappingFiles());
      depsClassMappingsBuilder.addTransitive(provider.getClassMappingFiles());
      depsDependencyMappingsBuilder.addTransitive(provider.getDependencyMappingFiles());
    }

    NestedSet<Artifact> depsHeaderMappings = depsHeaderMappingsBuilder.build();
    NestedSet<Artifact> depsClassMappings = depsClassMappingsBuilder.build();
    NestedSet<Artifact> depsDependencyMappings = depsDependencyMappingsBuilder.build();

    J2ObjcSrcsProvider.Builder srcsBuilder = new J2ObjcSrcsProvider.Builder();
    J2ObjcMappingFileProvider j2ObjcMappingFileProvider;

    if (!javaInputFiles.isEmpty()) {
      J2ObjcSource j2ObjcSource = buildJ2ObjcSource(ruleContext, javaInputFiles);

      createJ2ObjcTranspilationAction(ruleContext, depsHeaderMappings, depsClassMappings,
          javaInputFiles, compilationArgsProvider, j2ObjcSource);

      // J2ObjC merges all input header mapping files into the output header mapping file, so we
      // only need to export the output header mapping file here.
      NestedSet<Artifact> headerMappingFiles = NestedSetBuilder.<Artifact>stableOrder()
          .add(j2ObjcOutputHeaderMappingFile(ruleContext))
          .build();
      NestedSet<Artifact> dependencyMappingFiles = NestedSetBuilder.<Artifact>stableOrder()
          .add(j2ObjcOutputDependencyMappingFile(ruleContext))
          .addTransitive(depsDependencyMappings)
          .build();

      srcsBuilder.addSource(j2ObjcSource);
      j2ObjcMappingFileProvider = new J2ObjcMappingFileProvider(
          headerMappingFiles, depsClassMappings, dependencyMappingFiles);
    } else {
      j2ObjcMappingFileProvider = new J2ObjcMappingFileProvider(
          depsHeaderMappings, depsClassMappings, depsDependencyMappings);
    }

    for (J2ObjcSrcsProvider provider :
        ruleContext.getPrerequisites("exports", Mode.TARGET, J2ObjcSrcsProvider.class)) {
      srcsBuilder.addTransitive(provider);
    }

    srcsBuilder.addTransitiveFromDeps(ruleContext);

    return builder
        .addProvider(J2ObjcSrcsProvider.class, srcsBuilder.build())
        .addProvider(J2ObjcMappingFileProvider.class, j2ObjcMappingFileProvider)
        .build();
  }

  private static void createJ2ObjcTranspilationAction(
      RuleContext ruleContext,
      NestedSet<Artifact> depsHeaderMappingFiles,
      NestedSet<Artifact> depsClassMappingFiles,
      Iterable<Artifact> sources,
      JavaCompilationArgsProvider compArgsProvider,
      J2ObjcSource j2ObjcSource) {
    CustomCommandLine.Builder argBuilder = CustomCommandLine.builder();
    PathFragment javaExecutable = ruleContext.getHostConfiguration().getFragment(Jvm.class)
        .getJavaExecutable();
    argBuilder.add("--java").add(javaExecutable.getPathString());

    Artifact j2ObjcDeployJar = ruleContext.getPrerequisiteArtifact("$j2objc", Mode.HOST);
    argBuilder.addExecPath("--j2objc", j2ObjcDeployJar);

    argBuilder.add("--main_class").add("com.google.devtools.j2objc.J2ObjC");
    argBuilder.addJoinExecPaths("--translated_source_files", ",", j2ObjcSource.getObjcSrcs());
    argBuilder.add("--objc_file_path").addPath(j2ObjcSource.getObjcFilePath());

    Artifact outputDependencyMappingFile = j2ObjcOutputDependencyMappingFile(ruleContext);
    argBuilder.addExecPath("--output_dependency_mapping_file", outputDependencyMappingFile);

    Iterable<String> translationFlags = ruleContext
        .getFragment(J2ObjcConfiguration.class)
        .getTranslationFlags();
    argBuilder.add(translationFlags);

    if (!depsHeaderMappingFiles.isEmpty()) {
      argBuilder.addJoinExecPaths("--header-mapping", ",", depsHeaderMappingFiles);
    }

    Artifact outputHeaderMappingFile = j2ObjcOutputHeaderMappingFile(ruleContext);
    argBuilder.addExecPath("--output-header-mapping", outputHeaderMappingFile);

    if (!depsClassMappingFiles.isEmpty()) {
      argBuilder.addJoinExecPaths("--mapping", ",", depsClassMappingFiles);
    }

    argBuilder.add("-d").addPath(j2ObjcSource.getObjcFilePath());

    // In J2ObjC, the jars you pass as dependencies must be precisely the same as the
    // jars used to transpile those dependencies--we cannot use ijars here.
    NestedSet<Artifact> compileTimeJars =
        compArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars();
    if (!compileTimeJars.isEmpty()) {
      argBuilder.addJoinExecPaths("-classpath", ":", compileTimeJars);
    }

    argBuilder.addExecPaths(sources);

    Artifact paramFile = j2ObjcOutputParamFile(ruleContext);
    ruleContext.registerAction(new ParameterFileWriteAction(
        ruleContext.getActionOwner(),
        paramFile,
        argBuilder.build(),
        ParameterFile.ParameterFileType.UNQUOTED, ISO_8859_1));

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setMnemonic("TranspilingJ2objc")
        .setExecutable(ruleContext.getPrerequisiteArtifact("$j2objc_wrapper", Mode.HOST))
        .addInput(ruleContext.getPrerequisiteArtifact("$j2objc_wrapper", Mode.HOST))
        .addInput(j2ObjcDeployJar)
        .addInputs(sources)
        .addTransitiveInputs(compileTimeJars)
        .addInputs(JavaCompilationHelper.getHostJavabaseInputs(ruleContext))
        .addTransitiveInputs(depsHeaderMappingFiles)
        .addTransitiveInputs(depsClassMappingFiles)
        .addInput(paramFile)
        .setCommandLine(CustomCommandLine.builder()
            .addPaths("@%s", paramFile.getExecPath())
            .build())
        .addOutputs(j2ObjcSource.getObjcSrcs())
        .addOutputs(j2ObjcSource.getObjcHdrs())
        .addOutput(outputHeaderMappingFile)
        .addOutput(outputDependencyMappingFile);

    ruleContext.registerAction(builder.build(ruleContext));
  }

  private static List<? extends J2ObjcMappingFileProvider> getJ2ObjCMappings(RuleContext context) {
    ImmutableList.Builder<J2ObjcMappingFileProvider> mappingFileProviderBuilder =
        new ImmutableList.Builder<>();
    addJ2ObjCMappingsForAttribute(mappingFileProviderBuilder, context, "deps");
    addJ2ObjCMappingsForAttribute(mappingFileProviderBuilder, context, "runtime_deps");
    addJ2ObjCMappingsForAttribute(mappingFileProviderBuilder, context, "exports");
    return mappingFileProviderBuilder.build();
  }

  private static void addJ2ObjCMappingsForAttribute(
      ImmutableList.Builder<J2ObjcMappingFileProvider> builder, RuleContext context,
      String attributeName) {
    if (context.attributes().has(attributeName, Type.LABEL_LIST)) {
      for (TransitiveInfoCollection dependencyInfoDatum :
          context.getPrerequisites(attributeName, Mode.TARGET)) {
        J2ObjcMappingFileProvider provider =
            dependencyInfoDatum.getProvider(J2ObjcMappingFileProvider.class);
        if (provider != null) {
          builder.add(provider);
        }
      }
    }
  }

  private static Artifact j2ObjcOutputHeaderMappingFile(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".mapping.j2objc");
  }

  private static Artifact j2ObjcOutputDependencyMappingFile(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".dependency_mapping.j2objc");
  }

  private static Artifact j2ObjcOutputParamFile(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".param.j2objc");
  }

  private J2ObjcSource buildJ2ObjcSource(RuleContext ruleContext,
      Iterable<Artifact> javaInputSourceFiles) {
    PathFragment objcFileRootRelativePath = ruleContext.getUniqueDirectory("_j2objc");
    PathFragment objcFilePath = ruleContext
        .getConfiguration()
        .getBinFragment()
        .getRelative(objcFileRootRelativePath);
    Iterable<Artifact> objcSrcs = getOutputObjcFiles(ruleContext, javaInputSourceFiles,
        objcFileRootRelativePath, ".m");
    Iterable<Artifact> objcHdrs = getOutputObjcFiles(ruleContext, javaInputSourceFiles,
        objcFileRootRelativePath, ".h");
    return new J2ObjcSource(ruleContext.getRule().getLabel(), objcSrcs, objcHdrs, objcFilePath,
        SourceType.JAVA);
  }

  private Iterable<Artifact> getOutputObjcFiles(RuleContext ruleContext,
      Iterable<Artifact> javaSrcs, PathFragment objcFileRootRelativePath, String suffix) {
    ImmutableList.Builder<Artifact> objcSources = ImmutableList.builder();

    for (Artifact javaSrc : javaSrcs) {
      objcSources.add(ruleContext.getRelatedArtifact(
          objcFileRootRelativePath.getRelative(javaSrc.getExecPath()), suffix));
    }

    return objcSources.build();
  }
}
