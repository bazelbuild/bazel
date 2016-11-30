// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.actions.Artifact.IS_TREE_ARTIFACT;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaGenJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraCompileArgs;
import com.google.devtools.build.lib.rules.objc.J2ObjcSource.SourceType;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/**
 * J2ObjC transpilation aspect for Java rules.
 */
public class J2ObjcAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "J2ObjcAspect";
  private final String toolsRepository;
  private final AbstractJ2ObjcProtoAspect j2ObjcProtoAspect;

  private static final ExtraCompileArgs EXTRA_COMPILE_ARGS = new ExtraCompileArgs(
      "-fno-strict-overflow");

  public J2ObjcAspect(String toolsRepository, AbstractJ2ObjcProtoAspect j2ObjcProtoAspect) {
    this.toolsRepository = toolsRepository;
    this.j2ObjcProtoAspect = j2ObjcProtoAspect;
  }

  private static final Iterable<Attribute> DEPENDENT_ATTRIBUTES = ImmutableList.of(
      new Attribute(":jre_lib", Mode.TARGET),
      new Attribute("deps", Mode.TARGET),
      new Attribute("exports", Mode.TARGET),
      new Attribute("runtime_deps", Mode.TARGET));

  private static final Label JRE_CORE_LIB =
      Label.parseAbsoluteUnchecked("//third_party/java/j2objc:jre_core_lib");

  private static final Label JRE_EMUL_LIB =
      Label.parseAbsoluteUnchecked("//third_party/java/j2objc:jre_emul_lib");

  private static final LateBoundLabel<BuildConfiguration> JRE_LIB =
      new LateBoundLabel<BuildConfiguration>(JRE_CORE_LIB, J2ObjcConfiguration.class) {
    @Override
    public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
      return configuration.getFragment(J2ObjcConfiguration.class).explicitJreDeps()
          ? JRE_CORE_LIB : JRE_EMUL_LIB;
    }
  };

  /** Adds additional attribute aspects and attributes to the given AspectDefinition.Builder. */
  protected AspectDefinition.Builder addAdditionalAttributes(AspectDefinition.Builder builder) {
    return builder;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return addAdditionalAttributes(new AspectDefinition.Builder(this))
        .attributeAspect("deps", this, j2ObjcProtoAspect)
        .attributeAspect("exports", this, j2ObjcProtoAspect)
        .attributeAspect("runtime_deps", this, j2ObjcProtoAspect)
        .requireProviders(JavaSourceInfoProvider.class, JavaCompilationArgsProvider.class)
        .requiresConfigurationFragments(
            AppleConfiguration.class, J2ObjcConfiguration.class, ObjcConfiguration.class)
        .requiresHostConfigurationFragments(Jvm.class)
        .add(
            attr("$j2objc", LABEL)
                .cfg(HOST)
                .exec()
                .value(
                    Label.parseAbsoluteUnchecked(
                        toolsRepository + "//tools/j2objc:j2objc_deploy.jar")))
        .add(
            attr("$j2objc_wrapper", LABEL)
                .allowedFileTypes(FileType.of(".py"))
                .cfg(HOST)
                .exec()
                .singleArtifact()
                .value(
                    Label.parseAbsoluteUnchecked(
                        toolsRepository + "//tools/j2objc:j2objc_wrapper")))
        .add(
            attr("$jre_emul_jar", LABEL)
                .cfg(HOST)
                .value(
                    Label.parseAbsoluteUnchecked(
                        toolsRepository + "//third_party/java/j2objc:jre_emul.jar")))
        .add(attr(":jre_lib", LABEL).value(JRE_LIB))
        .add(
            attr("$xcrunwrapper", LABEL)
                .cfg(HOST)
                .exec()
                .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/objc:xcrunwrapper")))
        .add(
            attr(ObjcRuleClasses.LIBTOOL_ATTRIBUTE, LABEL)
                .cfg(HOST)
                .exec()
                .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/objc:libtool")))
        .add(
            attr(":xcode_config", LABEL)
                .allowedRuleClasses("xcode_config")
                .checkConstraints()
                .direct_compile_time_input()
                .cfg(HOST)
                .value(new AppleToolchain.XcodeConfigLabel(toolsRepository)))
        .add(
            attr("$zipper", LABEL)
                .cfg(HOST)
                .exec()
                .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/zip:zipper")))
        .build();
  }

  @Override
  public ConfiguredAspect create(
      ConfiguredTarget base, RuleContext ruleContext, AspectParameters parameters)
      throws InterruptedException {
    ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(this, parameters, ruleContext);
    JavaCompilationArgsProvider compilationArgsProvider =
        base.getProvider(JavaCompilationArgsProvider.class);
    JavaSourceInfoProvider sourceInfoProvider =
        base.getProvider(JavaSourceInfoProvider.class);
    JavaGenJarsProvider genJarProvider =
        base.getProvider(JavaGenJarsProvider.class);
    ImmutableSet<Artifact> javaInputFiles = ImmutableSet.<Artifact>builder()
        .addAll(sourceInfoProvider.getSourceFiles())
        .addAll(sourceInfoProvider.getSourceJars())
        .addAll(sourceInfoProvider.getSourceJarsForJarFiles())
        .build();

    Optional<Artifact> genSrcJar;
    boolean annotationProcessingEnabled = ruleContext.getFragment(J2ObjcConfiguration.class)
        .annotationProcessingEnabled();
    if (genJarProvider != null && annotationProcessingEnabled) {
      genSrcJar = Optional.fromNullable(genJarProvider.getGenSourceJar());
    } else {
      genSrcJar = Optional.<Artifact>absent();
    }

    XcodeProvider xcodeProvider;
    ObjcCommon common;

    if (!javaInputFiles.isEmpty()) {
      J2ObjcSource j2ObjcSource = buildJ2ObjcSource(ruleContext, javaInputFiles, genSrcJar);
      J2ObjcMappingFileProvider depJ2ObjcMappingFileProvider =
          depJ2ObjcMappingFileProvider(ruleContext);
      createJ2ObjcTranspilationAction(
          ruleContext,
          depJ2ObjcMappingFileProvider.getHeaderMappingFiles(),
          depJ2ObjcMappingFileProvider.getClassMappingFiles(),
          javaInputFiles,
          compilationArgsProvider,
          j2ObjcSource,
          genSrcJar);

      common = common(
          ruleContext,
          j2ObjcSource.getObjcSrcs(),
          j2ObjcSource.getObjcHdrs(),
          j2ObjcSource.getHeaderSearchPaths(),
          DEPENDENT_ATTRIBUTES);

      xcodeProvider = xcodeProvider(
          ruleContext,
          common,
          j2ObjcSource.getObjcHdrs(),
          j2ObjcSource.getHeaderSearchPaths(),
          DEPENDENT_ATTRIBUTES);

      try {
        new LegacyCompilationSupport(ruleContext)
            .registerCompileAndArchiveActions(common, EXTRA_COMPILE_ARGS)
            .registerFullyLinkAction(common.getObjcProvider(),
                ruleContext.getImplicitOutputArtifact(CompilationSupport.FULLY_LINKED_LIB));
      } catch (RuleErrorException e) {
        ruleContext.ruleError(e.getMessage());
      }
    } else {
      common = common(
          ruleContext,
          ImmutableList.<Artifact>of(),
          ImmutableList.<Artifact>of(),
          ImmutableList.<PathFragment>of(),
          DEPENDENT_ATTRIBUTES);
      xcodeProvider = xcodeProvider(
          ruleContext,
          common,
          ImmutableList.<Artifact>of(),
          ImmutableList.<PathFragment>of(),
          DEPENDENT_ATTRIBUTES);
    }

    return builder
        .addProvider(j2ObjcMappingFileProvider(ruleContext, !javaInputFiles.isEmpty()))
        .addProvider(common.getObjcProvider())
        .addProvider(xcodeProvider)
        .build();
  }

  private J2ObjcMappingFileProvider j2ObjcMappingFileProvider(RuleContext ruleContext,
      boolean hasTranslatedSource) {
    J2ObjcMappingFileProvider depJ2ObjcMappingFileProvider =
        depJ2ObjcMappingFileProvider(ruleContext);
    J2ObjcMappingFileProvider j2ObjcMappingFileProvider = depJ2ObjcMappingFileProvider;
    if (hasTranslatedSource) {
      // J2ObjC merges all input header mapping files into the output header mapping file, so we
      // only need to export the output header mapping file here.
      NestedSet<Artifact> headerMappingFiles = NestedSetBuilder.<Artifact>stableOrder()
          .add(j2ObjcOutputHeaderMappingFile(ruleContext))
          .build();
      NestedSet<Artifact> dependencyMappingFiles = NestedSetBuilder.<Artifact>stableOrder()
          .add(j2ObjcOutputDependencyMappingFile(ruleContext))
          .addTransitive(depJ2ObjcMappingFileProvider.getDependencyMappingFiles())
          .build();

      NestedSet<Artifact> archiveSourceMappingFiles = NestedSetBuilder.<Artifact>stableOrder()
          .add(j2ObjcOutputArchiveSourceMappingFile(ruleContext))
          .addTransitive(depJ2ObjcMappingFileProvider.getArchiveSourceMappingFiles())
          .build();

      j2ObjcMappingFileProvider = new J2ObjcMappingFileProvider(
          headerMappingFiles,
          depJ2ObjcMappingFileProvider.getClassMappingFiles(),
          dependencyMappingFiles,
          archiveSourceMappingFiles);
    }

    return j2ObjcMappingFileProvider;
  }

  private List<Artifact> genJarOutputs(RuleContext ruleContext) {
    return ImmutableList.of(
        j2ObjcGenJarTranslatedSourceFiles(ruleContext),
        j2objcGenJarTranslatedHeaderFiles(ruleContext));
  }

  private List<String> genJarFlags(RuleContext ruleContext) {
    return ImmutableList.of(
        "--output_gen_source_dir",
        j2ObjcGenJarTranslatedSourceFiles(ruleContext).getExecPathString(),
        "--output_gen_header_dir",
        j2objcGenJarTranslatedHeaderFiles(ruleContext).getExecPathString());
  }

  private void createJ2ObjcTranspilationAction(
      RuleContext ruleContext,
      NestedSet<Artifact> depsHeaderMappingFiles,
      NestedSet<Artifact> depsClassMappingFiles,
      Iterable<Artifact> sources,
      JavaCompilationArgsProvider compArgsProvider,
      J2ObjcSource j2ObjcSource,
      Optional<Artifact> genSrcJar) {
    CustomCommandLine.Builder argBuilder = CustomCommandLine.builder();
    PathFragment javaExecutable = ruleContext.getFragment(Jvm.class, HOST).getJavaExecutable();
    argBuilder.add("--java").add(javaExecutable.getPathString());

    Artifact j2ObjcDeployJar = ruleContext.getPrerequisiteArtifact("$j2objc", Mode.HOST);
    argBuilder.addExecPath("--j2objc", j2ObjcDeployJar);

    argBuilder.add("--main_class").add("com.google.devtools.j2objc.J2ObjC");
    argBuilder.addJoinExecPaths(
        "--translated_source_files",
        ",",
        Iterables.filter(j2ObjcSource.getObjcSrcs(), Predicates.not(IS_TREE_ARTIFACT)));
    argBuilder.add("--objc_file_path").addPath(j2ObjcSource.getObjcFilePath());

    Artifact outputDependencyMappingFile = j2ObjcOutputDependencyMappingFile(ruleContext);
    argBuilder.addExecPath("--output_dependency_mapping_file", outputDependencyMappingFile);

    ImmutableList.Builder<Artifact> genSrcOutputFiles = ImmutableList.builder();
    if (genSrcJar.isPresent()) {
      genSrcOutputFiles.addAll(genJarOutputs(ruleContext));
      argBuilder.addExecPath("--gen_src_jar", genSrcJar.get());
      argBuilder.add(genJarFlags(ruleContext));
    }

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

    Artifact archiveSourceMappingFile = j2ObjcOutputArchiveSourceMappingFile(ruleContext);
    argBuilder.addExecPath("--output_archive_source_mapping_file", archiveSourceMappingFile);

    Artifact compiledLibrary = ObjcRuleClasses.j2objcIntermediateArtifacts(ruleContext).archive();
    argBuilder.addExecPath("--compiled_archive_file_path", compiledLibrary);

    Artifact bootclasspathJar = ruleContext.getPrerequisiteArtifact("$jre_emul_jar", Mode.HOST);
    argBuilder.add("-Xbootclasspath:" + bootclasspathJar.getExecPathString());

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
        ParameterFile.ParameterFileType.UNQUOTED,
        ISO_8859_1));

    SpawnAction.Builder builder = new SpawnAction.Builder()
        .setMnemonic("TranspilingJ2objc")
        .setExecutable(ruleContext.getPrerequisiteArtifact("$j2objc_wrapper", Mode.HOST))
        .addInput(ruleContext.getPrerequisiteArtifact("$j2objc_wrapper", Mode.HOST))
        .addInput(j2ObjcDeployJar)
        .addInput(bootclasspathJar)
        .addInputs(sources)
        .addInputs(genSrcJar.asSet())
        .addTransitiveInputs(compileTimeJars)
        .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext))
        .addTransitiveInputs(depsHeaderMappingFiles)
        .addTransitiveInputs(depsClassMappingFiles)
        .addInput(paramFile)
        .setCommandLine(CustomCommandLine.builder()
            .addPaths("@%s", paramFile.getExecPath())
            .build())
        .addOutputs(Iterables.filter(j2ObjcSource.getObjcSrcs(), Predicates.not(IS_TREE_ARTIFACT)))
        .addOutputs(Iterables.filter(j2ObjcSource.getObjcHdrs(), Predicates.not(IS_TREE_ARTIFACT)))
        .addOutputs(genSrcOutputFiles.build())
        .addOutput(outputHeaderMappingFile)
        .addOutput(outputDependencyMappingFile)
        .addOutput(archiveSourceMappingFile);

    ruleContext.registerAction(builder.build(ruleContext));
  }

  private J2ObjcMappingFileProvider depJ2ObjcMappingFileProvider(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> depsHeaderMappingsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> depsClassMappingsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> depsDependencyMappingsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> depsArchiveSourceMappingsBuilder = NestedSetBuilder.stableOrder();

    for (J2ObjcMappingFileProvider mapping : getJ2ObjCMappings(ruleContext)) {
      depsHeaderMappingsBuilder.addTransitive(mapping.getHeaderMappingFiles());
      depsClassMappingsBuilder.addTransitive(mapping.getClassMappingFiles());
      depsDependencyMappingsBuilder.addTransitive(mapping.getDependencyMappingFiles());
      depsArchiveSourceMappingsBuilder.addTransitive(mapping.getArchiveSourceMappingFiles());
    }

    return new J2ObjcMappingFileProvider(
        depsHeaderMappingsBuilder.build(),
        depsClassMappingsBuilder.build(),
        depsDependencyMappingsBuilder.build(),
        depsArchiveSourceMappingsBuilder.build());
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
    if (context.attributes().has(attributeName, BuildType.LABEL_LIST)) {
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

  private static Artifact j2ObjcOutputArchiveSourceMappingFile(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(
        ruleContext, ".archive_source_mapping.j2objc");
  }

  private static Artifact j2ObjcGenJarTranslatedSourceFiles(RuleContext ruleContext) {
    PathFragment rootRelativePath = ruleContext
        .getUniqueDirectory("_j2objc/gen_jar_files")
        .getRelative("source_files");
    return ruleContext.getTreeArtifact(rootRelativePath, ruleContext.getBinOrGenfilesDirectory());
  }

  private static Artifact j2objcGenJarTranslatedHeaderFiles(RuleContext ruleContext) {
    PathFragment rootRelativePath = ruleContext
        .getUniqueDirectory("_j2objc/gen_jar_files")
        .getRelative("header_files");
    return ruleContext.getTreeArtifact(rootRelativePath, ruleContext.getBinOrGenfilesDirectory());
  }

  private static Artifact j2ObjcGenJarSourceZip(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".genjar_source.zip");
  }

  private static Artifact j2ObjcGenJarSourceZipManifest(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".genjar_source.txt");
  }

  private static Artifact j2ObjcGenJarHeaderZip(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".genjar_header.zip");
  }

  private static Artifact j2ObjcGenJarHeaderZipManifest(RuleContext ruleContext) {
    return ObjcRuleClasses.artifactByAppendingToBaseName(ruleContext, ".genjar_header.txt");
  }

  private J2ObjcSource buildJ2ObjcSource(RuleContext ruleContext,
      Iterable<Artifact> javaInputSourceFiles, Optional<Artifact> genSrcJar) {
    PathFragment objcFileRootRelativePath = ruleContext.getUniqueDirectory("_j2objc");
    PathFragment objcFileRootExecPath = ruleContext
        .getConfiguration()
        .getBinFragment()
        .getRelative(objcFileRootRelativePath);
    Iterable<Artifact> objcSrcs = getOutputObjcFiles(ruleContext, javaInputSourceFiles,
        objcFileRootRelativePath, ".m");
    Iterable<Artifact> objcHdrs = getOutputObjcFiles(ruleContext, javaInputSourceFiles,
        objcFileRootRelativePath, ".h");
    Iterable<PathFragment> headerSearchPaths = J2ObjcLibrary.j2objcSourceHeaderSearchPaths(
        ruleContext, objcFileRootExecPath, javaInputSourceFiles);

    Optional<Artifact> genJarTranslatedSrcs = Optional.absent();
    Optional<Artifact> genJarTranslatedHdrs = Optional.absent();
    Optional<PathFragment> genJarFileHeaderSearchPaths = Optional.absent();

    if (genSrcJar.isPresent()) {
      genJarTranslatedSrcs = Optional.of(j2ObjcGenJarTranslatedSourceFiles(ruleContext));
      genJarTranslatedHdrs = Optional.of(j2objcGenJarTranslatedHeaderFiles(ruleContext));
      genJarFileHeaderSearchPaths = Optional.of(genJarTranslatedHdrs.get().getExecPath());
    }

    return new J2ObjcSource(
        ruleContext.getRule().getLabel(),
        Iterables.concat(objcSrcs, genJarTranslatedSrcs.asSet()),
        Iterables.concat(objcHdrs, genJarTranslatedHdrs.asSet()),
        objcFileRootExecPath,
        SourceType.JAVA,
        Iterables.concat(headerSearchPaths, genJarFileHeaderSearchPaths.asSet()));
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

  /**
   * Sets up and returns an {@link ObjcCommon} object containing the J2ObjC-translated code.
   *
   */
  static ObjcCommon common(RuleContext ruleContext, Iterable<Artifact> transpiledSources,
      Iterable<Artifact> transpiledHeaders, Iterable<PathFragment> headerSearchPaths,
      Iterable<Attribute> dependentAttributes) {
    ObjcCommon.Builder builder = new ObjcCommon.Builder(ruleContext);
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.j2objcIntermediateArtifacts(ruleContext);

    if (!Iterables.isEmpty(transpiledSources) || !Iterables.isEmpty(transpiledHeaders)) {
      CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
          .addNonArcSrcs(transpiledSources)
          .setIntermediateArtifacts(intermediateArtifacts)
          .setPchFile(Optional.<Artifact>absent())
          .addAdditionalHdrs(transpiledHeaders)
          .build();
      builder.setCompilationArtifacts(compilationArtifacts);
      builder.setHasModuleMap();
    }

    for (Attribute dependentAttribute : dependentAttributes) {
      if (ruleContext.getAttribute(dependentAttribute.getName()) != null) {
        builder.addDepObjcProviders(ruleContext.getPrerequisites(
            dependentAttribute.getName(),
            dependentAttribute.getAccessMode(),
            ObjcProvider.class));
      }
    }

    return builder
        .addUserHeaderSearchPaths(headerSearchPaths)
        .setIntermediateArtifacts(intermediateArtifacts)
        .build();
  }

  /**
   * Sets up and returns an {@link XcodeProvider} object containing the J2ObjC-translated code.
   *
   */
  static XcodeProvider xcodeProvider(RuleContext ruleContext, ObjcCommon common,
      Iterable<Artifact> transpiledHeaders, Iterable<PathFragment> headerSearchPaths,
      Iterable<Attribute> dependentAttributes) {
    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    XcodeSupport xcodeSupport = new XcodeSupport(ruleContext);
    xcodeSupport.addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC);

    for (Attribute dependentAttribute : dependentAttributes) {
      if (ruleContext.getAttribute(dependentAttribute.getName()) != null) {
        xcodeSupport.addDependencies(xcodeProviderBuilder, dependentAttribute);
      }
    }

    if (!Iterables.isEmpty(transpiledHeaders)) {
      xcodeProviderBuilder
          .addUserHeaderSearchPaths(headerSearchPaths)
          .addCopts(ruleContext.getFragment(ObjcConfiguration.class).getCopts())
          .addHeaders(transpiledHeaders);
    }

    if (common.getCompilationArtifacts().isPresent()) {
      xcodeProviderBuilder.setCompilationArtifacts(common.getCompilationArtifacts().get());
    }

    return xcodeProviderBuilder.build();
  }
}
