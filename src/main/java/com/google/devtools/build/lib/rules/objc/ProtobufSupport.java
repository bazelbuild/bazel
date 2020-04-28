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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoSourceFileBlacklist;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Support for generating Objective C proto static libraries that registers actions which generate
 * and compile the Objective C protos by using the open source protobuf library and compiler.
 *
 * <p>Each group represents one proto_library target depended on by objc_proto_library targets using
 * the portable_proto_filters attribute. This group contains all the necessary protos to satisfy its
 * internal dependencies.
 *
 * <p>Grouping has a first pass in which for each proto required to be built, we find the smallest
 * group containing it, and store that information in a map. We then reverse that map into a multi
 * map, in which the keys are the input protos and the values are the output protos to be
 * generated/compiled with the input group as dependencies. This minimizes the number of inputs
 * required for each generation/compilation action and the probability of regeneration when one of
 * the proto files change, improving cache hits.
 */
final class ProtobufSupport {

  private static final String HEADER_SUFFIX = ".pbobjc.h";
  private static final String SOURCE_SUFFIX = ".pbobjc.m";

  private static final String BUNDLED_PROTOS_IDENTIFIER = "BundledProtos";

  private static final String UNIQUE_DIRECTORY_NAME = "_generated_objc_protos";

  private final RuleContext ruleContext;
  private final BuildConfiguration buildConfiguration;
  private final ProtoAttributes attributes;
  private final Collection<ObjcProtoProvider> objcProtoProviders;
  private final NestedSet<Artifact> portableProtoFilters;
  private final CcToolchainProvider toolchain;
  private final ImmutableSet<Artifact> dylibHandledProtos;
  private Optional<ObjcProvider> objcProvider;

  /**
   * Creates a new proto support for the protobuf library. This support code bundles up all the
   * transitive protos within the groups in which they were defined. We use that information to
   * minimize the number of inputs per generation/compilation actions by only providing what is
   * really needed to the actions.
   *
   * @param ruleContext context this proto library is constructed in
   * @param buildConfiguration the configuration from which to get prerequisites when building proto
   *     targets in a split configuration
   * @param dylibHandledProtos a set of protos linked into dynamic libraries that the current rule
   *     depends on; these protos will not be output by this support, thus avoiding duplicate
   *     symbols
   * @param objcProtoProviders the list of ObjcProtoProviders that this proto support should process
   * @param toolchain if not null, the toolchain to override the default toolchain for the rule
   *     context.
   */
  ProtobufSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      NestedSet<Artifact> dylibHandledProtos,
      Collection<ObjcProtoProvider> objcProtoProviders,
      NestedSet<Artifact> portableProtoFilters,
      CcToolchainProvider toolchain) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.attributes = new ProtoAttributes(ruleContext);
    this.objcProtoProviders = objcProtoProviders;
    this.portableProtoFilters = portableProtoFilters;
    this.toolchain = toolchain;
    this.dylibHandledProtos = dylibHandledProtos.toSet();
    this.objcProvider = Optional.absent();
  }

  /** Registers the action that will compile the generated code. */
  ProtobufSupport registerCompilationAction() throws RuleErrorException, InterruptedException {
    if (!hasOutputProtos()) {
      return this;
    }
    List<PathFragment> userHeaderSearchPaths = ImmutableList.of(getWorkspaceRelativeOutputDir());

    CompilationArtifacts compilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(getUniqueIntermediateArtifactsForSourceCompile())
            .addNonArcSrcs(getGeneratedProtoOutputs(getOutputProtos(), SOURCE_SUFFIX))
            .addAdditionalHdrs(getGeneratedProtoOutputs(getAllProtos(), HEADER_SUFFIX))
            .addAdditionalHdrs(getProtobufHeaders())
            .build();

    ObjcCommon common =
        getCommon(getUniqueIntermediateArtifactsForSourceCompile(), compilationArtifacts);

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder()
            .setRuleContext(ruleContext)
            .setConfig(buildConfiguration)
            .setIntermediateArtifacts(getUniqueIntermediateArtifactsForSourceCompile())
            .setCompilationAttributes(new CompilationAttributes.Builder().build())
            .setToolchainProvider(toolchain)
            .doNotUsePch()
            .build();

    compilationSupport.registerCompileAndArchiveActions(common, userHeaderSearchPaths);
    setObjcProvider(compilationSupport.getObjcProvider());

    return this;
  }

  private void setObjcProvider(ObjcProvider objcProvider) {
    checkState(!this.objcProvider.isPresent());
    this.objcProvider = Optional.of(objcProvider);
  }

  /**
   * Returns the ObjcProvider for this target, or Optional.absent() if there were no protos to
   * generate.
   */
  public Optional<ObjcProvider> getObjcProvider() {
    return objcProvider;
  }

  private NestedSet<Artifact> getProtobufHeaders() {
    NestedSetBuilder<Artifact> protobufHeaders = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protobufHeaders.addTransitive(objcProtoProvider.getProtobufHeaders());
    }
    return protobufHeaders.build();
  }

  private NestedSet<Artifact> getAllProtos() {
    NestedSetBuilder<Artifact> protosSet = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protosSet.addTransitive(objcProtoProvider.getProtoFiles());
    }
    return protosSet.build();
  }

  private Boolean hasOutputProtos() {
    Set<PathFragment> dylibHandledProtoPaths = runfilesPaths(dylibHandledProtos);
    return Iterables.any(
        getAllProtos().toSet(),
        artifact -> !dylibHandledProtoPaths.contains(artifact.getRootRelativePath()));
  }

  private Iterable<Artifact> getOutputProtos() {
    Set<PathFragment> dylibHandledProtoPaths = runfilesPaths(dylibHandledProtos);
    return Iterables.filter(
        getAllProtos().toSet(),
        artifact -> !dylibHandledProtoPaths.contains(artifact.getRootRelativePath()));
  }

  private NestedSet<PathFragment> getProtobufHeaderSearchPaths() {
    NestedSetBuilder<PathFragment> protobufHeaderSearchPaths = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protobufHeaderSearchPaths.addTransitive(objcProtoProvider.getProtobufHeaderSearchPaths());
    }
    return protobufHeaderSearchPaths.build();
  }

  private static Set<PathFragment> runfilesPaths(Iterable<Artifact> artifacts) {
    HashSet<PathFragment> pathsSet = new HashSet<>();
    for (Artifact artifact : artifacts) {
      pathsSet.add(artifact.getRootRelativePath());
    }
    return pathsSet;
  }

  private static String getBundledProtosSuffix() {
    return "_" + BUNDLED_PROTOS_IDENTIFIER;
  }

  private static String getBundledProtosPrefix() {
    return BUNDLED_PROTOS_IDENTIFIER + "_";
  }

  private IntermediateArtifacts getUniqueIntermediateArtifactsForSourceCompile() {
    return new IntermediateArtifacts(
        ruleContext, getBundledProtosSuffix(), getBundledProtosPrefix(), buildConfiguration);
  }

  private ObjcCommon getCommon(
      IntermediateArtifacts intermediateArtifacts, CompilationArtifacts compilationArtifacts)
      throws InterruptedException {
    return new ObjcCommon.Builder(ObjcCommon.Purpose.COMPILE_AND_LINK, ruleContext)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setCompilationArtifacts(compilationArtifacts)
        .addIncludes(getProtobufHeaderSearchPaths())
        .build();
  }

  ProtobufSupport registerGenerationAction() {
    if (!hasOutputProtos()) {
      return this;
    }

    Artifact outputGroupFile =
        ruleContext.getUniqueDirectoryArtifact(
            "_protos", "output_group", buildConfiguration.getGenfilesDirectory());

    Artifact skipGroupFile =
        ruleContext.getUniqueDirectoryArtifact(
            "_protos", "skip_group", buildConfiguration.getGenfilesDirectory());

    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext, skipGroupFile, getProtoInputsFileContents(dylibHandledProtos), false));
    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext, outputGroupFile, getProtoInputsFileContents(getAllProtos()), false));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("GenObjcBundledProtos")
            .addInput(attributes.getProtoCompiler())
            .addInputs(attributes.getProtoCompilerSupport())
            .addTransitiveInputs(portableProtoFilters)
            .addInput(outputGroupFile)
            .addInput(skipGroupFile)
            .addTransitiveInputs(getAllProtos())
            .addOutputs(getGeneratedProtoOutputs(getAllProtos(), HEADER_SUFFIX))
            .addOutputs(getGeneratedProtoOutputs(getOutputProtos(), SOURCE_SUFFIX))
            .setExecutable(attributes.getProtoCompiler().getExecPath())
            .addCommandLine(getGenerationCommandLine(outputGroupFile, skipGroupFile))
            .build(ruleContext));

    return this;
  }

  private static String getProtoInputsFileContents(NestedSet<Artifact> protoFiles) {
    return getProtoInputsFileContents(protoFiles.toList());
  }

  private static String getProtoInputsFileContents(Iterable<Artifact> protoFiles) {
    // Sort the file names to make the remote action key independent of the precise deps structure.
    // compile_protos.py will sort the input list anyway.
    Iterable<Artifact> sorted = Ordering.natural().immutableSortedCopy(protoFiles);
    return Artifact.joinRootRelativePaths("\n", sorted);
  }

  private CustomCommandLine getGenerationCommandLine(
      Artifact outputGroupFile, Artifact skipGroupFile) {
    return new CustomCommandLine.Builder()
        .add("--output-dir")
        .addDynamicString(getWorkspaceRelativeOutputDir().getSafePathString())
        .add("--proto-root-dir")
        .addDynamicString(getGenfilesPathString())
        .add("--proto-root-dir")
        .add(".")
        .add("--input-file-list")
        .addExecPath(outputGroupFile)
        .addExecPaths(VectorArg.addBefore("--config").each(portableProtoFilters))
        .add("--skip-groups-impls")
        .addExecPath(skipGroupFile)
        .build();
  }

  private String getGenfilesPathString() {
    return buildConfiguration.getGenfilesDirectory().getExecPathString();
  }

  private PathFragment getWorkspaceRelativeOutputDir() {
    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path
    // of dependers.
    PathFragment rootRelativeOutputDir = ruleContext.getUniqueDirectory(UNIQUE_DIRECTORY_NAME);

    return buildConfiguration.getBinDirectory().getExecPath().getRelative(rootRelativeOutputDir);
  }

  private List<Artifact> getGeneratedProtoOutputs(
      NestedSet<Artifact> protoFiles, String extension) {
    return getGeneratedProtoOutputs(protoFiles.toList(), extension);
  }

  private List<Artifact> getGeneratedProtoOutputs(Iterable<Artifact> protoFiles, String extension) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    ProtoSourceFileBlacklist wellKnownProtoBlacklist =
        new ProtoSourceFileBlacklist(ruleContext, attributes.getWellKnownTypeProtos());
    for (Artifact protoFile : protoFiles) {
      if (wellKnownProtoBlacklist.isBlacklisted(protoFile)) {
        continue;
      }
      String protoFileName = FileSystemUtils.removeExtension(protoFile.getFilename());
      String generatedOutputName = attributes.getGeneratedProtoFilename(protoFileName, true);

      PathFragment generatedFilePath =
          protoFile.getRootRelativePath().getParentDirectory().getRelative(generatedOutputName);

      PathFragment outputFile = FileSystemUtils.appendExtension(generatedFilePath, extension);

      if (outputFile != null) {
        builder.add(
            ruleContext.getUniqueDirectoryArtifact(
                UNIQUE_DIRECTORY_NAME, outputFile, buildConfiguration.getBinDirectory()));

      }
    }
    return builder.build();
  }

  /** Returns the transitive portable proto filter files from a list of ObjcProtoProviders. */
  static NestedSet<Artifact> getTransitivePortableProtoFilters(
      Iterable<ObjcProtoProvider> objcProtoProviders) {
    NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      portableProtoFilters.addTransitive(objcProtoProvider.getPortableProtoFilters());
    }
    return portableProtoFilters.build();
  }

  /** Returns a target specific generated artifact that represents a portable filter file. */
  static Artifact getGeneratedPortableFilter(
      RuleContext ruleContext, BuildConfiguration buildConfiguration) {
    return ruleContext.getUniqueDirectoryArtifact(
        "_proto_filters",
        "generated_filter_file.pbascii",
        buildConfiguration.getGenfilesDirectory());
  }

  /**
   * Registers a FileWriteAction what writes a filter file into the given artifact. The contents of
   * this file is a portable filter that allows all the transitive proto files contained in the
   * given {@link ProtoInfo} providers.
   */
  static void registerPortableFilterGenerationAction(
      RuleContext ruleContext, Artifact generatedPortableFilter, List<ProtoInfo> protoInfos) {
    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext,
            generatedPortableFilter,
            getGeneratedPortableFilterContents(ruleContext, protoInfos),
            false));
  }

  private static String getGeneratedPortableFilterContents(
      RuleContext ruleContext, Iterable<ProtoInfo> protoInfos) {
    NestedSetBuilder<Artifact> protoFilesBuilder = NestedSetBuilder.stableOrder();
    for (ProtoInfo protoInfo : protoInfos) {
      protoFilesBuilder.addTransitive(protoInfo.getTransitiveProtoSources());
    }

    Iterable<String> protoFilePaths =
        Artifact.toRootRelativePaths(
            Ordering.natural().immutableSortedCopy(protoFilesBuilder.build().toList()));

    Iterable<String> filterLines =
        Iterables.transform(
            protoFilePaths, protoFilePath -> String.format("allowed_file: \"%s\"", protoFilePath));

    return String.format(
            "# Generated portable filter for %s\n\n", ruleContext.getLabel().getCanonicalForm())
        + Joiner.on("\n").join(filterLines);
  }

}
