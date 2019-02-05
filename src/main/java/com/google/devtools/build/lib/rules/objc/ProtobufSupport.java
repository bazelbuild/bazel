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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
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

  private static final String UNIQUE_DIRECTORY_NAME = "_generated_protos";

  private final RuleContext ruleContext;
  private final BuildConfiguration buildConfiguration;
  private final ProtoAttributes attributes;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Set<PathFragment> dylibHandledProtoPaths;
  private final Iterable<ObjcProtoProvider> objcProtoProviders;
  private final NestedSet<Artifact> portableProtoFilters;
  private final CcToolchainProvider toolchain;

  // Each entry of this map represents a generation action and a compilation action. The input set
  // are dependencies of the output set. The output set is always a subset of, or the same set as,
  // the input set. For example, given a sample entry of the inputsToOutputsMap like:
  //
  //    {A, B, C} => {B, C}
  //
  // this represents:
  // 1. A generation action in which the inputs are A, B and C, and the outputs are B.pbobjc.h,
  //    B.pbobjc.m, C.pbobjc.h and C.pbobjc.m.
  // 2. A compilation action in which the inputs are A.pbobjc.h, B.pbobjc.h, C.pbobjc.h,
  //    B.pbobjc.m and C.pbobjc.m, while the outputs are B.pbobjc.o and C.pbobjc.o.
  //
  // Given that each input set appears only once, by the nature of the structure, we can safely use
  // it as an identifier of the entry.
  private final ImmutableSetMultimap<ImmutableSet<Artifact>, Artifact> inputsToOutputsMap;

  /**
   * Creates a new proto support for the protobuf library. This support code bundles up all the
   * transitive protos within the groups in which they were defined. We use that information to
   * minimize the number of inputs per generation/compilation actions by only providing what is
   * really needed to the actions.
   *
   * @param ruleContext context this proto library is constructed in
   * @param buildConfiguration the configuration from which to get prerequisites when building proto
   *     targets in a split configuration
   * @param protoInfos the list of ProtoInfos that this proto support should process
   * @param objcProtoProviders the list of ObjcProtoProviders that this proto support should process
   */
  public ProtobufSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      Iterable<ProtoInfo> protoInfos,
      Iterable<ObjcProtoProvider> objcProtoProviders,
      NestedSet<Artifact> portableProtoFilters) {
    this(
        ruleContext,
        buildConfiguration,
        NestedSetBuilder.<Artifact>stableOrder().build(),
        protoInfos,
        objcProtoProviders,
        portableProtoFilters,
        null);
  }

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
   * @param protoInfos the list of ProtoInfos that this proto support should process
   * @param objcProtoProviders the list of ObjcProtoProviders that this proto support should process
   * @param toolchain if not null, the toolchain to override the default toolchain for the rule
   *     context.
   */
  public ProtobufSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      NestedSet<Artifact> dylibHandledProtos,
      Iterable<ProtoInfo> protoInfos,
      Iterable<ObjcProtoProvider> objcProtoProviders,
      NestedSet<Artifact> portableProtoFilters,
      CcToolchainProvider toolchain) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.attributes = new ProtoAttributes(ruleContext);
    this.dylibHandledProtoPaths = runfilesPaths(dylibHandledProtos.toSet());
    this.objcProtoProviders = objcProtoProviders;
    this.portableProtoFilters = portableProtoFilters;
    this.intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext, buildConfiguration);
    this.inputsToOutputsMap = getInputsToOutputsMap(attributes, protoInfos, objcProtoProviders);
    this.toolchain = toolchain;
  }

  /**
   * Registers the proto generation actions. These actions generate the ObjC/CPP code to be compiled
   * by this rule.
   */
  public ProtobufSupport registerGenerationActions() {
    int actionId = 0;

    for (ImmutableSet<Artifact> inputProtos : orderedInputOutputKeySet()) {
      Iterable<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);
      registerGenerationAction(outputProtos, inputProtos, getUniqueBundledProtosSuffix(actionId));
      actionId++;
    }

    if (!isLinkingTarget()) {
      registerModuleMapGenerationAction();
    }

    return this;
  }

  private void registerModuleMapGenerationAction() {
    CompilationArtifacts.Builder moduleMapCompilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(intermediateArtifacts)
            .addAdditionalHdrs(getProtobufHeaders())
            .addAdditionalHdrs(
                getGeneratedProtoOutputs(inputsToOutputsMap.values(), HEADER_SUFFIX));

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder()
            .setRuleContext(ruleContext)
            .setCompilationAttributes(new CompilationAttributes.Builder().build())
            .doNotUsePch()
            .build();

    compilationSupport.registerGenerateModuleMapAction(moduleMapCompilationArtifacts.build());
  }

  /**
   * Registers the actions that will compile the generated code.
   */
  public ProtobufSupport registerCompilationActions()
      throws RuleErrorException, InterruptedException {
    Iterable<PathFragment> userHeaderSearchPaths =
        ImmutableList.of(getWorkspaceRelativeOutputDir());

    CompilationArtifacts.Builder compilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(getUniqueIntermediateArtifactsForSourceCompile());

    for (ImmutableSet<Artifact> inputProtos : orderedInputOutputKeySet()) {
      ImmutableSet<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);
      addCompilationArtifacts(compilationArtifacts, inputProtos, outputProtos);
    }

    ObjcCommon common =
        getCommon(getUniqueIntermediateArtifactsForSourceCompile(), compilationArtifacts.build());

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

    return this;
  }

  /** Adds the generated files to the set of files to be output when this rule is built. */
  public ProtobufSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild) {
    for (ImmutableSet<Artifact> inputProtoFiles : inputsToOutputsMap.keySet()) {
      ImmutableSet<Artifact> outputProtoFiles = inputsToOutputsMap.get(inputProtoFiles);
      Iterable<Artifact> generatedSources = getProtoSourceFilesForCompilation(outputProtoFiles);
      Iterable<Artifact> generatedHeaders = getGeneratedProtoOutputs(outputProtoFiles,
          HEADER_SUFFIX);

      filesToBuild.addAll(generatedSources).addAll(generatedHeaders);
    }
    return this;
  }

  /**
   * Returns the ObjcProvider for this target, or Optional.absent() if there were no protos to
   * generate.
   */
  public Optional<ObjcProvider> getObjcProvider() throws InterruptedException {
    if (inputsToOutputsMap.isEmpty()) {
      return Optional.absent();
    }

    Iterable<PathFragment> includes = ImmutableList.of(getWorkspaceRelativeOutputDir());
    ObjcCommon.Builder commonBuilder = new ObjcCommon.Builder(ruleContext);

    if (!isLinkingTarget()) {
      commonBuilder.setIntermediateArtifacts(intermediateArtifacts).setHasModuleMap();
    }

    CompilationArtifacts.Builder compilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(getUniqueIntermediateArtifactsForSourceCompile());

    for (ImmutableSet<Artifact> inputProtos : orderedInputOutputKeySet()) {
      ImmutableSet<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);
      addCompilationArtifacts(compilationArtifacts, inputProtos, outputProtos);
    }

    ObjcCommon common =
        getCommon(getUniqueIntermediateArtifactsForSourceCompile(), compilationArtifacts.build());
    commonBuilder.addDepObjcProviders(ImmutableSet.of(common.getObjcProvider()));

    if (isLinkingTarget()) {
      commonBuilder.addIncludes(includes);
    } else {
      commonBuilder.addDirectDependencyIncludes(includes);
    }

    return Optional.of(commonBuilder.build().getObjcProvider());
  }

  private NestedSet<Artifact> getProtobufHeaders() {
    NestedSetBuilder<Artifact> protobufHeaders = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protobufHeaders.addTransitive(objcProtoProvider.getProtobufHeaders());
    }
    return protobufHeaders.build();
  }

  private NestedSet<PathFragment> getProtobufHeaderSearchPaths() {
    NestedSetBuilder<PathFragment> protobufHeaderSearchPaths = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protobufHeaderSearchPaths.addTransitive(objcProtoProvider.getProtobufHeaderSearchPaths());
    }
    return protobufHeaderSearchPaths.build();
  }

  private static Set<PathFragment> runfilesPaths(Set<Artifact> artifacts) {
    HashSet<PathFragment> pathsSet = new HashSet<>();
    for (Artifact artifact : artifacts) {
      pathsSet.add(artifact.getRunfilesPath());
    }
    return pathsSet;
  }

  private static ImmutableSetMultimap<ImmutableSet<Artifact>, Artifact> getInputsToOutputsMap(
      ProtoAttributes attributes,
      Iterable<ProtoInfo> protoInfos,
      Iterable<ObjcProtoProvider> objcProtoProviders) {
    ImmutableList.Builder<NestedSet<Artifact>> protoSets =
        new ImmutableList.Builder<NestedSet<Artifact>>();

    // Traverse all the dependencies ObjcProtoProviders and ProtoInfos to aggregate
    // all the transitive groups of proto.
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protoSets.addAll(objcProtoProvider.getProtoGroups());
    }
    for (ProtoInfo protoInfo : protoInfos) {
      protoSets.add(protoInfo.getTransitiveProtoSources());
    }

    HashMap<Artifact, Set<Artifact>> protoToGroupMap = new HashMap<>();

    // For each proto in each proto group, store the smallest group in which it is contained. This
    // group will be considered the smallest input group with which the proto can be generated.
    for (NestedSet<Artifact> nestedProtoSet : protoSets.build()) {
      ImmutableSet<Artifact> protoSet = ImmutableSet.copyOf(nestedProtoSet.toSet());
      for (Artifact proto : protoSet) {
        // If the proto is well known, don't store it as we don't need to generate it; it comes
        // generated with the runtime library.
        if (attributes.isProtoWellKnown(proto)) {
          continue;
        }
        if (!protoToGroupMap.containsKey(proto)) {
          protoToGroupMap.put(proto, protoSet);
        } else {
          protoToGroupMap.put(proto, Sets.intersection(protoSet, protoToGroupMap.get(proto)));
        }
      }
    }

    // Now that we have the smallest proto inputs groups for each proto to be generated, we reverse
    // that map into a multimap to take advantage of the fact that multiple protos can be generated
    // with the same inputs, to avoid having multiple generation actions with the same inputs and
    // different ouputs. This only applies for the generation actions, as the compilation actions
    // compile one generated file at a time.
    // It's OK to use ImmutableSet<Artifact> as the key, since Artifact caches it's hashCode, and
    // ImmutableSet calculates it's hashCode in O(n).
    ImmutableSetMultimap.Builder<ImmutableSet<Artifact>, Artifact> inputsToOutputsMapBuilder =
        ImmutableSetMultimap.builder();

    for (Artifact proto : protoToGroupMap.keySet()) {
      inputsToOutputsMapBuilder.put(ImmutableSet.copyOf(protoToGroupMap.get(proto)), proto);
    }
    return inputsToOutputsMapBuilder.build();
  }

  /**
   * Returns an ordered list of ImmutableSets<Artifact>s representing the keys to the inputs-outputs
   * map. Using an ordered list ensures that for the same inputs, the keys are processed in the same
   * order, and avoids non-determinism in the intermediate outputs.
   */
  private List<ImmutableSet<Artifact>> orderedInputOutputKeySet() {
    return new Ordering<ImmutableSet<Artifact>>() {
      @Override
      public int compare(ImmutableSet<Artifact> o1, ImmutableSet<Artifact> o2) {
        return Integer.compare(o1.hashCode(), o2.hashCode());
      }
    }.sortedCopy(inputsToOutputsMap.keySet());
  }

  private String getBundledProtosSuffix() {
    return "_" + BUNDLED_PROTOS_IDENTIFIER;
  }

  private String getBundledProtosPrefix() {
    return BUNDLED_PROTOS_IDENTIFIER + "_";
  }

  private String getUniqueBundledProtosSuffix(int actionId) {
    return getBundledProtosSuffix() + "_" + actionId;
  }

  private IntermediateArtifacts getUniqueIntermediateArtifactsForSourceCompile() {
    return new IntermediateArtifacts(
        ruleContext, getBundledProtosSuffix(), getBundledProtosPrefix(), buildConfiguration);
  }

  private ObjcCommon getCommon(
      IntermediateArtifacts intermediateArtifacts, CompilationArtifacts compilationArtifacts)
      throws InterruptedException {
    ObjcCommon.Builder commonBuilder =
        new ObjcCommon.Builder(ruleContext)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setCompilationArtifacts(compilationArtifacts);
    if (isLinkingTarget()) {
      commonBuilder.addIncludes(getProtobufHeaderSearchPaths());
    } else {
      commonBuilder.addDepObjcProviders(
          ruleContext.getPrerequisites(
              ObjcRuleClasses.PROTO_LIB_ATTR, Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR));
    }
    return commonBuilder.build();
  }

  private void addCompilationArtifacts(
      CompilationArtifacts.Builder compilationArtifactsBuilder,
      Iterable<Artifact> inputProtoFiles,
      Iterable<Artifact> outputProtoFiles) {
    // Filter the well known protos from the set of headers. We don't generate the headers for them
    // as they are part of the runtime library.
    Iterable<Artifact> filteredInputProtos = attributes.filterWellKnownProtos(inputProtoFiles);

    compilationArtifactsBuilder
        .addAdditionalHdrs(getGeneratedProtoOutputs(filteredInputProtos, HEADER_SUFFIX))
        .addAdditionalHdrs(getProtobufHeaders());

    if (isLinkingTarget()) {
      compilationArtifactsBuilder.addNonArcSrcs(
          getProtoSourceFilesForCompilation(outputProtoFiles));
    }
  }

  private Iterable<Artifact> getProtoSourceFilesForCompilation(
      Iterable<Artifact> outputProtoFiles) {
    Predicate<Artifact> notDylibHandled =
        artifact -> !dylibHandledProtoPaths.contains(artifact.getRunfilesPath());
    Iterable<Artifact> filteredOutputs =
        Iterables.filter(outputProtoFiles, notDylibHandled);
    return getGeneratedProtoOutputs(filteredOutputs, SOURCE_SUFFIX);
  }

  private void registerGenerationAction(
      Iterable<Artifact> outputProtos, Iterable<Artifact> inputProtos, String protoFileSuffix) {
    Artifact protoInputsFile = getProtoInputsFile(protoFileSuffix);

    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext, protoInputsFile, getProtoInputsFileContents(outputProtos), false));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setMnemonic("GenObjcBundledProtos")
            .addInput(attributes.getProtoCompiler())
            .addInputs(attributes.getProtoCompilerSupport())
            .addTransitiveInputs(portableProtoFilters)
            .addInput(protoInputsFile)
            .addInputs(inputProtos)
            .addOutputs(getGeneratedProtoOutputs(outputProtos, HEADER_SUFFIX))
            // We register all proto generated sources as output, even though we only compile a
            // subset of them with getProtoSourceFilesForCompilation(), as we want blaze to track
            // all generated files in the action, and avoid "Permission Denied" errors when the
            // local file output cache is populated from a previous build.
            .addOutputs(getGeneratedProtoOutputs(outputProtos, SOURCE_SUFFIX))
            .setExecutable(attributes.getProtoCompiler().getExecPath())
            .addCommandLine(getGenerationCommandLine(protoInputsFile))
            .build(ruleContext));
  }

  private Artifact getProtoInputsFile(String suffix) {
    return ruleContext.getUniqueDirectoryArtifact(
        "_protos",
        "_proto_input_files" + suffix,
        buildConfiguration.getGenfilesDirectory());
  }

  private String getProtoInputsFileContents(Iterable<Artifact> outputProtos) {
    // Sort the file names to make the remote action key independent of the precise deps structure.
    // compile_protos.py will sort the input list anyway.
    Iterable<Artifact> sorted = Ordering.natural().immutableSortedCopy(outputProtos);
    return Artifact.joinRootRelativePaths("\n", sorted);
  }

  private CustomCommandLine getGenerationCommandLine(Artifact protoInputsFile) {
    return new CustomCommandLine.Builder()
        .add("--input-file-list")
        .addExecPath(protoInputsFile)
        .add("--output-dir")
        .addDynamicString(getWorkspaceRelativeOutputDir().getSafePathString())
        .add("--force")
        .add("--proto-root-dir")
        .addDynamicString(getGenfilesPathString())
        .add("--proto-root-dir")
        .add(".")
        .addExecPaths(VectorArg.addBefore("--config").each(portableProtoFilters))
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

  private Iterable<Artifact> getGeneratedProtoOutputs(
      Iterable<Artifact> outputProtos, String extension) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact protoFile : outputProtos) {
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

  private boolean isLinkingTarget() {
    // Since this is the ProtobufSupport helper class, check whether the current target is
    // an objc_proto_library. If not, it must be a linking rule (e.g. apple_binary).
    return !attributes.isObjcProtoLibrary();
  }

  /**
   * Returns the transitive portable proto filter files from a list of ObjcProtoProviders.
   */
  public static NestedSet<Artifact> getTransitivePortableProtoFilters(
      Iterable<ObjcProtoProvider> objcProtoProviders) {
    NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      portableProtoFilters.addTransitive(objcProtoProvider.getPortableProtoFilters());
    }
    return portableProtoFilters.build();
  }

  /**
   * Returns a target specific generated artifact that represents a portable filter file.
   */
  public static Artifact getGeneratedPortableFilter(RuleContext ruleContext,
      BuildConfiguration buildConfiguration) {
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
  public static void registerPortableFilterGenerationAction(
      RuleContext ruleContext, Artifact generatedPortableFilter, Iterable<ProtoInfo> protoInfos) {
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
            Ordering.natural().immutableSortedCopy(protoFilesBuilder.build()));

    Iterable<String> filterLines =
        Iterables.transform(
            protoFilePaths, protoFilePath -> String.format("allowed_file: \"%s\"", protoFilePath));

    return String.format(
            "# Generated portable filter for %s\n\n", ruleContext.getLabel().getCanonicalForm())
        + Joiner.on("\n").join(filterLines);
  }

}
