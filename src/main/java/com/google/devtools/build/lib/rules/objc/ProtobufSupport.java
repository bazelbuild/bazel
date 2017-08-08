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
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;

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
  private final Iterable<ObjcProvider> dependencyObjcProviders;
  private final NestedSet<Artifact> portableProtoFilters;

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
   * @param protoProviders the list of ProtoSourcesProviders that this proto support should process
   * @param objcProtoProviders the list of ObjcProtoProviders that this proto support should process
   * @param dependencyObjcProviders the list of extra objc dependencies to add to this support's
   *     ObjcProvider
   */
  public ProtobufSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      Iterable<ProtoSourcesProvider> protoProviders,
      Iterable<ObjcProtoProvider> objcProtoProviders,
      Iterable<ObjcProvider> dependencyObjcProviders,
      NestedSet<Artifact> portableProtoFilters) {
    this(
        ruleContext,
        buildConfiguration,
        NestedSetBuilder.<Artifact>stableOrder().build(),
        protoProviders,
        objcProtoProviders,
        dependencyObjcProviders,
        portableProtoFilters);
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
   * @param protoProviders the list of ProtoSourcesProviders that this proto support should process
   * @param objcProtoProviders the list of ObjcProtoProviders that this proto support should process
   * @param dependencyObjcProviders the list of extra objc dependencies to add to this support's
   *     ObjcProvider
   */
  public ProtobufSupport(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      NestedSet<Artifact> dylibHandledProtos,
      Iterable<ProtoSourcesProvider> protoProviders,
      Iterable<ObjcProtoProvider> objcProtoProviders,
      Iterable<ObjcProvider> dependencyObjcProviders,
      NestedSet<Artifact> portableProtoFilters) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.attributes = new ProtoAttributes(ruleContext);
    this.dylibHandledProtoPaths = runfilesPaths(dylibHandledProtos.toSet());
    this.objcProtoProviders = objcProtoProviders;
    this.dependencyObjcProviders = dependencyObjcProviders;
    this.portableProtoFilters = portableProtoFilters;
    this.intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext, buildConfiguration);
    this.inputsToOutputsMap = getInputsToOutputsMap(attributes, protoProviders, objcProtoProviders);
  }

  /**
   * Registers the proto generation actions. These actions generate the ObjC/CPP code to be compiled
   * by this rule.
   */
  public ProtobufSupport registerGenerationActions() {
    int actionId = 0;

    for (ImmutableSet<Artifact> inputProtos : inputsToOutputsMap.keySet()) {
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
    int actionId = 0;
    Iterable<PathFragment> userHeaderSearchPaths =
        ImmutableList.of(getWorkspaceRelativeOutputDir());
    for (ImmutableSet<Artifact> inputProtos : inputsToOutputsMap.keySet()) {
      ImmutableSet<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);

      IntermediateArtifacts intermediateArtifacts = getUniqueIntermediateArtifacts(actionId);

      CompilationArtifacts compilationArtifacts =
          getCompilationArtifacts(intermediateArtifacts, inputProtos, outputProtos);

      ObjcCommon common = getCommon(intermediateArtifacts, compilationArtifacts);

      new LegacyCompilationSupport(
              ruleContext,
              buildConfiguration,
              intermediateArtifacts,
              new CompilationAttributes.Builder().build(),
              /*useDeps=*/ false,
              new TreeMap<String, NestedSet<Artifact>>(),
              /*isTestRule=*/ false,
              /*usePch=*/ false)
          .registerCompileAndArchiveActions(common, userHeaderSearchPaths);

      actionId++;
    }
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

    int actionId = 0;
    for (ImmutableSet<Artifact> inputProtos : inputsToOutputsMap.keySet()) {
      ImmutableSet<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);
      IntermediateArtifacts intermediateArtifacts = getUniqueIntermediateArtifacts(actionId);

      CompilationArtifacts compilationArtifacts =
          getCompilationArtifacts(intermediateArtifacts, inputProtos, outputProtos);

      ObjcCommon common = getCommon(intermediateArtifacts, compilationArtifacts);
      filesToBuild.addAll(common.getCompiledArchive().asSet());
      actionId++;
    }

    return this;
  }

  /**
   * Returns the ObjcProvider for this target, or Optional.absent() if there were no protos to
   * generate.
   */
  public Optional<ObjcProvider> getObjcProvider() {
    if (inputsToOutputsMap.isEmpty()) {
      return Optional.absent();
    }

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();

    int actionId = 0;
    for (ImmutableSet<Artifact> inputProtos : inputsToOutputsMap.keySet()) {
      ImmutableSet<Artifact> outputProtos = inputsToOutputsMap.get(inputProtos);
      IntermediateArtifacts intermediateArtifacts = getUniqueIntermediateArtifacts(actionId);

      CompilationArtifacts compilationArtifacts =
          getCompilationArtifacts(intermediateArtifacts, inputProtos, outputProtos);

      ObjcCommon common = getCommon(intermediateArtifacts, compilationArtifacts);
      if (isLinkingTarget()) {
        // If ProtobufSupport is being called from a linking target (i.e. apple_binary), we
        // propagate the generated headers (contained inside common.getObjcProvider()) into the
        // ObjcProvider, in case the generated headers are used from dependent ios_test targets,
        // which uses XcTestAppProvider to access those headers.
        // TODO(b/64152968): Remove conditional when ios_test is gone and only propagate to
        //     direct dependents.
        objcProviderBuilder.addTransitiveAndPropagate(common.getObjcProvider());
      } else {
        // If ProtobufSupport is being called from a non-linking target (i.e. objc_proto_library),
        // we propagate the headers only to direct dependents of this target. This avoids over
        // accumulation of header files when a top-level target depends on many objc_proto_library
        // targets.
        objcProviderBuilder.addAsDirectDeps(common.getObjcProvider());
      }
      actionId++;
    }

    // Same as with the headers, we only propagate the includes to direct dependents for non-linking
    // targets (i.e. objc_proto_library), and we fully propagate the includes for linking targets
    // in case ios_test dependents need to use the headers.
    // TODO(b/64152968): Remove conditional when ios_test is gone and only propagate to direct
    //     dependents.
    Iterable<PathFragment> includes = ImmutableList.of(getWorkspaceRelativeOutputDir());
    if (isLinkingTarget()) {
      objcProviderBuilder.addAll(ObjcProvider.INCLUDE, includes);
    } else {
      objcProviderBuilder.addAllForDirectDependents(ObjcProvider.INCLUDE, includes);
      // For non linking targets, also create and propagate this target's module map.
      objcProviderBuilder.addModuleMap(intermediateArtifacts.moduleMap());
    }

    objcProviderBuilder.addTransitiveAndPropagate(dependencyObjcProviders);

    return Optional.of(objcProviderBuilder.build());
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
      Iterable<ProtoSourcesProvider> protoProviders,
      Iterable<ObjcProtoProvider> objcProtoProviders) {
    ImmutableList.Builder<NestedSet<Artifact>> protoSets =
        new ImmutableList.Builder<NestedSet<Artifact>>();

    // Traverse all the dependencies ObjcProtoProviders and ProtoSourcesProviders to aggregate
    // all the transitive groups of proto.
    for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
      protoSets.addAll(objcProtoProvider.getProtoGroups());
    }
    for (ProtoSourcesProvider protoProvider : protoProviders) {
      protoSets.add(protoProvider.getTransitiveProtoSources());
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

  private String getBundledProtosSuffix() {
    return "_" + BUNDLED_PROTOS_IDENTIFIER;
  }

  private String getUniqueBundledProtosPrefix(int actionId) {
    return BUNDLED_PROTOS_IDENTIFIER + "_" + actionId;
  }

  private String getUniqueBundledProtosSuffix(int actionId) {
    return getBundledProtosSuffix() + "_" + actionId;
  }

  private IntermediateArtifacts getUniqueIntermediateArtifacts(int actionId) {
    return new IntermediateArtifacts(
        ruleContext,
        getUniqueBundledProtosSuffix(actionId),
        getUniqueBundledProtosPrefix(actionId),
        buildConfiguration);
  }

  private ObjcCommon getCommon(
      IntermediateArtifacts intermediateArtifacts, CompilationArtifacts compilationArtifacts) {
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

  private CompilationArtifacts getCompilationArtifacts(
      IntermediateArtifacts intermediateArtifacts,
      Iterable<Artifact> inputProtoFiles,
      Iterable<Artifact> outputProtoFiles) {
    // Filter the well known protos from the set of headers. We don't generate the headers for them
    // as they are part of the runtime library.
    Iterable<Artifact> filteredInputProtos = attributes.filterWellKnownProtos(inputProtoFiles);

    CompilationArtifacts.Builder compilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(intermediateArtifacts)
            .addAdditionalHdrs(getGeneratedProtoOutputs(filteredInputProtos, HEADER_SUFFIX))
            .addAdditionalHdrs(getProtobufHeaders());

    if (isLinkingTarget()) {
      compilationArtifacts.addNonArcSrcs(getProtoSourceFilesForCompilation(outputProtoFiles));
    }

    return compilationArtifacts.build();
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
            .addOutputs(getProtoSourceFilesForCompilation(outputProtos))
            .setExecutable(attributes.getProtoCompiler().getExecPath())
            .setCommandLine(getGenerationCommandLine(protoInputsFile))
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
        .add(protoInputsFile.getExecPathString())
        .add("--output-dir")
        .add(getWorkspaceRelativeOutputDir().getSafePathString())
        .add("--force")
        .add("--proto-root-dir")
        .add(getGenfilesPathString())
        .add("--proto-root-dir")
        .add(".")
        .addBeforeEachExecPath("--config", portableProtoFilters)
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

    return PathFragment.create(
        buildConfiguration.getBinDirectory().getExecPath(), rootRelativeOutputDir);
  }

  private Iterable<Artifact> getGeneratedProtoOutputs(
      Iterable<Artifact> outputProtos, String extension) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact protoFile : outputProtos) {
      String protoFileName = FileSystemUtils.removeExtension(protoFile.getFilename());
      String generatedOutputName = attributes.getGeneratedProtoFilename(protoFileName, true);

      PathFragment generatedFilePath = PathFragment.create(
          protoFile.getRootRelativePath().getParentDirectory(),
          PathFragment.create(generatedOutputName));

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
    // Since this is the ProtobufSupport helper class, check whether the current target has
    // configured the protobuf attributes. If not, it's not an objc_proto_library rule, so it must
    // be a linking rule (e.g. apple_binary).
    return !attributes.requiresProtobuf();
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
   * Registers a FileWriteAction what writes a filter file into the given artifact. The contents
   * of this file is a portable filter that allows all the transitive proto files contained in the
   * given {@link ProtoSourcesProvider} providers.
   */
  public static void registerPortableFilterGenerationAction(
      RuleContext ruleContext,
      Artifact generatedPortableFilter,
      Iterable<ProtoSourcesProvider> protoProviders) {
    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext,
            generatedPortableFilter,
            getGeneratedPortableFilterContents(ruleContext, protoProviders),
            false));
  }

  private static String getGeneratedPortableFilterContents(
      RuleContext ruleContext, Iterable<ProtoSourcesProvider> protoProviders) {
    NestedSetBuilder<Artifact> protoFilesBuilder = NestedSetBuilder.stableOrder();
    for (ProtoSourcesProvider protoProvider : protoProviders) {
      protoFilesBuilder.addTransitive(protoProvider.getTransitiveProtoSources());
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
