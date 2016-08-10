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

import static com.google.common.base.CaseFormat.LOWER_UNDERSCORE;
import static com.google.common.base.CaseFormat.UPPER_CAMEL;

import com.google.common.base.CharMatcher;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;


/**
 * Support for generating Objective C proto static libraries that registers actions that generate
 * the Objective C protos and validates the rules' attributes.
 *
 * This ProtoSupport class supports 2 protocol buffer compilers, named ProtocolBuffers2 and the
 * open-sourced version named protobuf. When refering to a specific library, the naming will either
 * refer to PB2 (for ProtocolBuffers2) or protobuf (the open-source version). When the context is
 * independent of the library, the naming will just refer to "proto". The selection of which proto
 * library to use depends on the presence of the 'portable_proto_filters' rule attribute.
 *
 * Keep in mind that these libraries are independent of the proto syntax used. ProtocolBuffers2
 * supports proto2 syntax, but the protobuf library supports both proto2 and proto3 syntax.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class ProtoSupport {
  private static final PathFragment BAZEL_TOOLS_PREFIX = new PathFragment("external/bazel_tools/");

  /**
   * Type of target that is generating the protos. There is a distinction because when generating
   * the protos from an objc_proto_library, the attributes of the rule need to be considered, but
   * when generating from a linking target (e.g. objc_binary), the attributes don't exist, but the
   * ObjcProtoProviders need to be checked.
   */
  enum TargetType {
    /**
     * The generating rule is an objc_proto_library rule.
     */
    PROTO_TARGET,

    /**
     * The generating target is a linking rule, which generates, compiles and links all the protos
     * in the transitive closure of dependencies.
     */
    LINKING_TARGET,
  }

  private static final String UNIQUE_DIRECTORY_NAME = "_generated_protos";

  /**
   * List of file name segments that should be upper cased when being generated. More information
   * available in the generateProtobufFilename() method.
   */
  private static final ImmutableSet<String> UPPERCASE_SEGMENTS =
      ImmutableSet.of("url", "http", "https");

  private final RuleContext ruleContext;
  private final ProtoAttributes attributes;
  private final TargetType targetType;
  private final IntermediateArtifacts intermediateArtifacts;

  /**
   * Creates a new proto support.
   *
   * @param ruleContext context this proto library is constructed in
   * @param targetType the type of target generating the protos
   */
  public ProtoSupport(RuleContext ruleContext, TargetType targetType) {
    this.ruleContext = ruleContext;
    this.attributes = new ProtoAttributes(ruleContext);
    this.targetType = targetType;
    if (targetType != TargetType.PROTO_TARGET) {
      // Use a a prefixed version of the intermediate artifacts to avoid naming collisions, as
      // the proto compilation step happens in the same context as the linking target.
      this.intermediateArtifacts =
          new IntermediateArtifacts(
              ruleContext, "_protos", "protos", ruleContext.getConfiguration());
    } else {
      this.intermediateArtifacts = ObjcRuleClasses.intermediateArtifacts(ruleContext);
    }
  }

  /**
   * Returns the intermediate artifacts associated with generated proto compilation.
   */
  public IntermediateArtifacts getIntermediateArtifacts() {
    return intermediateArtifacts;
  }

  /**
   * Registers actions required for compiling the proto files.
   *
   * @return this proto support
   */
  public ProtoSupport registerActions() {
    if (!Iterables.isEmpty(getFilteredProtoSources())) {
      registerProtoInputListFileAction();
      registerGenerateProtoFilesAction();
    }
    return this;
  }

  /**
   * Returns the common object for a proto specific compilation environment.
   */
  public ObjcCommon getCommon() {
    ObjcCommon.Builder commonBuilder =
        new ObjcCommon.Builder(ruleContext)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setHasModuleMap()
            .setCompilationArtifacts(getCompilationArtifacts());

    if (targetType == TargetType.LINKING_TARGET) {
      commonBuilder.addDepObjcProviders(
          ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class));
    } else if (targetType == TargetType.PROTO_TARGET) {
      commonBuilder.addDepObjcProviders(
          ruleContext.getPrerequisites(
              ObjcRuleClasses.PROTO_LIB_ATTR, Mode.TARGET, ObjcProvider.class));

      if (experimentalAutoUnion()) {
        commonBuilder.addDirectDependencyHeaderSearchPaths(getUserHeaderSearchPaths());
      } else {
        commonBuilder.addUserHeaderSearchPaths(getUserHeaderSearchPaths());
      }
    }
    return commonBuilder.build();
  }

  /**
   * Adds required configuration to the XcodeProvider support class for proto compilation.
   *
   * @param xcodeProviderBuilder The builder for the XcodeProvider support class.
   * @return this proto support
   */
  public ProtoSupport addXcodeProviderOptions(XcodeProvider.Builder xcodeProviderBuilder)
      throws RuleErrorException {
    xcodeProviderBuilder
        .addUserHeaderSearchPaths(getUserHeaderSearchPaths())
        .addHeaders(getGeneratedHeaders())
        .setCompilationArtifacts(getCompilationArtifacts());

    if (targetType == TargetType.PROTO_TARGET) {
      xcodeProviderBuilder.addCopts(ObjcRuleClasses.objcConfiguration(ruleContext).getCopts());
    } else if (targetType == TargetType.LINKING_TARGET) {
      Label protosLabel = null;
      try {
        protosLabel = ruleContext.getLabel().getLocalTargetLabel(
            ruleContext.getLabel().getName() + "_BundledProtos");
      } catch (LabelSyntaxException e) {
        ruleContext.throwWithRuleError(e.getLocalizedMessage());
      }
      ObjcCommon protoCommon = getCommon();
      new XcodeSupport(ruleContext, intermediateArtifacts, protosLabel)
          .addXcodeSettings(xcodeProviderBuilder,
              protoCommon.getObjcProvider(),
              XcodeProductType.LIBRARY_STATIC)
          .addDependencies(xcodeProviderBuilder, new Attribute("deps", Mode.TARGET));
    }
    return this;
  }

  /**
   * Adds the files needed to be built by the rule.
   *
   * @param filesToBuild An aggregated set of the files to be built by the rule.
   * @return this proto support
   */
  public ProtoSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild) {
    filesToBuild.addAll(getGeneratedSources()).addAll(getGeneratedHeaders());
    return this;
  }

  /**
   * Returns the proto compilation artifacts for the current rule.
   */
  public CompilationArtifacts getCompilationArtifacts() {
    ImmutableList<Artifact> generatedSources = getGeneratedSources();
    CompilationArtifacts.Builder builder =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(intermediateArtifacts)
            .setPchFile(Optional.<Artifact>absent())
            .addAdditionalHdrs(getGeneratedHeaders());

    if (experimentalAutoUnion()) {
      if (targetType == TargetType.LINKING_TARGET) {
        builder.addNonArcSrcs(generatedSources);
      }
    } else {
      if (targetType == TargetType.PROTO_TARGET) {
        builder.addNonArcSrcs(generatedSources);
      }
    }

    return builder.build();
  }

  /**
   * Returns the include paths for the generated protos.
   */
  public ImmutableSet<PathFragment> getUserHeaderSearchPaths() {
    return ImmutableSet.of(getWorkspaceRelativeOutputDir());
  }

  private Iterable<Artifact> getAllProtoSources() {
    NestedSetBuilder<Artifact> protos = NestedSetBuilder.stableOrder();

    if (experimentalAutoUnion() && targetType == TargetType.LINKING_TARGET) {
      Iterable<ObjcProtoProvider> objcProtoProviders =
          ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.class);
      for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
        protos.addTransitive(objcProtoProvider.getProtoSources());
      }
    }

    protos.addTransitive(attributes.getProtoFiles());

    return protos.build();
  }

  private Iterable<Artifact> getFilteredProtoSources() {
    // Transform the well known proto artifacts by removing the external/bazel_tools prefix if
    // present. Otherwise the comparison for filtering out the well known types is not possible.
    ImmutableSet.Builder<PathFragment> wellKnownProtoPathsBuilder = new ImmutableSet.Builder<>();
    for (Artifact wellKnownProto : attributes.getWellKnownTypeProtos()) {
      PathFragment execPath = wellKnownProto.getExecPath();
      if (execPath.startsWith(BAZEL_TOOLS_PREFIX)) {
        wellKnownProtoPathsBuilder.add(execPath.relativeTo(BAZEL_TOOLS_PREFIX));
      } else {
        wellKnownProtoPathsBuilder.add(execPath);
      }
    }

    ImmutableSet<PathFragment> wellKnownProtoPaths = wellKnownProtoPathsBuilder.build();

    // Filter out the well known types from being sent to be generated, as these protos have already
    // been generated and linked in libprotobuf.a.
    ImmutableSet.Builder<Artifact> filteredProtos = new ImmutableSet.Builder<>();
    for (Artifact proto : getAllProtoSources()) {
      if (!wellKnownProtoPaths.contains(proto.getExecPath())) {
        filteredProtos.add(proto);
      }
    }

    return filteredProtos.build();
  }

  private NestedSet<Artifact> getPortableProtoFilters() {
    NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.stableOrder();

    if (experimentalAutoUnion() && targetType == TargetType.LINKING_TARGET) {
      Iterable<ObjcProtoProvider> objcProtoProviders =
          ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProtoProvider.class);
      for (ObjcProtoProvider objcProtoProvider : objcProtoProviders) {
        portableProtoFilters.addTransitive(objcProtoProvider.getPortableProtoFilters());
      }
    }

    portableProtoFilters.addAll(attributes.getPortableProtoFilters());
    return portableProtoFilters.build();
  }

  private boolean experimentalAutoUnion() {
    ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
    return objcConfiguration.experimentalAutoTopLevelUnionObjCProtos();
  }

  private void registerProtoInputListFileAction() {
    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(),
            getProtoInputListFile(),
            getProtoInputListFileContents(),
            false));
  }

  private void registerGenerateProtoFilesAction() {
    ruleContext.registerAction(
        ObjcRuleClasses.spawnOnDarwinActionBuilder()
            .setMnemonic("GenObjcProtos")
            .addTransitiveInputs(getGenerateActionInputs())
            .addOutputs(getGenerateActionOutputs())
            .setExecutable(new PathFragment("/usr/bin/python"))
            .setCommandLine(getGenerateCommandLine())
            .build(ruleContext));
  }

  private Artifact getProtoInputListFile() {
    return ruleContext.getUniqueDirectoryArtifact(
        "_protos", "_proto_input_files", ruleContext.getConfiguration().getGenfilesDirectory());
  }

  private String getProtoInputListFileContents() {
    // Sort the file names to make the remote action key independent of the precise deps structure.
    // compile_protos.py will sort the input list anyway.
    Iterable<Artifact> sorted = Ordering.natural().immutableSortedCopy(getFilteredProtoSources());
    return Artifact.joinExecPaths("\n", sorted);
  }

  private PathFragment getWorkspaceRelativeOutputDir() {
    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path
    // of dependers.
    PathFragment rootRelativeOutputDir = ruleContext.getUniqueDirectory(UNIQUE_DIRECTORY_NAME);

    return new PathFragment(
        ruleContext.getBinOrGenfilesDirectory().getExecPath(), rootRelativeOutputDir);
  }

  private ImmutableList<Artifact> getGeneratedHeaders() {
    return generatedOutputArtifacts(FileType.of(".pbobjc.h"));
  }

  private ImmutableList<Artifact> getGeneratedSources() {
    return generatedOutputArtifacts(FileType.of(".pbobjc.m"));
  }

  private NestedSet<Artifact> getGenerateActionInputs() {
    NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(attributes.getProtoCompiler())
            .addAll(getAllProtoSources())
            .add(getProtoInputListFile())
            .addAll(attributes.getProtoCompilerSupport())
            .addTransitive(getPortableProtoFilters());

    return inputsBuilder.build();
  }

  private Iterable<Artifact> getGenerateActionOutputs() {
    return Iterables.concat(getGeneratedHeaders(), getGeneratedSources());
  }

  private CustomCommandLine getGenerateCommandLine() {
    return new CustomCommandLine.Builder()
        .add(attributes.getProtoCompiler().getExecPathString())
        .add("--input-file-list")
        .add(getProtoInputListFile().getExecPathString())
        .add("--output-dir")
        .add(getWorkspaceRelativeOutputDir().getSafePathString())
        .add("--force")
        .add("--proto-root-dir")
        .add(".")
        .addBeforeEachExecPath("--config", getPortableProtoFilters())
        .build();
  }

  private ImmutableList<Artifact> generatedOutputArtifacts(FileType newFileType) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact protoFile : getFilteredProtoSources()) {
      String protoFileName = FileSystemUtils.removeExtension(protoFile.getFilename());
      String generatedOutputName = generateProtobufFilename(protoFileName);
      PathFragment generatedFilePath =
          new PathFragment(
              protoFile.getRootRelativePath().getParentDirectory(),
              new PathFragment(generatedOutputName));

      PathFragment outputFile =
          FileSystemUtils.appendExtension(
              generatedFilePath, newFileType.getExtensions().get(0));

      if (outputFile != null) {
        builder.add(
            ruleContext.getUniqueDirectoryArtifact(
                UNIQUE_DIRECTORY_NAME, outputFile, ruleContext.getBinOrGenfilesDirectory()));
      }
    }
    return builder.build();
  }

  /**
   * Processes the case of the proto file name in the same fashion as the objective_c generator's
   * UnderscoresToCamelCase function.
   *
   * https://github.com/google/protobuf/blob/master/src/google/protobuf/compiler/objectivec/objectivec_helpers.cc
   */
  private String generateProtobufFilename(String protoFilename) {
    boolean lastCharWasDigit = false;
    boolean lastCharWasUpper = false;
    boolean lastCharWasLower = false;

    StringBuilder currentSegment = new StringBuilder();

    ArrayList<String> segments = new ArrayList<>();

    for (int i = 0; i < protoFilename.length(); i++) {
      char currentChar = protoFilename.charAt(i);
      if (CharMatcher.javaDigit().matches(currentChar)) {
        if (!lastCharWasDigit) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(currentChar);
        lastCharWasDigit = true;
        lastCharWasUpper = false;
        lastCharWasLower = false;
      } else if (CharMatcher.javaLowerCase().matches(currentChar)) {
        if (!lastCharWasLower && !lastCharWasUpper) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(currentChar);
        lastCharWasDigit = false;
        lastCharWasUpper = false;
        lastCharWasLower = true;
      } else if (CharMatcher.javaUpperCase().matches(currentChar)) {
        if (!lastCharWasUpper) {
          segments.add(currentSegment.toString());
          currentSegment = new StringBuilder();
        }
        currentSegment.append(Character.toLowerCase(currentChar));
        lastCharWasDigit = false;
        lastCharWasUpper = true;
        lastCharWasLower = false;
      } else {
        lastCharWasDigit = false;
        lastCharWasUpper = false;
        lastCharWasLower = false;
      }
    }

    segments.add(currentSegment.toString());

    StringBuilder casedSegments = new StringBuilder();
    for (String segment : segments) {
      if (UPPERCASE_SEGMENTS.contains(segment)) {
        casedSegments.append(segment.toUpperCase());
      } else {
        casedSegments.append(LOWER_UNDERSCORE.to(UPPER_CAMEL, segment));
      }
    }
    return casedSegments.toString();
  }
}
