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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import javax.annotation.Nullable;

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
  private static final Function<Artifact, PathFragment> PARENT_PATHFRAGMENT =
      new Function<Artifact, PathFragment>() {
        @Override
        public PathFragment apply(Artifact input) {
          return input.getExecPath().getParentDirectory();
        }
      };

  @VisibleForTesting
  static final String NO_PROTOS_ERROR =
      "no protos to compile - a non-empty deps attribute is required";

  @VisibleForTesting
  static final String FILES_DEPRECATED_WARNING =
      "Using files and filegroups in objc_proto_library is deprecated";

  @VisibleForTesting
  static final String PORTABLE_PROTO_FILTERS_NOT_EXCLUSIVE_ERROR =
      "The portable_proto_filters attribute is incompatible with the options_file, output_cpp, "
          + "per_proto_includes and use_objc_header_names attributes.";

  @VisibleForTesting
  static final String PORTABLE_PROTO_FILTERS_EMPTY_ERROR =
      "The portable_proto_filters attribute can't be empty";

  private static final String UNIQUE_DIRECTORY_NAME = "_generated_protos";

  private final RuleContext ruleContext;
  private final Attributes attributes;

  /**
   * Creates a new proto support.
   *
   * @param ruleContext context this proto library is constructed in
   */
  public ProtoSupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.attributes = new Attributes(ruleContext);
  }

  /**
   * Validates proto support.
   * <ul>
   * <li>Validates that there are protos specified to be compiled.
   * <li>Validates that, when enabling the open source protobuf library, the options for the PB2
   *     are not specified also.
   * <li>Validates that, when enabling the open source protobuf library, the rule specifies at least
   *     one portable proto filter file.
   * </ul>
   *
   * @return this proto support
   */
  public ProtoSupport validate() {
    if (attributes.getProtoFiles().isEmpty()) {
      ruleContext.ruleError(NO_PROTOS_ERROR);
    }

    if (attributes.usesProtobufLibrary()) {
      if (attributes.getPortableProtoFilters().isEmpty()) {
        ruleContext.ruleError(PORTABLE_PROTO_FILTERS_EMPTY_ERROR);
      }

      if (attributes.outputsCpp()
          || attributes.usesObjcHeaderNames()
          || attributes.needsPerProtoIncludes()
          || attributes.getOptionsFile() != null) {
        ruleContext.ruleError(PORTABLE_PROTO_FILTERS_NOT_EXCLUSIVE_ERROR);
      }
    }
    return this;
  }

  /**
   * Registers actions required for compiling the proto files.
   *
   * @return this proto support
   */
  public ProtoSupport registerActions() {
    registerProtoInputListFileAction();
    if (!attributes.getProtoFiles().isEmpty()) {
      registerGenerateProtoFilesAction();
    }
    return this;
  }

  /**
   * Adds required configuration to the ObjcCommon support class for proto compilation.
   *
   * @param commonBuilder The builder for the ObjcCommon support class.
   * @return this bundle support
   */
  public ProtoSupport addCommonOptions(ObjcCommon.Builder commonBuilder) {
    commonBuilder
        .setCompilationArtifacts(getCompilationArtifacts())
        .addUserHeaderSearchPaths(getSearchPathEntries())
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .addDepObjcProviders(
            ruleContext.getPrerequisites(
                ObjcProtoLibraryRule.PROTO_LIB_ATTR, Mode.TARGET, ObjcProvider.class))
        .setHasModuleMap();
    return this;
  }

  /**
   * Adds required configuration to the XcodeProvider support class for proto compilation.
   *
   * @param xcodeProviderBuilder The builder for the XcodeProvider support class.
   * @return this bundle support
   */
  public ProtoSupport addXcodeProviderOptions(XcodeProvider.Builder xcodeProviderBuilder) {
    xcodeProviderBuilder
        .addUserHeaderSearchPaths(getSearchPathEntries())
        .addCopts(ObjcRuleClasses.objcConfiguration(ruleContext).getCopts())
        .addHeaders(getGeneratedHeaders())
        .setCompilationArtifacts(getCompilationArtifacts());
    return this;
  }

  /**
   * Adds the files needed to be built by the rule.
   *
   * @param filesToBuild An aggregated set of the files to be built by the rule.
   * @return this bundle support
   */
  public ProtoSupport addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild) {
    filesToBuild.addAll(getGeneratedSources()).addAll(getGeneratedHeaders());
    return this;
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
            .addInputs(getGenerateActionInputs())
            .addOutputs(getGenerateActionOutputs())
            .setExecutable(new PathFragment("/usr/bin/python"))
            .setCommandLine(getGenerateCommandLine())
            .build(ruleContext));
  }

  private PathFragment getWorkspaceRelativeOutputDir() {
    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path
    // of dependers.
    PathFragment rootRelativeOutputDir = ruleContext.getUniqueDirectory(UNIQUE_DIRECTORY_NAME);

    return new PathFragment(
        ruleContext.getBinOrGenfilesDirectory().getExecPath(), rootRelativeOutputDir);
  }

  private CompilationArtifacts getCompilationArtifacts() {
    ImmutableList<Artifact> generatedSources = getGeneratedSources();
    return new CompilationArtifacts.Builder()
        .addNonArcSrcs(generatedSources)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .setPchFile(Optional.<Artifact>absent())
        .addAdditionalHdrs(getGeneratedHeaders())
        .addAdditionalHdrs(generatedSources)
        .build();
  }

  private ImmutableList<Artifact> getGeneratedHeaders() {
    boolean useObjcName = attributes.usesObjcHeaderNames() || attributes.usesProtobufLibrary();
    return generatedOutputArtifacts(FileType.of(".pb" + (useObjcName ? "objc.h" : ".h")));
  }

  private ImmutableList<Artifact> getGeneratedSources() {
    return generatedOutputArtifacts(
        FileType.of(
            ".pb"
                + (attributes.usesProtobufLibrary() ? "objc" : "")
                + (attributes.outputsCpp() ? ".cc" : ".m")));
  }

  private ImmutableSet<PathFragment> getSearchPathEntries() {
    PathFragment workspaceRelativeOutputDir = getWorkspaceRelativeOutputDir();

    ImmutableSet.Builder<PathFragment> searchPathEntriesBuilder =
        new ImmutableSet.Builder<PathFragment>().add(workspaceRelativeOutputDir);

    if (attributes.needsPerProtoIncludes()) {
      PathFragment generatedProtoDir =
          new PathFragment(workspaceRelativeOutputDir, ruleContext.getLabel().getPackageFragment());

      searchPathEntriesBuilder
          .add(generatedProtoDir)
          .addAll(Iterables.transform(getGeneratedHeaders(), PARENT_PATHFRAGMENT));
    }

    return searchPathEntriesBuilder.build();
  }

  private Artifact getProtoInputListFile() {
    return ruleContext.getUniqueDirectoryArtifact(
        "_protos", "_proto_input_files", ruleContext.getConfiguration().getGenfilesDirectory());
  }

  private String getProtoInputListFileContents() {
    return Artifact.joinExecPaths("\n", attributes.getProtoFiles());
  }

  private NestedSet<Artifact> getGenerateActionInputs() {
    NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(attributes.getProtoCompiler())
            .addAll(attributes.getProtoFiles())
            .add(getProtoInputListFile())
            .addAll(attributes.getProtoLibrary())
            .addAll(attributes.getProtoCompilerSupport())
            .addAll(attributes.getPortableProtoFilters());

    Artifact optionsFile = attributes.getOptionsFile();
    if (optionsFile != null) {
      inputsBuilder.add(optionsFile);
    }

    return inputsBuilder.build();
  }

  private Iterable<Artifact> getGenerateActionOutputs() {
    return Iterables.concat(getGeneratedHeaders(), getGeneratedSources());
  }

  private CustomCommandLine getGenerateCommandLine() {
    if (attributes.usesProtobufLibrary()) {
      return getProtobufCommandLine();
    } else {
      return getPb2CommandLine();
    }
  }

  private CustomCommandLine getPb2CommandLine() {
    CustomCommandLine.Builder commandLineBuilder =
        new CustomCommandLine.Builder()
            .add(attributes.getProtoCompiler().getExecPathString())
            .add("--input-file-list")
            .add(getProtoInputListFile().getExecPathString())
            .add("--output-dir")
            .add(getWorkspaceRelativeOutputDir().getSafePathString())
            .add("--working-dir")
            .add(".");

    if (attributes.getOptionsFile() != null) {
      commandLineBuilder
          .add("--compiler-options-path")
          .add(attributes.getOptionsFile().getExecPathString());
    }

    if (attributes.outputsCpp()) {
      commandLineBuilder.add("--generate-cpp");
    }

    if (attributes.usesObjcHeaderNames()) {
      commandLineBuilder.add("--use-objc-header-names");
    }
    return commandLineBuilder.build();
  }

  private CustomCommandLine getProtobufCommandLine() {
    CustomCommandLine.Builder commandLineBuilder =
        new CustomCommandLine.Builder()
            .add(attributes.getProtoCompiler().getExecPathString())
            .add("--input-file-list")
            .add(getProtoInputListFile().getExecPathString())
            .add("--output-dir")
            .add(getWorkspaceRelativeOutputDir().getSafePathString())
            .add("--force")
            .add("--proto-root-dir")
            .add(".");

    boolean configAdded = false;
    for (Artifact portableProtoFilter : attributes.getPortableProtoFilters()) {
      String configFlag;
      if (!configAdded) {
        configFlag = "--config";
        configAdded = true;
      } else {
        configFlag = "--extra-filter-config";
      }

      commandLineBuilder.add(configFlag).add(portableProtoFilter.getExecPathString());
    }
    return commandLineBuilder.build();
  }

  private ImmutableList<Artifact> generatedOutputArtifacts(FileType newFileType) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact protoFile : attributes.getProtoFiles()) {
      String generatedOutputName;
      if (attributes.outputsCpp()) {
        generatedOutputName = protoFile.getFilename();
      } else {
        String lowerUnderscoreBaseName = protoFile.getFilename().replace('-', '_').toLowerCase();
        generatedOutputName = LOWER_UNDERSCORE.to(UPPER_CAMEL, lowerUnderscoreBaseName);
      }

      PathFragment generatedFilePath =
          new PathFragment(
              protoFile.getRootRelativePath().getParentDirectory(),
              new PathFragment(generatedOutputName));

      PathFragment outputFile =
          FileSystemUtils.replaceExtension(
              generatedFilePath, newFileType.getExtensions().get(0), ".proto");

      if (outputFile != null) {
        builder.add(
            ruleContext.getUniqueDirectoryArtifact(
                UNIQUE_DIRECTORY_NAME, outputFile, ruleContext.getBinOrGenfilesDirectory()));
      }
    }
    return builder.build();
  }

  /**
   * Common rule attributes used by an Objective C proto library.
   */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Returns whether the generated files should be C++ or Objective C.
     */
    boolean outputsCpp() {
      return ruleContext.attributes().get(ObjcProtoLibraryRule.OUTPUT_CPP_ATTR, Type.BOOLEAN);
    }

    /**
     * Returns whether the generated header files should have be of type pb.h or pbobjc.h.
     */
    boolean usesObjcHeaderNames() {
      return ruleContext
          .attributes()
          .get(ObjcProtoLibraryRule.USE_OBJC_HEADER_NAMES_ATTR, Type.BOOLEAN);
    }

    /**
     * Returns whether the includes should include each of the proto generated headers.
     */
    boolean needsPerProtoIncludes() {
      return ruleContext
          .attributes()
          .get(ObjcProtoLibraryRule.PER_PROTO_INCLUDES_ATTR, Type.BOOLEAN);
    }

    /**
     * Returns whether to use the protobuf library instead of the PB2 library.
     */
    boolean usesProtobufLibrary() {
      return ruleContext
          .attributes()
          .isAttributeValueExplicitlySpecified(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR);
    }

    /**
     * Returns the list of portable proto filters.
     */
    ImmutableList<Artifact> getPortableProtoFilters() {
      return ruleContext
          .getPrerequisiteArtifacts(ObjcProtoLibraryRule.PORTABLE_PROTO_FILTERS_ATTR, Mode.HOST)
          .list();
    }

    /**
     * Returns the options file, or null if it was not specified.
     */
    @Nullable
    Artifact getOptionsFile() {
      return ruleContext.getPrerequisiteArtifact(ObjcProtoLibraryRule.OPTIONS_FILE_ATTR, Mode.HOST);
    }

    /**
     * Returns the proto compiler to be used.
     */
    Artifact getProtoCompiler() {
      return ruleContext.getPrerequisiteArtifact(
          ObjcProtoLibraryRule.PROTO_COMPILER_ATTR, Mode.HOST);
    }

    /**
     * Returns the list of files needed by the proto compiler.
     */
    ImmutableList<Artifact> getProtoCompilerSupport() {
      return ruleContext
          .getPrerequisiteArtifacts(ObjcProtoLibraryRule.PROTO_COMPILER_SUPPORT_ATTR, Mode.HOST)
          .list();
    }

    /**
     * Returns the list of files that compose the proto library. This is the implicit dependency
     * added to the objc_proto_library target.
     */
    ImmutableList<Artifact> getProtoLibrary() {
      return ruleContext
          .getPrerequisiteArtifacts(ObjcProtoLibraryRule.PROTO_LIB_ATTR, Mode.TARGET)
          .list();
    }

    /**
     * Returns the list of proto files to compile.
     */
    NestedSet<Artifact> getProtoFiles() {
      return NestedSetBuilder.<Artifact>stableOrder()
          .addAll(getProtoDepsFiles())
          .addTransitive(getProtoDepsSources())
          .build();
    }

    /**
     * Returns the list of proto files that were added directly into the deps attributes. This way
     * of specifying the protos is deprecated, and displays a warning when the target does so.
     */
    private ImmutableList<Artifact> getProtoDepsFiles() {
      PrerequisiteArtifacts prerequisiteArtifacts =
          ruleContext.getPrerequisiteArtifacts("deps", Mode.TARGET);
      ImmutableList<Artifact> protos = prerequisiteArtifacts.filter(FileType.of(".proto")).list();
      if (!protos.isEmpty()) {
        ruleContext.attributeWarning("deps", FILES_DEPRECATED_WARNING);
      }
      return protos;
    }

    /**
     * Returns the list of proto files that were added using proto_library dependencies.
     */
    private NestedSet<Artifact> getProtoDepsSources() {
      NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.stableOrder();
      Iterable<ProtoSourcesProvider> providers =
          ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoSourcesProvider.class);
      for (ProtoSourcesProvider provider : providers) {
        artifacts.addTransitive(provider.getTransitiveProtoSources());
      }
      return artifacts.build();
    }
  }
}
