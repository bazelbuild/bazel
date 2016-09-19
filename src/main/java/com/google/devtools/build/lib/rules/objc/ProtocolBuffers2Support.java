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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.base.Function;
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
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Support for generating Objective C proto static libraries that registers actions which generate
 * and compile the Objective C protos by using the deprecated ProtocolBuffers2 library and compiler.
 *
 * <p>Methods on this class can be called in any order without impacting the result.
 */
final class ProtocolBuffers2Support {

  private static final String UNIQUE_DIRECTORY_NAME = "_generated_protos";

  private static final Function<Artifact, PathFragment> PARENT_PATHFRAGMENT =
      new Function<Artifact, PathFragment>() {
        @Override
        public PathFragment apply(Artifact input) {
          return input.getExecPath().getParentDirectory();
        }
      };

  private final RuleContext ruleContext;
  private final ProtoAttributes attributes;

  /**
   * Creates a new proto support for the ProtocolBuffers2 library.
   *
   * @param ruleContext context this proto library is constructed in
   */
  public ProtocolBuffers2Support(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.attributes = new ProtoAttributes(ruleContext);
  }

  /**
   * Register the proto generation actions. These actions generate the ObjC/CPP code to be compiled
   * by this rule.
   */
  public ProtocolBuffers2Support registerGenerationActions() throws InterruptedException {
    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(),
            getProtoInputsFile(),
            getProtoInputsFileContents(
                attributes.filterWellKnownProtos(attributes.getProtoFiles())),
            false));

    ruleContext.registerAction(
        ObjcRuleClasses.spawnOnDarwinActionBuilder()
            .setMnemonic("GenObjcPB2Protos")
            .addInput(attributes.getProtoCompiler())
            .addInputs(attributes.getProtoCompilerSupport())
            .addInput(getProtoInputsFile())
            .addTransitiveInputs(attributes.getProtoFiles())
            .addInputs(attributes.getOptionsFile().asSet())
            .addOutputs(getGeneratedProtoOutputs(getHeaderExtension()))
            .addOutputs(getGeneratedProtoOutputs(getSourceExtension()))
            .setExecutable(new PathFragment("/usr/bin/python"))
            .setCommandLine(getGenerationCommandLine())
            .build(ruleContext));
    return this;
  }

  /** Registers the actions that will compile the generated code. */
  public ProtocolBuffers2Support registerCompilationActions() {
    new CompilationSupport(ruleContext)
        .registerCompileAndArchiveActions(getCommon())
        .registerGenerateModuleMapAction(Optional.of(getCompilationArtifacts()));
    return this;
  }

  /** Adds the generated files to the set of files to be output when this rule is built. */
  public ProtocolBuffers2Support addFilesToBuild(NestedSetBuilder<Artifact> filesToBuild)
      throws InterruptedException {
    filesToBuild
        .addAll(getGeneratedProtoOutputs(getHeaderExtension()))
        .addAll(getGeneratedProtoOutputs(getSourceExtension()))
        .addAll(getCompilationArtifacts().getArchive().asSet());
    return this;
  }

  /** Returns the ObjcProvider for this target. */
  public ObjcProvider getObjcProvider() {
    return getCommon().getObjcProvider();
  }

  /** Returns the XcodeProvider for this target. */
  public XcodeProvider getXcodeProvider() {
    XcodeProvider.Builder xcodeProviderBuilder =
        new XcodeProvider.Builder()
            .addUserHeaderSearchPaths(getUserHeaderSearchPaths())
            .setCompilationArtifacts(getCompilationArtifacts());

    new XcodeSupport(ruleContext)
        .addXcodeSettings(xcodeProviderBuilder, getCommon().getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(
            xcodeProviderBuilder, new Attribute(ObjcRuleClasses.PROTO_LIB_ATTR, Mode.TARGET));

    return xcodeProviderBuilder.build();
  }

  private String getHeaderExtension() {
    return ".pb" + (attributes.usesObjcHeaderNames() ? "objc.h" : ".h");
  }

  private String getSourceExtension() {
    return ".pb" + (attributes.outputsCpp() ? ".cc" : ".m");
  }

  private ObjcCommon getCommon() {
    return new ObjcCommon.Builder(ruleContext)
        .setIntermediateArtifacts(new IntermediateArtifacts(ruleContext, ""))
        .setHasModuleMap()
        .setCompilationArtifacts(getCompilationArtifacts())
        .addUserHeaderSearchPaths(getUserHeaderSearchPaths())
        .addDepObjcProviders(
            ruleContext.getPrerequisites(
                ObjcRuleClasses.PROTO_LIB_ATTR, Mode.TARGET, ObjcProvider.class))
        .build();
  }

  private CompilationArtifacts getCompilationArtifacts() {
    Iterable<Artifact> generatedSources = getGeneratedProtoOutputs(getSourceExtension());
    return new CompilationArtifacts.Builder()
        .setIntermediateArtifacts(new IntermediateArtifacts(ruleContext, ""))
        .setPchFile(Optional.<Artifact>absent())
        .addAdditionalHdrs(getGeneratedProtoOutputs(getHeaderExtension()))
        .addAdditionalHdrs(generatedSources)
        .addNonArcSrcs(generatedSources)
        .build();
  }

  private Artifact getProtoInputsFile() {
    return ruleContext.getUniqueDirectoryArtifact(
        "_protos", "_proto_input_files", ruleContext.getConfiguration().getGenfilesDirectory());
  }

  private String getProtoInputsFileContents(Iterable<Artifact> outputProtos) {
    // Sort the file names to make the remote action key independent of the precise deps structure.
    // compile_protos.py will sort the input list anyway.
    Iterable<Artifact> sorted = Ordering.natural().immutableSortedCopy(outputProtos);
    return Artifact.joinExecPaths("\n", sorted);
  }

  private CustomCommandLine getGenerationCommandLine() {
    CustomCommandLine.Builder commandLineBuilder =
        new CustomCommandLine.Builder()
            .add(attributes.getProtoCompiler().getExecPathString())
            .add("--input-file-list")
            .add(getProtoInputsFile().getExecPathString())
            .add("--output-dir")
            .add(getWorkspaceRelativeOutputDir().getSafePathString())
            .add("--working-dir")
            .add(".");

    if (attributes.getOptionsFile().isPresent()) {
      commandLineBuilder
          .add("--compiler-options-path")
          .add(attributes.getOptionsFile().get().getExecPathString());
    }

    if (attributes.outputsCpp()) {
      commandLineBuilder.add("--generate-cpp");
    }

    if (attributes.usesObjcHeaderNames()) {
      commandLineBuilder.add("--use-objc-header-names");
    }
    return commandLineBuilder.build();
  }

  public ImmutableSet<PathFragment> getUserHeaderSearchPaths() {
    ImmutableSet.Builder<PathFragment> searchPathEntriesBuilder =
        new ImmutableSet.Builder<PathFragment>().add(getWorkspaceRelativeOutputDir());

    if (attributes.needsPerProtoIncludes()) {
      PathFragment generatedProtoDir =
          new PathFragment(
              getWorkspaceRelativeOutputDir(), ruleContext.getLabel().getPackageFragment());

      searchPathEntriesBuilder
          .add(generatedProtoDir)
          .addAll(
              Iterables.transform(
                  getGeneratedProtoOutputs(getHeaderExtension()), PARENT_PATHFRAGMENT));
    }

    return searchPathEntriesBuilder.build();
  }

  private PathFragment getWorkspaceRelativeOutputDir() {
    // Generate sources in a package-and-rule-scoped directory; adds both the
    // package-and-rule-scoped directory and the header-containing-directory to the include path
    // of dependers.
    PathFragment rootRelativeOutputDir = ruleContext.getUniqueDirectory(UNIQUE_DIRECTORY_NAME);

    return new PathFragment(
        ruleContext.getBinOrGenfilesDirectory().getExecPath(), rootRelativeOutputDir);
  }

  private Iterable<Artifact> getGeneratedProtoOutputs(String extension) {
    ImmutableList.Builder<Artifact> builder = new ImmutableList.Builder<>();
    for (Artifact protoFile : attributes.filterWellKnownProtos(attributes.getProtoFiles())) {
      String protoFileName = FileSystemUtils.removeExtension(protoFile.getFilename());
      String generatedOutputName;
      if (attributes.outputsCpp()) {
        generatedOutputName = protoFileName;
      } else {
        generatedOutputName = attributes.getGeneratedProtoFilename(protoFileName, false);
      }

      PathFragment generatedFilePath =
          new PathFragment(
              protoFile.getRootRelativePath().getParentDirectory(),
              new PathFragment(generatedOutputName));

      PathFragment outputFile = FileSystemUtils.appendExtension(generatedFilePath, extension);

      if (outputFile != null) {
        builder.add(
            ruleContext.getUniqueDirectoryArtifact(
                UNIQUE_DIRECTORY_NAME, outputFile, ruleContext.getBinOrGenfilesDirectory()));
      }
    }
    return builder.build();
  }
}
