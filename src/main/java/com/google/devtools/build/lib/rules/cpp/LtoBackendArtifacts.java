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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * LtoBackendArtifacts represents a set of artifacts for a single ThinLTO backend compile.
 *
 * <p>ThinLTO expands the traditional 2 step compile (N x compile .cc, 1x link (N .o files) into a 4
 * step process:
 *
 * <ul>
 *   <li>1. Bitcode generation (N times). This is produces intermediate LLVM bitcode from a source
 *       file. For this product, it reuses the .o extension.
 *   <li>2. Indexing (once on N files). This takes all bitcode .o files, and for each .o file, it
 *       decides from which other .o files symbols can be inlined. In addition, it generates an
 *       index for looking up these symbols, and an imports file for identifying new input files for
 *       each step 3 {@link LtoBackendAction}.
 *   <li>3. Backend compile (N times). This is the traditional compilation, and uses the same
 *       command line as the Bitcode generation in 1). Since the compiler has many bit code files
 *       available, it can inline functions and propagate constants across .o files. This step is
 *       costly, as it will do traditional optimization. The result is a .lto.o file, a traditional
 *       ELF object file.
 *   <li>4. Backend link (once). This is the traditional link, and produces the final executable.
 * </ul>
 */
public final class LtoBackendArtifacts {

  // A file containing mapping of symbol => bitcode file containing the symbol.
  // It will be null when this is a shared non-lto backend.
  @Nullable private final Artifact index;

  // The bitcode file which is the input of the compile.
  private final Artifact bitcodeFile;

  // A file containing a list of bitcode files necessary to run the backend step.
  // It will be null when this is a shared non-lto backend.
  @Nullable private final Artifact imports;

  // The result of executing the above command line, an ELF object file.
  private final Artifact objectFile;

  // The corresponding dwoFile if fission is used.
  private Artifact dwoFile;

  LtoBackendArtifacts(
      RuleErrorConsumer ruleErrorConsumer,
      BuildOptions buildOptions,
      CppConfiguration cppConfiguration,
      PathFragment ltoOutputRootPrefix,
      Artifact bitcodeFile,
      BitcodeFiles allBitcodeFiles,
      ActionConstructionContext actionConstructionContext,
      RepositoryName repositoryName,
      BuildConfiguration configuration,
      LinkArtifactFactory linkArtifactFactory,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      boolean usePic,
      boolean generateDwo,
      List<String> userCompileFlags)
      throws RuleErrorException {
    this.bitcodeFile = bitcodeFile;
    PathFragment obj = ltoOutputRootPrefix.getRelative(bitcodeFile.getOutputDirRelativePath());

    objectFile =
        linkArtifactFactory.create(actionConstructionContext, repositoryName, configuration, obj);
    imports =
        linkArtifactFactory.create(
            actionConstructionContext,
            repositoryName,
            configuration,
            FileSystemUtils.appendExtension(obj, ".imports"));
    index =
        linkArtifactFactory.create(
            actionConstructionContext,
            repositoryName,
            configuration,
            FileSystemUtils.appendExtension(obj, ".thinlto.bc"));

    scheduleLtoBackendAction(
        ruleErrorConsumer,
        buildOptions,
        cppConfiguration,
        actionConstructionContext,
        repositoryName,
        featureConfiguration,
        ccToolchain,
        fdoContext,
        usePic,
        generateDwo,
        configuration,
        linkArtifactFactory,
        userCompileFlags,
        allBitcodeFiles);
  }

  // Interface to create an LTO backend that does not perform any cross-module optimization.
  public LtoBackendArtifacts(
      RuleErrorConsumer ruleErrorConsumer,
      BuildOptions buildOptions,
      CppConfiguration cppConfiguration,
      PathFragment ltoOutputRootPrefix,
      Artifact bitcodeFile,
      ActionConstructionContext actionConstructionContext,
      RepositoryName repositoryName,
      BuildConfiguration configuration,
      LinkArtifactFactory linkArtifactFactory,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      boolean usePic,
      boolean generateDwo,
      List<String> userCompileFlags)
      throws RuleErrorException {
    this.bitcodeFile = bitcodeFile;

    PathFragment obj = ltoOutputRootPrefix.getRelative(bitcodeFile.getOutputDirRelativePath());
    objectFile =
        linkArtifactFactory.create(actionConstructionContext, repositoryName, configuration, obj);
    imports = null;
    index = null;

    scheduleLtoBackendAction(
        ruleErrorConsumer,
        buildOptions,
        cppConfiguration,
        actionConstructionContext,
        repositoryName,
        featureConfiguration,
        ccToolchain,
        fdoContext,
        usePic,
        generateDwo,
        configuration,
        linkArtifactFactory,
        userCompileFlags,
        /*bitcodeFiles=*/ null);
  }

  public Artifact getObjectFile() {
    return objectFile;
  }

  Artifact getBitcodeFile() {
    return bitcodeFile;
  }

  public Artifact getDwoFile() {
    return dwoFile;
  }

  void addIndexingOutputs(ImmutableSet.Builder<Artifact> builder) {
    // For objects from linkstatic libraries, we may not be including them in the LTO indexing
    // step when linked into a test, but rather will use shared non-LTO backends for better
    // scalability when running large numbers of tests.
    if (index == null) {
      return;
    }
    builder.add(imports);
    builder.add(index);
  }

  private void scheduleLtoBackendAction(
      RuleErrorConsumer ruleErrorConsumer,
      BuildOptions buildOptions,
      CppConfiguration cppConfiguration,
      ActionConstructionContext actionConstructionContext,
      RepositoryName repositoryName,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider ccToolchain,
      FdoContext fdoContext,
      boolean usePic,
      boolean generateDwo,
      BuildConfiguration configuration,
      LinkArtifactFactory linkArtifactFactory,
      List<String> userCompileFlags,
      @Nullable BitcodeFiles bitcodeFiles)
      throws RuleErrorException {
    LtoBackendAction.Builder builder = new LtoBackendAction.Builder();

    builder.addInput(bitcodeFile);

    Preconditions.checkState(
        (index == null) == (imports == null),
        "Either both or neither index and imports files should be null");
    if (imports != null) {
      builder.addImportsInfo(bitcodeFiles, imports);
      // Although the imports file is not used by the LTOBackendAction while the action is
      // executing, it is needed during the input discovery phase, and we must list it as an input
      // to the action in order for it to be preserved under --discard_orphaned_artifacts.
      builder.addInput(imports);
    }
    if (index != null) {
      builder.addInput(index);
    }
    builder.addTransitiveInputs(ccToolchain.getCompilerFiles());

    builder.addOutput(objectFile);

    builder.setProgressMessage("LTO Backend Compile %s", objectFile.getExecPath());
    builder.setMnemonic("CcLtoBackendCompile");

    CcToolchainVariables.Builder buildVariablesBuilder =
        CcToolchainVariables.builder(ccToolchain.getBuildVariables(buildOptions, cppConfiguration));
    if (index != null) {
      buildVariablesBuilder.addStringVariable("thinlto_index", index.getExecPath().toString());
    } else {
      // An empty input indicates not to perform cross-module optimization.
      buildVariablesBuilder.addStringVariable("thinlto_index", "/dev/null");
    }
    // The output from the LTO backend step is a native object file.
    buildVariablesBuilder.addStringVariable(
        "thinlto_output_object_file", objectFile.getExecPath().toString());
    // The input to the LTO backend step is the bitcode file.
    buildVariablesBuilder.addStringVariable(
        "thinlto_input_bitcode_file", bitcodeFile.getExecPath().toString());
    addProfileForLtoBackend(builder, fdoContext, featureConfiguration, buildVariablesBuilder);
    // Add the context sensitive instrument path to the backend.
    if (featureConfiguration.isEnabled(CppRuleClasses.CS_FDO_INSTRUMENT)) {
      buildVariablesBuilder.addStringVariable(
          CompileBuildVariables.CS_FDO_INSTRUMENT_PATH.getVariableName(),
          ccToolchain.getCSFdoInstrument());
    }

    if (generateDwo) {
      dwoFile =
          linkArtifactFactory.create(
              actionConstructionContext,
              repositoryName,
              configuration,
              FileSystemUtils.replaceExtension(objectFile.getOutputDirRelativePath(), ".dwo"));
      builder.addOutput(dwoFile);
      buildVariablesBuilder.addStringVariable(
          CompileBuildVariables.PER_OBJECT_DEBUG_INFO_FILE.getVariableName(),
          dwoFile.getExecPathString());
      buildVariablesBuilder.addStringVariable(
          CompileBuildVariables.IS_USING_FISSION.getVariableName(), "");
    }
    buildVariablesBuilder.addStringSequenceVariable(
        CompileBuildVariables.USER_COMPILE_FLAGS.getVariableName(), userCompileFlags);

    CcToolchainVariables buildVariables = buildVariablesBuilder.build();

    if (cppConfiguration.useStandaloneLtoIndexingCommandLines()) {
      if (!featureConfiguration.actionIsConfigured(CppActionNames.LTO_BACKEND)) {
        throw ruleErrorConsumer.throwWithRuleError(
            "Thinlto build is requested, but the C++ toolchain doesn't define an action_config for"
                + " 'lto-backend' action.");
      }
      PathFragment compiler =
          PathFragment.create(
              featureConfiguration.getToolPathForAction(CppActionNames.LTO_BACKEND));
      builder.setExecutable(compiler);
    } else {
      PathFragment compiler = ccToolchain.getToolPathFragment(Tool.GCC, ruleErrorConsumer);
      builder.setExecutable(compiler);
    }

    CommandLine ltoCommandLine =
        new CommandLine() {

          @Override
          public Iterable<String> arguments() throws CommandLineExpansionException {
            return arguments(/* artifactExpander= */ null);
          }

          @Override
          public Iterable<String> arguments(ArtifactExpander artifactExpander)
              throws CommandLineExpansionException {
            ImmutableList.Builder<String> args = ImmutableList.builder();
            try {
              args.addAll(
                  featureConfiguration.getCommandLine(
                      CppActionNames.LTO_BACKEND, buildVariables, artifactExpander));
            } catch (ExpansionException e) {
              throw new CommandLineExpansionException(e.getMessage());
            }
            // If this is a PIC compile (set based on the CppConfiguration), the PIC
            // option should be added after the rest of the command line so that it
            // cannot be overridden. This is consistent with the ordering in the
            // CppCompileAction's compiler options.
            if (usePic) {
              args.add("-fPIC");
            }
            return args.build();
          }
        };
    builder.addCommandLine(ltoCommandLine);

    actionConstructionContext.registerAction(builder.build(actionConstructionContext));
  }

  /**
   * Adds the AFDO profile path to the variable builder and the profile tothe inputs of the action.
   */
  @ThreadSafe
  private static void addProfileForLtoBackend(
      LtoBackendAction.Builder builder,
      FdoContext fdoContext,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables.Builder buildVariables) {
    Artifact prefetch = fdoContext.getPrefetchHintsArtifact();
    if (prefetch != null) {
      buildVariables.addStringVariable("fdo_prefetch_hints_path", prefetch.getExecPathString());
      builder.addInput(fdoContext.getPrefetchHintsArtifact());
    }
    if (!featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)
        && !featureConfiguration.isEnabled(CppRuleClasses.CS_FDO_OPTIMIZE)
        && !featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO)) {
      return;
    }

    FdoContext.BranchFdoProfile branchFdoProfile =
        Preconditions.checkNotNull(fdoContext.getBranchFdoProfile());
    Artifact profile = branchFdoProfile.getProfileArtifact();
    buildVariables.addStringVariable("fdo_profile_path", profile.getExecPathString());
    builder.addInput(branchFdoProfile.getProfileArtifact());
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof LtoBackendArtifacts)) {
      return false;
    }
    LtoBackendArtifacts that = (LtoBackendArtifacts) o;
    return Objects.equals(index, that.index)
        && bitcodeFile.equals(that.bitcodeFile)
        && Objects.equals(imports, that.imports)
        && objectFile.equals(that.objectFile)
        && Objects.equals(dwoFile, that.dwoFile);
  }

  @Override
  public int hashCode() {
    return Objects.hash(index, bitcodeFile, imports, objectFile, dwoFile);
  }
}
