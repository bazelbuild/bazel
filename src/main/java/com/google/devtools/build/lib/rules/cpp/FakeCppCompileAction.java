// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.logging.Logger;

/**
 * Action that represents a fake C++ compilation step.
 */
@ThreadCompatible
public class FakeCppCompileAction extends CppCompileAction {

  private static final Logger logger = Logger.getLogger(FakeCppCompileAction.class.getName());

  public static final UUID GUID = UUID.fromString("8ab63589-be01-4a39-b770-b98ae8b03493");

  private final PathFragment tempOutputFile;

  FakeCppCompileAction(
      ActionOwner owner,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables variables,
      Artifact sourceFile,
      CppConfiguration cppConfiguration,
      boolean shareable,
      boolean shouldScanIncludes,
      boolean shouldPruneModules,
      boolean usePic,
      boolean useHeaderModules,
      NestedSet<Artifact> mandatoryInputs,
      NestedSet<Artifact> inputsForInvalidation,
      ImmutableList<Artifact> builtinIncludeFiles,
      NestedSet<Artifact> prunableHeaders,
      Artifact outputFile,
      PathFragment tempOutputFile,
      Artifact dotdFile,
      ActionEnvironment env,
      CcCompilationContext ccCompilationContext,
      CoptsFilter nocopts,
      CppSemantics cppSemantics,
      ImmutableList<PathFragment> builtInIncludeDirectories,
      ImmutableMap<String, String> executionInfo,
      Artifact grepIncludes) {
    super(
        owner,
        featureConfiguration,
        variables,
        sourceFile,
        cppConfiguration,
        shareable,
        shouldScanIncludes,
        shouldPruneModules,
        usePic,
        useHeaderModules,
        mandatoryInputs,
        inputsForInvalidation,
        builtinIncludeFiles,
        prunableHeaders,
        outputFile,
        dotdFile,
        /* gcnoFile=*/ null,
        /* dwoFile=*/ null,
        /* ltoIndexingFile=*/ null,
        env,
        // We only allow inclusion of header files explicitly declared in
        // "srcs", so we only use declaredIncludeSrcs, not declaredIncludeDirs.
        // (Disallowing use of undeclared headers for cc_fake_binary is needed
        // because the header files get included in the runfiles for the
        // cc_fake_binary and for the negative compilation tests that depend on
        // the cc_fake_binary, and the runfiles must be determined at analysis
        // time, so they can't depend on the contents of the ".d" file.)
        CcCompilationContext.disallowUndeclaredHeaders(ccCompilationContext),
        nocopts,
        /* additionalIncludeScanningRoots=*/ ImmutableList.of(),
        GUID,
        executionInfo,
        CppActionNames.CPP_COMPILE,
        cppSemantics,
        builtInIncludeDirectories,
        grepIncludes);
    this.tempOutputFile = Preconditions.checkNotNull(tempOutputFile);
  }

  @Override
  @ThreadCompatible
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    setModuleFileFlags();
    List<SpawnResult> spawnResults;
    // First, do a normal compilation, to generate the ".d" file. The generated object file is built
    // to a temporary location (tempOutputFile) and ignored afterwards.
    logger.info("Generating " + getDotdFile());
    byte[] dotDContents = null;
    try {
      Spawn spawn = createSpawn(actionExecutionContext.getClientEnv());
      SpawnActionContext context = actionExecutionContext.getContext(SpawnActionContext.class);
      spawnResults = context.exec(spawn, actionExecutionContext);
      // The SpawnActionContext guarantees that the first list entry is the successful one.
      dotDContents = getDotDContents(spawnResults.get(0));
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          "C++ compilation of rule '" + getOwner().getLabel() + "'",
          actionExecutionContext.getVerboseFailures(),
          this);
    } finally {
      clearAdditionalInputs();
    }
    CppIncludeExtractionContext scanningContext =
        actionExecutionContext.getContext(CppIncludeExtractionContext.class);
    Path execRoot = actionExecutionContext.getExecRoot();

    NestedSet<Artifact> discoveredInputs;
    if (getDotdFile() == null) {
      discoveredInputs = NestedSetBuilder.<Artifact>stableOrder().build();
    } else {
      HeaderDiscovery.Builder discoveryBuilder =
          new HeaderDiscovery.Builder()
              .setAction(this)
              .setSourceFile(getSourceFile())
              .setDependencies(
                  processDepset(actionExecutionContext, execRoot, dotDContents).getDependencies())
              .setPermittedSystemIncludePrefixes(getPermittedSystemIncludePrefixes(execRoot))
              .setAllowedDerivedInputs(getAllowedDerivedInputs());

      if (needsIncludeValidation) {
        discoveryBuilder.shouldValidateInclusions();
      }

      discoveredInputs =
          discoveryBuilder
              .build()
              .discoverInputsFromDependencies(execRoot, scanningContext.getArtifactResolver());
    }

    dotDContents = null; // Garbage collect in-memory .d contents.

    // Even cc_fake_binary rules need to properly declare their dependencies...
    // In fact, they need to declare their dependencies even more than cc_binary rules do.
    // CcCommonConfiguredTarget passes in an empty set of declaredIncludeDirs,
    // so this check below will only allow inclusion of header files that are explicitly
    // listed in the "srcs" of the cc_fake_binary or in the "srcs" of a cc_library that it
    // depends on.
    try {
      validateInclusions(actionExecutionContext, discoveredInputs);
    } catch (ActionExecutionException e) {
      // TODO(bazel-team): (2009) make this into an error, once most of the current warnings
      // are fixed.
      actionExecutionContext
          .getEventHandler()
          .handle(
              Event.warn(
                  getOwner().getLocation(),
                  e.getMessage() + ";\n  this warning may eventually become an error"));
    }

    if (discoversInputs()) {
      updateActionInputs(discoveredInputs);
    } else {
      Preconditions.checkState(
          discoveredInputs.isEmpty(),
          "Discovered inputs without discovering inputs? %s %s",
          discoveredInputs,
          this);
    }

    // Generate a fake ".o" file containing the command line needed to generate
    // the real object file.
    logger.info("Generating " + outputFile);

    // A cc_fake_binary rule generates fake .o files and a fake target file,
    // which merely contain instructions on building the real target. We need to
    // be careful to use a new set of output file names in the instructions, as
    // to not overwrite the fake output files when someone tries to follow the
    // instructions. As the real compilation is executed by the test from its
    // runfiles directory (where writing is forbidden), we patch the command
    // line to write to $TEST_TMPDIR instead.
    final String outputPrefix = "$TEST_TMPDIR/";
    String argv;
    try {
      argv =
          getArguments().stream()
              .map(
                  input -> {
                    String result = ShellEscaper.escapeString(input);
                    // Replace -c <tempOutputFile> so it's -c <outputFile>.
                    if (input.equals(tempOutputFile.getPathString())) {
                      result =
                          outputPrefix + ShellEscaper.escapeString(outputFile.getExecPathString());
                    }
                    if (input.equals(outputFile.getExecPathString())
                        || (getDotdFile() != null
                            && input.equals(getDotdFile().getExecPathString()))) {
                      result = outputPrefix + ShellEscaper.escapeString(input);
                    }
                    return result;
                  })
              .collect(joining(" "));
    } catch (CommandLineExpansionException e) {
      throw new ActionExecutionException(
          "failed to generate compile command for rule '"
              + getOwner().getLabel()
              + ": "
              + e.getMessage(),
          this,
          /* catastrophe= */ false);
    }

    // Write the command needed to build the real .o file to the fake .o file.
    // Generate a command to ensure that the output directory exists; otherwise
    // the compilation would fail.
    try {
      // Ensure that the .d file and .o file are siblings, so that the "mkdir" below works for
      // both.
      Preconditions.checkState(
          getDotdFile() == null
              || outputFile
                  .getExecPath()
                  .getParentDirectory()
                  .equals(getDotdFile().getExecPath().getParentDirectory()));
      FileSystemUtils.writeContent(
          actionExecutionContext.getInputPath(outputFile),
          ISO_8859_1,
          actionExecutionContext.getInputPath(outputFile).getBaseName()
              + ": "
              + "mkdir -p "
              + outputPrefix
              + "$(dirname "
              + outputFile.getExecPath()
              + ")"
              + " && "
              + argv
              + "\n");
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create fake compile command for rule '"
          + getOwner().getLabel() + ": " + e.getMessage(), this, false);
    }
    return ActionResult.create(spawnResults);
  }

  @Override
  public String getMnemonic() {
    return "FakeCppCompile";
  }

  @Override
  public ResourceSet estimateResourceConsumptionLocal() {
    return AbstractAction.DEFAULT_RESOURCE_SET;
  }
}
