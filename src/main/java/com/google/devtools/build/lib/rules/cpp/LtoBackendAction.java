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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.LtoAction;
import com.google.devtools.build.lib.server.FailureDetails.LtoAction.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Action used by LtoBackendArtifacts to create an LtoBackendAction. Similar to {@link SpawnAction},
 * except that inputs are discovered from the imports file created by the ThinLTO indexing step for
 * each backend artifact.
 *
 * <p>See {@link LtoBackendArtifacts} for a high level description of the ThinLTO build process. The
 * LTO indexing step takes all bitcode .o files and decides which other .o file symbols can be
 * imported/inlined. The additional input files for each backend action are then written to an
 * imports file. Therefore these new inputs must be discovered here by subsetting the imports paths
 * from the set of all bitcode artifacts, before executing the backend action.
 *
 * <p>For more information on ThinLTO see
 * http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html.
 */
public final class LtoBackendAction extends SpawnAction {
  private static final String GUID = "72ce1eca-4625-4e24-a0d8-bb91bb8b0e0e";

  private final NestedSet<Artifact> mandatoryInputs;
  private final BitcodeFiles bitcodeFiles;
  private final Artifact imports;

  public LtoBackendAction(
      NestedSet<Artifact> inputs,
      @Nullable BitcodeFiles allBitcodeFiles,
      @Nullable Artifact importsFile,
      Collection<Artifact> outputs,
      Artifact primaryOutput,
      ActionOwner owner,
      CommandLines argv,
      CommandLineLimits commandLineLimits,
      boolean isShellCommand,
      ActionEnvironment env,
      Map<String, String> executionInfo,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier,
      String mnemonic) {
    super(
        owner,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        inputs,
        outputs,
        primaryOutput,
        AbstractAction.DEFAULT_RESOURCE_SET,
        argv,
        commandLineLimits,
        isShellCommand,
        env,
        ImmutableMap.copyOf(executionInfo),
        progressMessage,
        runfilesSupplier,
        mnemonic,
        false,
        null,
        null);
    mandatoryInputs = inputs;
    Preconditions.checkState(
        (allBitcodeFiles == null) == (importsFile == null),
        "Either both or neither bitcodeFiles and imports files should be null");
    bitcodeFiles = allBitcodeFiles;
    imports = importsFile;
  }

  @Override
  public boolean discoversInputs() {
    return imports != null;
  }

  private NestedSet<Artifact> computeBitcodeInputs(HashSet<PathFragment> inputPaths) {
    NestedSetBuilder<Artifact> bitcodeInputs = NestedSetBuilder.stableOrder();
    for (Artifact inputArtifact : bitcodeFiles.getFiles().toList()) {
      if (inputPaths.contains(inputArtifact.getExecPath())) {
        bitcodeInputs.add(inputArtifact);
      }
    }
    return bitcodeInputs.build();
  }

  @Nullable
  @Override
  public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    List<String> lines;
    try {
      lines = FileSystemUtils.readLinesAsLatin1(actionExecutionContext.getInputPath(imports));
    } catch (IOException e) {
      String message =
          String.format(
              "error reading imports file %s: %s",
              actionExecutionContext.getInputPath(imports), e.getMessage());
      DetailedExitCode code = createDetailedExitCode(message, Code.IMPORTS_READ_IO_EXCEPTION);
      throw new ActionExecutionException(message, e, this, false, code);
    }

    // Build set of files this LTO backend artifact will import from.
    HashSet<PathFragment> importSet = new HashSet<>();
    for (String line : lines) {
      if (line.isEmpty()) {
        continue;
      }
      PathFragment execPath = PathFragment.create(line);
      if (execPath.isAbsolute()) {
        String message =
            String.format(
                "Absolute paths not allowed in imports file %s: %s",
                actionExecutionContext.getInputPath(imports), execPath);
        DetailedExitCode code =
            createDetailedExitCode(message, Code.INVALID_ABSOLUTE_PATH_IN_IMPORTS);
        throw new ActionExecutionException(message, this, false, code);
      }
      importSet.add(execPath);
    }

    // Convert the import set of paths to the set of bitcode file artifacts.
    NestedSet<Artifact> bitcodeInputSet = computeBitcodeInputs(importSet);
    if (bitcodeInputSet.memoizedFlattenAndGetSize() != importSet.size()) {
      Set<PathFragment> missingInputs =
          Sets.difference(
              importSet,
              bitcodeInputSet.toList().stream()
                  .map(Artifact::getExecPath)
                  .collect(Collectors.toSet()));
      String message =
          String.format(
              "error computing inputs from imports file: %s, missing bitcode files (first 10): %s",
              actionExecutionContext.getInputPath(imports),
              // Limit the reported count to protect against a large error message.
              missingInputs.stream()
                  .map(Object::toString)
                  .sorted()
                  .limit(10)
                  .collect(Collectors.joining(", ")));
      DetailedExitCode code = createDetailedExitCode(message, Code.MISSING_BITCODE_FILES);
      throw new ActionExecutionException(message, this, false, code);
    }
    updateInputs(
        NestedSetBuilder.fromNestedSet(bitcodeInputSet).addTransitive(mandatoryInputs).build());
    return bitcodeInputSet;
  }

  @Override
  protected NestedSet<Artifact> getOriginalInputs() {
    return mandatoryInputs;
  }

  private static DetailedExitCode createDetailedExitCode(String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setLtoAction(LtoAction.newBuilder().setCode(detailedCode))
            .build());
  }

  @Override
  public NestedSet<Artifact> getMandatoryInputs() {
    return mandatoryInputs;
  }

  @Override
  public NestedSet<Artifact> getAllowedDerivedInputs() {
    return bitcodeFiles.getFiles();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws InterruptedException {
    fp.addString(GUID);
    try {
      fp.addStrings(getArguments());
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("LtoBackendAction command line expansion cannot fail");
    }
    fp.addString(getMnemonic());
    fp.addPaths(getRunfilesSupplier().getRunfilesDirs());
    ImmutableList<Artifact> runfilesManifests = getRunfilesSupplier().getManifests();
    for (Artifact runfilesManifest : runfilesManifests) {
      fp.addPath(runfilesManifest.getExecPath());
    }
    for (Artifact input : mandatoryInputs.toList()) {
      fp.addPath(input.getExecPath());
    }
    if (imports != null) {
      bitcodeFiles.addToFingerprint(fp);
      fp.addPath(imports.getExecPath());
    }
    env.addTo(fp);
    fp.addStringMap(getExecutionInfo());
  }

  /** Builder class to construct {@link LtoBackendAction} instances. */
  public static class Builder extends SpawnAction.Builder {
    private BitcodeFiles bitcodeFiles;
    private Artifact imports;

    public Builder addImportsInfo(BitcodeFiles allBitcodeFiles, Artifact importsFile) {
      this.bitcodeFiles = allBitcodeFiles;
      this.imports = importsFile;
      return this;
    }

    @Override
    protected SpawnAction createSpawnAction(
        ActionOwner owner,
        NestedSet<Artifact> tools,
        NestedSet<Artifact> inputsAndTools,
        ImmutableList<Artifact> outputs,
        Artifact primaryOutput,
        ResourceSet resourceSet,
        CommandLines commandLines,
        CommandLineLimits commandLineLimits,
        boolean isShellCommand,
        ActionEnvironment env,
        @Nullable BuildConfiguration configuration,
        ImmutableMap<String, String> executionInfo,
        CharSequence progressMessage,
        RunfilesSupplier runfilesSupplier,
        String mnemonic) {
      return new LtoBackendAction(
          inputsAndTools,
          bitcodeFiles,
          imports,
          outputs,
          primaryOutput,
          owner,
          commandLines,
          commandLineLimits,
          isShellCommand,
          env,
          executionInfo,
          progressMessage,
          runfilesSupplier,
          mnemonic);
    }
  }
}
