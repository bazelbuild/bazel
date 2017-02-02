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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Action used by LTOBackendArtifacts to create an LTOBackendAction. Similar to {@link SpawnAction},
 * except that inputs are discovered from the imports file created by the ThinLTO indexing step for
 * each backend artifact.
 *
 * <p>See {@link LTOBackendArtifacts} for a high level description of the ThinLTO build process. The
 * LTO indexing step takes all bitcode .o files and decides which other .o file symbols can be
 * imported/inlined. The additional input files for each backend action are then written to an
 * imports file. Therefore these new inputs must be discovered here by subsetting the imports paths
 * from the set of all bitcode artifacts, before executing the backend action.
 *
 * <p>For more information on ThinLTO see
 * http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html.
 */
public final class LTOBackendAction extends SpawnAction {
  // This can be read/written from multiple threads, and so accesses should be synchronized.
  @GuardedBy("this")
  private boolean inputsKnown;

  private Collection<Artifact> mandatoryInputs;
  private Map<PathFragment, Artifact> bitcodeFiles;
  private Artifact imports;

  private static final String GUID = "72ce1eca-4625-4e24-a0d8-bb91bb8b0e0e";

  public LTOBackendAction(
      Collection<Artifact> inputs,
      Map<PathFragment, Artifact> allBitcodeFiles,
      Artifact importsFile,
      Collection<Artifact> outputs,
      ActionOwner owner,
      CommandLine argv,
      Map<String, String> environment,
      Set<String> clientEnvironmentVariables,
      Map<String, String> executionInfo,
      String progressMessage,
      RunfilesSupplier runfilesSupplier,
      String mnemonic) {
    super(
        owner,
        ImmutableList.<Artifact>of(),
        inputs,
        outputs,
        AbstractAction.DEFAULT_RESOURCE_SET,
        argv,
        ImmutableMap.copyOf(environment),
        ImmutableSet.copyOf(clientEnvironmentVariables),
        ImmutableMap.copyOf(executionInfo),
        progressMessage,
        runfilesSupplier,
        mnemonic,
        false,
        null);

    inputsKnown = false;
    mandatoryInputs = inputs;
    bitcodeFiles = allBitcodeFiles;
    imports = importsFile;
  }

  @Override
  public boolean discoversInputs() {
    return true;
  }

  private Set<Artifact> computeBitcodeInputs(Collection<PathFragment> inputPaths) {
    HashSet<Artifact> bitcodeInputs = new HashSet<>();
    for (PathFragment inputPath : inputPaths) {
      Artifact inputArtifact = bitcodeFiles.get(inputPath);
      if (inputArtifact != null) {
        bitcodeInputs.add(inputArtifact);
      }
    }
    return bitcodeInputs;
  }

  @Nullable
  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    // Build set of files this LTO backend artifact will import from.
    HashSet<PathFragment> importSet = new HashSet<>();
    try {
      for (String line : FileSystemUtils.iterateLinesAsLatin1(imports.getPath())) {
        if (!line.isEmpty()) {
          PathFragment execPath = new PathFragment(line);
          if (execPath.isAbsolute()) {
            throw new ActionExecutionException(
                "Absolute paths not allowed in imports file " + imports.getPath() + ": " + execPath,
                this,
                false);
          }
          importSet.add(new PathFragment(line));
        }
      }
    } catch (IOException e) {
      throw new ActionExecutionException(
          "error iterating imports file " + imports.getPath(), e, this, false);
    }

    // Convert the import set of paths to the set of bitcode file artifacts.
    Set<Artifact> bitcodeInputSet = computeBitcodeInputs(importSet);
    if (bitcodeInputSet.size() != importSet.size()) {
      throw new ActionExecutionException(
          "error computing inputs from imports file " + imports.getPath(), this, false);
    }
    updateInputs(createInputs(bitcodeInputSet, getMandatoryInputs()));
    return bitcodeInputSet;
  }

  @Override
  public synchronized boolean inputsKnown() {
    return inputsKnown;
  }

  @Override
  public Collection<Artifact> getMandatoryInputs() {
    return mandatoryInputs;
  }

  private static Iterable<Artifact> createInputs(
      Set<Artifact> newInputs, Collection<Artifact> curInputs) {
    Set<Artifact> result = new LinkedHashSet<>(newInputs);
    result.addAll(curInputs);
    return result;
  }

  @Override
  public synchronized void updateInputs(Iterable<Artifact> discoveredInputs) {
    setInputs(discoveredInputs);
    inputsKnown = true;
  }

  @Override
  public Iterable<Artifact> getAllowedDerivedInputs() {
    return bitcodeFiles.values();
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    super.execute(actionExecutionContext);

    synchronized (this) {
      inputsKnown = true;
    }
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStrings(getArguments());
    f.addString(getMnemonic());
    f.addPaths(getRunfilesSupplier().getRunfilesDirs());
    ImmutableList<Artifact> runfilesManifests = getRunfilesSupplier().getManifests();
    for (Artifact runfilesManifest : runfilesManifests) {
      f.addPath(runfilesManifest.getExecPath());
    }
    for (Artifact input : getMandatoryInputs()) {
      f.addPath(input.getExecPath());
    }
    for (PathFragment bitcodePath : bitcodeFiles.keySet()) {
      f.addPath(bitcodePath);
    }
    f.addPath(imports.getExecPath());
    f.addStringMap(getEnvironment());
    f.addStringMap(getExecutionInfo());
    return f.hexDigestAndReset();
  }

  /** Builder class to construct {@link LTOBackendAction} instances. */
  public static class Builder extends SpawnAction.Builder {
    private Map<PathFragment, Artifact> bitcodeFiles;
    private Artifact imports;

    public Builder addImportsInfo(
        Map<PathFragment, Artifact> allBitcodeFiles, Artifact importsFile) {
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
        ResourceSet resourceSet,
        CommandLine actualCommandLine,
        ImmutableMap<String, String> env,
        ImmutableSet<String> clientEnvironmentVariables,
        ImmutableMap<String, String> executionInfo,
        String progressMessage,
        RunfilesSupplier runfilesSupplier,
        String mnemonic) {
      return new LTOBackendAction(
          inputsAndTools.toCollection(),
          bitcodeFiles,
          imports,
          outputs,
          owner,
          actualCommandLine,
          env,
          clientEnvironmentVariables,
          executionInfo,
          progressMessage,
          runfilesSupplier,
          mnemonic);
    }
  }
}
