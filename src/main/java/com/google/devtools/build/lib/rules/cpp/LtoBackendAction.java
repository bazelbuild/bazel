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

import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.LtoAction;
import com.google.devtools.build.lib.server.FailureDetails.LtoAction.Code;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Action used by LtoBackendArtifacts to create an LtoBackendAction. Similar to {@link SpawnAction},
 * except that inputs are discovered from the imports file created by the ThinLTO indexing step for
 * each backend artifact.
 *
 * <p>See {@link LtoBackendArtifacts} for a high level description of the ThinLTO build process. The
 * LTO indexing step takes all bitcode .o files and decides which other .o file symbols can be
 * imported/inlined. The additional input files for each backend action are then written to an
 * imports file. Therefore, these new inputs must be discovered here by subsetting the imports paths
 * from the set of all bitcode artifacts, before executing the backend action.
 *
 * <p>For more information on ThinLTO see
 * http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html.
 */
@AutoCodec
public final class LtoBackendAction extends SpawnAction {
  private static final String GUID = "72ce1eca-4625-4e24-a0d8-bb91bb8b0e0e";
  private static final String MNEMONIC = "CcLtoBackendCompile";

  private final NestedSet<Artifact> mandatoryInputs;
  private final BitcodeFiles bitcodeFiles;
  private final Artifact imports;
  private boolean inputsDiscovered = false;

  public LtoBackendAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      @Nullable BitcodeFiles allBitcodeFiles,
      @Nullable Artifact importsFile,
      ImmutableSet<Artifact> outputs,
      CommandLines argv,
      ActionEnvironment env,
      ImmutableMap<String, String> executionInfo) {
    super(
        owner,
        /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        inputs,
        outputs,
        AbstractAction.DEFAULT_RESOURCE_SET,
        argv,
        env,
        executionInfo,
        "LTO Backend Compile %{output}",
        MNEMONIC);
    mandatoryInputs = inputs;
    Preconditions.checkState(
        (allBitcodeFiles == null) == (importsFile == null),
        "Either both or neither bitcodeFiles and imports files should be null");
    bitcodeFiles = allBitcodeFiles;
    imports = importsFile;
  }

  /** Constructor for serialization. */
  @VisibleForSerialization
  @AutoCodec.Instantiator
  LtoBackendAction(
      ActionOwner owner,
      NestedSet<Artifact> mandatoryInputs,
      Object rawOutputs,
      CommandLines commandLines,
      SpawnAttributes attributes,
      @Nullable BitcodeFiles bitcodeFiles,
      Artifact imports) {
    super(
        owner,
        /* tools= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        mandatoryInputs,
        rawOutputs,
        commandLines,
        "LTO Backend Compile %{output}",
        attributes);
    this.mandatoryInputs = mandatoryInputs;
    this.bitcodeFiles = bitcodeFiles;
    this.imports = imports;
  }

  private boolean isPathMappingEnabled() {
    if (imports == null || getOutputs().size() != 1) {
      return false;
    }
    ImmutableMap<String, String> executionInfo = getExecutionInfo();
    if (executionInfo.containsKey(ExecutionRequirements.LOCAL)
        || (executionInfo.containsKey(ExecutionRequirements.NO_SANDBOX)
            && executionInfo.containsKey(ExecutionRequirements.NO_REMOTE))) {
      return false;
    }
    return executionInfo.containsKey(ExecutionRequirements.SUPPORTS_PATH_MAPPING);
  }

  @VisibleForTesting
  PathMapper getPathMapper() {
    if (!isPathMappingEnabled()) {
      return PathMapper.NOOP;
    }
    return LtoBackendPathMapper.tryCreate(imports, getPrimaryOutput());
  }

  @Override
  protected PathMapper getPathMapper(@Nullable InputMetadataProvider inputMetadataProvider) {
    return getPathMapper();
  }

  @Override
  protected OutputPathsMode getOutputPathsMode() {
    return OutputPathsMode.OFF;
  }

  @Override
  public boolean discoversInputs() {
    return imports != null;
  }

  @Override
  protected boolean inputsDiscovered() {
    return inputsDiscovered;
  }

  @Override
  protected void setInputsDiscovered(boolean inputsDiscovered) {
    this.inputsDiscovered = inputsDiscovered;
  }

  /**
   * Given a map of path to artifact, and a path, returns the artifact whose key is in the map, or
   * if none, an artifact whose key matches a prefix of the path. Assumes that artifacts whose paths
   * are directories are tree artifacts. Assumes that no artifact key is a sub directory of another
   * artifact key. For example, "path/file1" may return the artifact whose path is "path/file1" or
   * whose path is "path/". Returns empty if there are no matches.
   */
  private Optional<Artifact> getArtifactOrTreeArtifact(
      PathFragment path, Map<PathFragment, Artifact> pathToArtifact) {
    PathFragment currentPath = path;
    while (!currentPath.isEmpty()) {
      if (pathToArtifact.containsKey(currentPath)) {
        return Optional.of(pathToArtifact.get(currentPath));
      } else {
        currentPath = currentPath.getParentDirectory();
      }
    }
    return Optional.empty();
  }

  /**
   * Throws an error if any of the input paths is not in the bitcodeFiles or in a subdirecorty of a
   * file in bitcodeFiles
   */
  private NestedSet<Artifact> computeBitcodeInputs(
      HashSet<PathFragment> inputPaths, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    NestedSetBuilder<Artifact> bitcodeInputs = NestedSetBuilder.stableOrder();
    Map<PathFragment, Artifact> execPathToArtifact = bitcodeFiles.getFilesArtifactPathMap();
    Set<PathFragment> missingInputs = new HashSet<>();
    for (PathFragment inputPath : inputPaths) {
      Optional<Artifact> maybeArtifact = getArtifactOrTreeArtifact(inputPath, execPathToArtifact);
      if (maybeArtifact.isPresent()) {
        bitcodeInputs.add(maybeArtifact.get());
      } else {
        // One of the inputs is not present. We add it to missingInputs and will fail.
        missingInputs.add(inputPath);
      }
    }
    if (!missingInputs.isEmpty()) {
      String message =
          String.format(
              "error computing inputs from imports file: %s, missing bitcode files (first 10): %s",
              actionExecutionContext.getInputPath(imports),
              // Limit the reported count to protect against a large error message.
              missingInputs.stream()
                  .map(Object::toString)
                  .sorted()
                  .limit(10)
                  .collect(joining(", ")));
      DetailedExitCode code = createDetailedExitCode(message, Code.MISSING_BITCODE_FILES);
      throw new ActionExecutionException(message, this, false, code);
    }
    return bitcodeInputs.build();
  }

  private NestedSet<Artifact> computeImports(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Path importsFilePath = actionExecutionContext.getInputPath(imports);
    ImmutableList<String> lines;
    try {
      lines = FileSystemUtils.readLinesAsLatin1(importsFilePath);
    } catch (IOException e) {
      String message =
          String.format(
              "error reading imports file %s: %s",
              actionExecutionContext.getInputPath(imports), e.getMessage());
      DetailedExitCode code;
      if (e instanceof CacheNotFoundException
          || BulkTransferException.allCausedByCacheNotFoundException(e)) {
        code =
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage(message)
                    .setSpawn(Spawn.newBuilder().setCode(Spawn.Code.REMOTE_CACHE_EVICTED))
                    .build());
      } else {
        code = createDetailedExitCode(message, Code.IMPORTS_READ_IO_EXCEPTION);
      }
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
    // Throws an error if there is any path in the importset that is not pat of any artifact
    return computeBitcodeInputs(importSet, actionExecutionContext);
  }

  @Nullable
  @Override
  public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    NestedSet<Artifact> bitcodeInputSet = computeImports(actionExecutionContext);
    updateInputs(
        NestedSetBuilder.fromNestedSet(bitcodeInputSet).addTransitive(mandatoryInputs).build());
    return bitcodeInputSet;
  }

  @Override
  public NestedSet<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) throws ActionExecutionException {
    return computeImports(actionExecutionContext);
  }

  @Override
  public NestedSet<Artifact> getOriginalInputs() {
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
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp)
      throws InterruptedException {
    fp.addString(GUID);
    PathMapper pathMapper = getPathMapper();
    // NOMUTANTS(LCR): Version salt; keeps unmapped action keys unchanged during rollout.
    if (!pathMapper.isNoop()) {
      fp.addString(LtoBackendPathMapper.GUID);
    }
    try {
      fp.addStrings(getCommandLines().allArguments(pathMapper));
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("LtoBackendAction command line expansion cannot fail", e);
    }
    fp.addString(getMnemonic());
    for (Artifact input : mandatoryInputs.toList()) {
      fp.addPath(pathMapper.map(input.getExecPath()));
    }
    if (imports != null) {
      bitcodeFiles.addToFingerprint(fp);
      fp.addPath(pathMapper.map(imports.getExecPath()));
    }
    getEnvironment().addTo(OutputPathsMode.OFF, fp);
    fp.addStringMap(getExecutionInfo());
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    message.append(getProgressMessage()).append('\n');
    message.append("  GUID: ").append(GUID).append('\n');
    PathMapper pathMapper = getPathMapper();
    if (!pathMapper.isNoop()) {
      message.append("  PathMapper GUID: ").append(LtoBackendPathMapper.GUID).append('\n');
    }
    message.append("  Mnemonic: ").append(getMnemonic()).append('\n');
    for (Map.Entry<String, String> entry : getEnvironment().getFixedEnv().entrySet()) {
      message
          .append("  Environment variable: ")
          .append(ShellEscaper.escapeString(entry.getKey()))
          .append('=')
          .append(ShellEscaper.escapeString(entry.getValue()))
          .append('\n');
    }
    for (String var : getEnvironment().getInheritedEnv()) {
      message
          .append("  Environment variables taken from the client environment: ")
          .append(ShellEscaper.escapeString(var))
          .append('\n');
    }
    for (String var : getEnvironment().getUnsetEnv()) {
      message
          .append("  Unset environment variable: ")
          .append(ShellEscaper.escapeString(var))
          .append('\n');
    }
    for (Map.Entry<String, String> entry : getExecutionInfo().entrySet()) {
      message
          .append("  Execution info: ")
          .append(entry.getKey())
          .append('=')
          .append(entry.getValue())
          .append('\n');
    }
    try {
      for (String argument : ShellEscaper.escapeAll(getCommandLines().allArguments(pathMapper))) {
        message.append("  Argument: ").append(argument).append('\n');
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      message.append("Interrupted while expanding command line\n");
    } catch (CommandLineExpansionException e) {
      message.append("Could not expand command line: ").append(e).append('\n');
    }
    for (Artifact input : mandatoryInputs.toList()) {
      message
          .append("  Mandatory input: ")
          .append(pathMapper.getMappedExecPathString(input))
          .append('\n');
    }
    if (imports != null) {
      message
          .append("  Imports: ")
          .append(pathMapper.getMappedExecPathString(imports))
          .append('\n');
      Fingerprint fp = new Fingerprint();
      bitcodeFiles.addToFingerprint(fp);
      message
          .append("  Bitcode files digest: ")
          .append(BaseEncoding.base16().lowerCase().encode(fp.digestAndReset()))
          .append('\n');
    }
    return message.toString();
  }

  @VisibleForTesting
  static LtoBackendAction create(
      ActionOwner actionOwner,
      @Nullable BuildConfigurationValue configuration,
      NestedSet<Artifact> inputs,
      @Nullable BitcodeFiles allBitcodeFiles,
      @Nullable Artifact importsFile,
      ImmutableSet<Artifact> outputs,
      CommandLines argv,
      ActionEnvironment env) {
    return create(
        actionOwner,
        configuration,
        inputs,
        allBitcodeFiles,
        importsFile,
        outputs,
        argv,
        env,
        /* enablePathMapping= */ false);
  }

  static LtoBackendAction create(
      ActionOwner actionOwner,
      @Nullable BuildConfigurationValue configuration,
      NestedSet<Artifact> inputs,
      @Nullable BitcodeFiles allBitcodeFiles,
      @Nullable Artifact importsFile,
      ImmutableSet<Artifact> outputs,
      CommandLines argv,
      ActionEnvironment env,
      boolean enablePathMapping) {
    BuildConfigurationValue effectiveConfig =
        configuration != null
            ? configuration
            : (actionOwner.getBuildConfigurationInfo() instanceof BuildConfigurationValue cfg
                ? cfg
                : null);
    ImmutableMap<String, String> baseExecutionInfo =
        enablePathMapping && importsFile != null && outputs.size() == 1
            ? ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, "")
            : ImmutableMap.of();
    return new LtoBackendAction(
        actionOwner,
        inputs,
        allBitcodeFiles,
        importsFile,
        outputs,
        argv,
        env,
        effectiveConfig == null
            ? baseExecutionInfo
            : effectiveConfig.modifiedExecutionInfo(baseExecutionInfo, MNEMONIC));
  }

  /**
   * A {@link PathMapper} for {@link LtoBackendAction} that strips the target-specific LTO directory
   * prefix ({@code <pkg>/<target>.lto/} and {@code <pkg>/<target>.lto-obj/}) on this action's LTO
   * index, imports, and output object paths while preserving the configuration prefix ({@code
   * <cfg>}).
   *
   * <p>Non-empty ThinLTO index files ({@code .thinlto.bc}) embed the unmapped
   * configuration-prefixed paths of imported bitcode modules (for example, {@code
   * bazel-out/k8-opt/bin/pkg/_objs/lib/foo.pic.o}). Stripping {@code k8-opt} to {@code cfg} via
   * {@link com.google.devtools.build.lib.analysis.actions.StrippingPathMapper} would cause Clang's
   * ThinLTO backend module loader to fail to locate imported bitcode files. By rewriting only this
   * action's paired LTO index, imports, and output object paths ({@code <bin>/<pkg>/<target>.lto/}
   * to {@code <bin>/_lto/} and {@code <bin>/<pkg>/<target>.lto-obj/} to {@code <bin>/_lto-obj/})
   * while leaving all bitcode inputs untouched, identical ThinLTO backend compilations (including
   * skipped {@code --start-lib} modules and shared library modules with identical import summaries)
   * share remote cache entries across targets in the same configuration.
   */
  @VisibleForTesting
  static final class LtoBackendPathMapper implements PathMapper {
    static final String GUID = "4f8a1e72-9c3b-4d5e-b6a1-082736455463";
    private static final String CANONICAL_LTO_SEGMENT = "_lto";
    private static final String CANONICAL_LTO_OBJ_SEGMENT = "_lto-obj";
    private static final String IMPORTS_EXTENSION = ".imports";
    private static final String THINLTO_INDEX_EXTENSION = ".thinlto.bc";

    private record MappedPaths(
        PathFragment objPath,
        PathFragment mappedObjPath,
        PathFragment importsPath,
        PathFragment mappedImportsPath,
        PathFragment indexPath,
        PathFragment mappedIndexPath) {}

    private final MappedPaths mappedPaths;

    private LtoBackendPathMapper(MappedPaths mappedPaths) {
      this.mappedPaths = mappedPaths;
    }

    static PathMapper tryCreate(Artifact imports, Artifact primaryOutput) {
      PathFragment importsPath = imports.getExecPath();
      PathFragment objPath = primaryOutput.getExecPath();
      PathFragment rootExecPath = primaryOutput.getRoot().getExecPath();
      if (!imports.getRoot().getExecPath().equals(rootExecPath)) {
        return PathMapper.NOOP;
      }
      int startIdx = rootExecPath.segmentCount();
      // NOMUTANTS(AOR): Prefix directory requires at least one trailing child segment.
      int maxIdx = Math.min(importsPath.segmentCount(), objPath.segmentCount()) - 1;
      MappedPaths mappedPaths =
          findMappedPaths(
              importsPath, objPath, rootExecPath, startIdx, maxIdx, /* requireLtoObj= */ true);
      if (mappedPaths == null) {
        mappedPaths =
            findMappedPaths(
                importsPath, objPath, rootExecPath, startIdx, maxIdx, /* requireLtoObj= */ false);
      }
      if (mappedPaths == null) {
        return PathMapper.NOOP;
      }
      return new LtoBackendPathMapper(mappedPaths);
    }

    @Nullable
    private static MappedPaths findMappedPaths(
        PathFragment importsPath,
        PathFragment objPath,
        PathFragment rootExecPath,
        int startIdx,
        int maxIdx,
        boolean requireLtoObj) {
      MappedPaths fallback = null;
      for (int i = startIdx; i < maxIdx; i++) {
        // Segments [0, startIdx) match because imports and primaryOutput share rootExecPath.
        if (i > startIdx && !importsPath.getSegment(i - 1).equals(objPath.getSegment(i - 1))) {
          return fallback;
        }
        String importsSeg = importsPath.getSegment(i);
        if (!importsSeg.endsWith(".lto") || importsSeg.length() <= ".lto".length()) {
          continue;
        }
        String objSeg = objPath.getSegment(i);
        boolean matchesObj =
            requireLtoObj
                ? objSeg.length() == importsSeg.length() + 4
                    && objSeg.endsWith("-obj")
                    && objSeg.regionMatches(0, importsSeg, 0, importsSeg.length())
                : objSeg.equals(importsSeg);
        if (!matchesObj) {
          continue;
        }
        PathFragment objRel = objPath.subFragment(i + 1, objPath.segmentCount());
        PathFragment importsRel = importsPath.subFragment(i + 1, importsPath.segmentCount());
        String objRelStr = objRel.getPathString();
        String importsRelStr = importsRel.getPathString();
        boolean hasImportsExt = importsRelStr.equals(objRelStr + IMPORTS_EXTENSION);
        if (!hasImportsExt && !importsRelStr.equals(objRelStr)) {
          continue;
        }
        PathFragment indexPrefix = importsPath.subFragment(0, i + 1);
        PathFragment mappedIndexPrefix = rootExecPath.getChild(CANONICAL_LTO_SEGMENT);
        PathFragment mappedObjPrefix =
            rootExecPath.getChild(
                requireLtoObj ? CANONICAL_LTO_OBJ_SEGMENT : CANONICAL_LTO_SEGMENT);
        PathFragment indexRel =
            hasImportsExt
                ? objRel.replaceName(objRel.getBaseName() + THINLTO_INDEX_EXTENSION)
                : objRel;
        MappedPaths candidate =
            new MappedPaths(
                objPath,
                mappedObjPrefix.getRelative(objRel),
                importsPath,
                mappedIndexPrefix.getRelative(importsRel),
                indexPrefix.getRelative(indexRel),
                mappedIndexPrefix.getRelative(indexRel));
        if (objRel.startsWith(rootExecPath)) {
          return candidate;
        }
        fallback = candidate;
      }
      return fallback;
    }

    @Override
    public PathFragment map(PathFragment execPath) {
      if (execPath.equals(mappedPaths.objPath())) {
        return mappedPaths.mappedObjPath();
      }
      if (execPath.equals(mappedPaths.indexPath())) {
        return mappedPaths.mappedIndexPath();
      }
      if (execPath.equals(mappedPaths.importsPath())) {
        return mappedPaths.mappedImportsPath();
      }
      return execPath;
    }

    @Override
    public Object cacheKey() {
      return mappedPaths;
    }
  }
}
