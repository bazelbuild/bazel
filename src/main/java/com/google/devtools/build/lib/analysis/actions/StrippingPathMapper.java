// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper.BasicActionInput;
import com.google.devtools.build.lib.actions.ArgChunk;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine.SimpleArgChunk;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.CommandLineItem.ExceptionlessMapFn;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Main logic for experimental config-stripped execution paths:
 * https://github.com/bazelbuild/bazel/issues/6526.
 *
 * <p>The actions executors run look like: {@code tool_pkg/mytool src/source.file
 * bazel-out/x86-opt/pkg/gen.file -o bazel-out/x86-opt/pkg/myout}.
 *
 * <p>The "x86-opt" part is a path's "configuration prefix": information describing the build
 * configuration of the action creating the artifact. This example shows artifacts created with
 * {@code --cpu=x86 --compilation_mode=opt}.
 *
 * <p>Executors cache actions based on their a) command line, b) input and output paths, c) input
 * digests. Configuration prefixes harm caching because even if an action behaves exactly the same
 * for different CPU architectures, {@code <cpu>-opt} guarantees the paths will differ.
 *
 * <p>Config-stripping is an experimental feature that strips the configuration prefix from
 * qualifying actions before running them, thus improving caching. "Qualifying" actions are actions
 * known not to depend on the names of their input and output paths. Non-qualifying actions include
 * manifest generators and compilers that store debug symbol source paths.
 *
 * <p>As an experimental feature, most logic is centralized here to provide easy hooks into executor
 * and action code and avoid complicating large swaths of the code base.
 *
 * <p>Enable this feature by setting {@code --experimental_output_paths=strip}. This activates two
 * effects:
 *
 * <ol>
 *   <li>"Qualifying" actions strip config paths from their command lines. An action qualifies if
 *       its implementation logic uses {@link PathMappers#create} as described in its javadocs and
 *       has its mnemonic listed in {@link PathMappers#SUPPORTED_MNEMONICS}. Such an action must
 *       pass the {@link PathMapper} to all structured command line constructions. If any
 *       unstructured command line arguments refer to artifact paths, custom handling needs to be
 *       added to {@code mapCustomStarlarkArgv} or {@code getMapFn} below.
 *   <li>A supporting executor strips paths from qualifying actions' inputs and outputs before
 *       staging for execution by taking {@link Spawn#getPathMapper()} into account.
 * </ol>
 *
 * <p>So an action is responsible for declaring that it strips paths and adjusting its command line
 * accordingly. The executor is responsible for remapping action inputs and outputs to match.
 *
 * <p>Starlark actions that set {@link ExecutionRequirements#SUPPORTS_PATH_MAPPING} to {@link
 * ExecutionRequirements#PATH_MAPPING_OUTPUT_PACKAGE} also have the package of the target owning the
 * action removed from the paths of their outputs: For an action owned by {@code //pkg:target}, its
 * outputs {@code bazel-out/x86-opt/bin/pkg/foo.txt} and {@code
 * bazel-out/x86-opt/bin/pkg/lib/foo.txt} become {@code bazel-out/out/bin/foo.txt} and {@code
 * bazel-out/out/bin/lib/foo.txt}, while all other paths, including those of inputs in the same
 * package, are only config-stripped. The same applies to the param files of the action, which are
 * named after its primary output. This makes the cache key of an action whose inputs don't depend
 * on the package it is declared in (e.g. one that merges the same set of inputs from other
 * packages) independent of that package, which allows identical actions in different packages to
 * share cache entries and be deduplicated while in flight. Since the mapped path of an output no
 * longer contains the package, such actions must refer to each output by its own path (or a path
 * below an output directory) rather than derive it from a root, a directory or the path of an
 * input, must not persist the paths of their outputs in outputs read by other actions and must not
 * refer to their outputs in files written by other actions (e.g. {@code ctx.actions.write} with
 * {@code Args}). Output paths are never mapped this way in string arguments, which is why this is
 * disabled for actions with special handling of them.
 */
public final class StrippingPathMapper implements PathMapper {
  public static final String GUID = "8eb2ad5a-85d4-435b-858f-5c192e91997d";

  private static final String FIXED_CONFIG_SEGMENT = "cfg";
  // Used in place of the config segment for the outputs of actions that opt into output package
  // path mapping. Since all other output paths are mapped to the fixed config segment, this can't
  // result in collisions.
  private static final String OUTPUT_PACKAGE_SEGMENT = "out";
  private static final String ARCHIVED_TREE_ARTIFACTS_SEGMENT = ":archived_tree_artifacts";

  private final PathFragment outputRoot;
  private final String mnemonic;
  private final boolean isStarlarkAction;
  private final boolean supportsHeuristicPathMapping;
  private final boolean isJavaAction;
  private final ExceptionlessMapFn<Object> structuredArgStripper;
  private final StringStripper argStripper;
  private final ArtifactRoot outputArtifactRoot;
  private final MappedArtifactRoot strippedOutputArtifactRoot;
  // The outputs subject to output package path mapping if it is enabled, otherwise null.
  @Nullable private final PackageMappedOutputs packageMappedOutputs;
  @Nullable private final MappedArtifactRoot packageMappedOutputArtifactRoot;

  private StrippingPathMapper(
      Artifact primaryOutput,
      String mnemonic,
      boolean isStarlarkAction,
      boolean supportsHeuristicPathMapping,
      @Nullable PackageMappedOutputs packageMappedOutputs) {
    // This is expected to always be "(bazel|blaze)-out".
    this.outputRoot = primaryOutput.getExecPath().subFragment(0, 1);
    this.mnemonic = mnemonic;
    this.isStarlarkAction = isStarlarkAction;
    this.supportsHeuristicPathMapping = supportsHeuristicPathMapping;
    this.packageMappedOutputs = packageMappedOutputs;
    this.argStripper = new StringStripper(outputRoot.getPathString());
    this.structuredArgStripper =
        (object, args) -> {
          if (object instanceof String str) {
            args.accept(this.argStripper.strip(str));
          } else {
            args.accept(CommandLineItem.expandToCommandLine(object));
          }
        };
    this.isJavaAction = isJavaAction(mnemonic);
    this.outputArtifactRoot = primaryOutput.getRoot();
    this.strippedOutputArtifactRoot = new MappedArtifactRoot(map(outputArtifactRoot.getExecPath()));
    this.packageMappedOutputArtifactRoot =
        packageMappedOutputs == null
            ? null
            : new MappedArtifactRoot(packageMappedOutputs.mappedDirectory());
  }

  // This kind of special handling should not be extended. It is a hack that works around a
  // limitation of the native implementation of location expansion: The output is just a list of
  // strings, not a structured command line that would allow transparent path mapping.
  // Instead, reimplement location expansion in Starlark and have it return an Args object.
  private static boolean isJavaAction(String mnemonic) {
    return mnemonic.equals("Javac")
        || mnemonic.equals("JavacTurbine")
        || mnemonic.equals("Turbine")
        || mnemonic.equals("JavaResourceJar");
  }

  // Legacy fallback: for specific Android/Java mnemonics, strip only argument values that
  // immediately follow a hardcoded set of known flags (e.g. --mainData, --resources).
  private static boolean hasLegacyStarlarkArgsHandling(String mnemonic) {
    return mnemonic.contains("Android")
        || mnemonic.equals("MergeManifests")
        || mnemonic.equals("StarlarkRClassGenerator")
        || mnemonic.equals("StarlarkAARGenerator")
        || mnemonic.equals("JetifySrcs")
        || mnemonic.equals("Desugar");
  }

  /**
   * Creates a new {@link PathMapper} that strips config prefixes if the particular action instance
   * supports it.
   *
   * @param action the action to potentially strip paths from
   * @param isStarlarkAction whether the action is a Starlark action
   * @param inputMetadataProvider provider to verify colliding inputs have identical digests
   * @return a {@link StrippingPathMapper} if the action supports it, else {@link Optional#empty()}.
   */
  static Optional<PathMapper> tryCreate(
      AbstractAction action,
      boolean isStarlarkAction,
      @Nullable InputMetadataProvider inputMetadataProvider) {
    PathFragment outputRoot = action.getPrimaryOutput().getExecPath().subFragment(0, 1);
    boolean supportsHeuristicPathMapping =
        action
            .getExecutionInfo()
            .containsKey(ExecutionRequirements.SUPPORTS_HEURISTIC_PATH_MAPPING);
    PackageMappedOutputs packageMappedOutputs =
        getPackageMappedOutputs(action, isStarlarkAction, supportsHeuristicPathMapping);
    if (isPathStrippable(
        Iterables.concat(
            action.getInputs().toList(),
            action.getAdditionalArtifactsForPathMapping().toList(),
            action.discoversInputs()
                ? action.getAllowedDerivedInputs().toList()
                : ImmutableSet.of(),
            action.getOutputs()),
        outputRoot,
        packageMappedOutputs,
        inputMetadataProvider)) {
      return Optional.of(
          new StrippingPathMapper(
              action.getPrimaryOutput(),
              action.getMnemonic(),
              isStarlarkAction,
              supportsHeuristicPathMapping,
              packageMappedOutputs));
    }
    return Optional.empty();
  }

  /**
   * Returns the outputs of the action subject to output package path mapping if the action opted
   * into it and supports it, otherwise null.
   */
  @Nullable
  private static PackageMappedOutputs getPackageMappedOutputs(
      AbstractAction action, boolean isStarlarkAction, boolean supportsHeuristicPathMapping) {
    if (!ExecutionRequirements.PATH_MAPPING_OUTPUT_PACKAGE.equals(
        action.getExecutionInfo().get(ExecutionRequirements.SUPPORTS_PATH_MAPPING))) {
      return null;
    }
    // Output package path mapping only applies to structured arguments, so actions that may map
    // string arguments aren't supported. Natively built command lines can also be passed to
    // Starlark actions (e.g. by the C++ rules) and may map string arguments, so such actions must
    // not opt in.
    if (!isStarlarkAction
        || supportsHeuristicPathMapping
        || isJavaAction(action.getMnemonic())
        || hasLegacyStarlarkArgsHandling(action.getMnemonic())) {
      return null;
    }
    Label label = action.getOwner().getLabel();
    if (label == null) {
      return null;
    }
    return PackageMappedOutputs.create(
        label.getPackageIdentifier().getPackagePath(),
        action.getOutputs(),
        action.getPrimaryOutput());
  }

  /**
   * The outputs of an action, together with its param files, whose paths are mapped to the out
   * segment.
   *
   * <p>Executors also map output paths that weren't obtained from the output artifacts (e.g. when
   * uploading outputs to a remote cache), so lookups have to be equality-based. Paths below tree
   * artifact outputs are never mapped individually.
   *
   * @param directory the directory of the package of the target owning the action in the output
   *     root (e.g. {@code bazel-out/k8-fastbuild/bin/pkg})
   * @param mappedDirectory the replacement for {@code directory} (e.g. {@code bazel-out/out/bin})
   * @param paths the exec paths of the outputs below {@code directory}
   * @param paramFilePrefix the common prefix of the exec paths of the param files of the action,
   *     which are named after its primary output (see {@link
   *     com.google.devtools.build.lib.actions.CommandLines#expand})
   */
  private record PackageMappedOutputs(
      PathFragment directory,
      PathFragment mappedDirectory,
      ImmutableSet<PathFragment> paths,
      @Nullable String paramFilePrefix) {

    static PackageMappedOutputs create(
        PathFragment packagePath, Collection<Artifact> outputs, Artifact primaryOutput) {
      // Outputs of Starlark actions are always declared in the package of the target owning the
      // action, but may not be found below its derived artifact path with
      // --experimental_sibling_repository_layout, in which case they are only config-stripped.
      PathFragment rootExecPath = primaryOutput.getRoot().getExecPath();
      PathFragment directory = rootExecPath.getRelative(packagePath);
      PathFragment mappedDirectory =
          rootExecPath
              .subFragment(0, getConfigSegmentIndex(rootExecPath))
              .getRelative(OUTPUT_PACKAGE_SEGMENT)
              .getRelative(rootExecPath.subFragment(getConfigSegmentIndex(rootExecPath) + 1));
      return new PackageMappedOutputs(
          directory,
          mappedDirectory,
          outputs.stream()
              .map(Artifact::getExecPath)
              .filter(path -> path.startsWith(directory) && !path.equals(directory))
              .collect(toImmutableSet()),
          primaryOutput.getExecPath().startsWith(directory)
              ? primaryOutput.getExecPathString() + "-"
              : null);
    }

    boolean contains(PathFragment execPath) {
      return paths.contains(execPath) || isParamFile(execPath.getPathString());
    }

    private boolean isParamFile(String execPath) {
      if (paramFilePrefix == null
          || !execPath.startsWith(paramFilePrefix)
          || !execPath.endsWith(".params")) {
        return false;
      }
      int suffixStart = execPath.length() - ".params".length();
      if (suffixStart <= paramFilePrefix.length()) {
        return false;
      }
      for (int i = paramFilePrefix.length(); i < suffixStart; i++) {
        if (!Character.isDigit(execPath.charAt(i))) {
          return false;
        }
      }
      return true;
    }

    PathFragment map(PathFragment execPath) {
      return mappedDirectory.getRelative(execPath.relativeTo(directory));
    }

    int getExecPathLengthDiff() {
      return directory.getPathString().length() - mappedDirectory.getPathString().length();
    }
  }

  private static PathFragment mapOutputPath(
      PathFragment execPath, @Nullable PackageMappedOutputs packageMappedOutputs) {
    return packageMappedOutputs != null && packageMappedOutputs.contains(execPath)
        ? packageMappedOutputs.map(execPath)
        : strip(execPath);
  }

  @Override
  public String getMappedExecPathString(ActionInput artifact) {
    if (isSupported(artifact)) {
      return mapOutputPath(artifact.getExecPath(), packageMappedOutputs).getPathString();
    } else {
      return artifact.getExecPathString();
    }
  }

  @Override
  public PathFragment map(PathFragment execPath) {
    return isOutputPath(execPath, outputRoot)
        ? mapOutputPath(execPath, packageMappedOutputs)
        : execPath;
  }

  @Override
  public int computeExecPathLengthDiff(DerivedArtifact artifact) {
    PathFragment execPath = artifact.getExecPath();
    int configIndex = getConfigSegmentIndex(execPath);
    if (packageMappedOutputs != null && packageMappedOutputs.contains(execPath)) {
      return packageMappedOutputs.getExecPathLengthDiff();
    }
    return execPath.getSegment(configIndex).length() - FIXED_CONFIG_SEGMENT.length();
  }

  @Override
  public ArgChunk mapCustomStarlarkArgs(ArgChunk chunk) {
    if (!isStarlarkAction) {
      return chunk;
    }

    // Actions opted into heuristic path mapping have all arguments containing output
    // paths stripped, regardless of flag names or argument position.
    if (supportsHeuristicPathMapping) {
      Iterable<String> args = chunk.arguments(this);
      return new SimpleArgChunk(
          () -> Iterators.transform(args.iterator(), this::stripIfOutputPath));
    }

    if (!hasLegacyStarlarkArgsHandling(mnemonic)) {
      return chunk;
    }

    // TODO: b/327187486 - This materializes strings when totalArgLength() is called. Can it
    //  compute the total arg length without creating garbage strings?
    Iterable<String> args = chunk.arguments(this);
    return new SimpleArgChunk(() -> new CustomStarlarkArgsIterator(args.iterator(), argStripper));
  }

  private String stripIfOutputPath(String arg) {
    if (arg.contains(outputRoot.getPathString())) {
      return argStripper.strip(arg);
    }
    return arg;
  }

  @Override
  public ExceptionlessMapFn<Object> getMapFn(@Nullable String previousFlag) {
    if (isJavaAction) {
      if (Objects.equals(previousFlag, "--javacopts")
          || Objects.equals(previousFlag, "--resources")) {
        return structuredArgStripper;
      }
    }
    return MapFn.DEFAULT;
  }

  /**
   * Heuristically maps all path-like strings in the given argument by replacing output root path
   * segments with the fixed configuration segment.
   */
  @Override
  public String mapHeuristically(String arg) {
    return stripIfOutputPath(arg);
  }

  @Override
  public String mapString(String arg) {
    return supportsHeuristicPathMapping ? mapHeuristically(arg) : arg;
  }

  @Override
  public FileRootApi mapRoot(Artifact artifact) {
    if (packageMappedOutputs != null && packageMappedOutputs.contains(artifact.getExecPath())) {
      // Note that the root-relative path of the mapped exec path differs from the artifact's
      // root-relative path as the package has been removed.
      return packageMappedOutputArtifactRoot;
    }
    if (Objects.equals(artifact.getRoot(), outputArtifactRoot)) {
      // The mapped root's path does not depend on the artifact, so we can share an instance.
      return strippedOutputArtifactRoot;
    }
    // Fall back for source roots as well as runfiles tree artifacts, which should be very rare.
    return PathMapper.super.mapRoot(artifact);
  }

  private boolean isSupported(ActionInput artifact) {
    if (artifact instanceof DerivedArtifact) {
      return true;
    }
    if (artifact instanceof BasicActionInput || artifact instanceof VirtualActionInput) {
      return isOutputPath(artifact, outputRoot);
    }
    return false;
  }

  private static final class CustomStarlarkArgsIterator implements Iterator<String> {
    // Add your favorite arg to custom-process here. When Bazel finds one of these in the argument
    // list (an argument name), it strips output path prefixes from the following argument (the
    // argument value).
    private static final ImmutableSet<String> STARLARK_ARGS_TO_STRIP =
        ImmutableSet.of(
            "--mainData",
            "--primaryData",
            "--directData",
            "--data",
            "--resources",
            "--mergeManifests",
            "--library",
            "-i",
            "--input");

    private final Iterator<String> args;
    private final StringStripper argStripper;
    private boolean stripNext = false;

    CustomStarlarkArgsIterator(Iterator<String> args, StringStripper argStripper) {
      this.args = args;
      this.argStripper = argStripper;
    }

    @Override
    public boolean hasNext() {
      return args.hasNext();
    }

    @Override
    public String next() {
      String next = args.next();
      if (stripNext) {
        next = argStripper.strip(next);
      }
      stripNext = STARLARK_ARGS_TO_STRIP.contains(next);
      return next;
    }
  }

  /**
   * Utility class to strip output path configuration prefixes from arbitrary strings.
   *
   * <p>This never applies package path mapping, which is why it isn't enabled for actions that use
   * it.
   */
  private static class StringStripper {
    private final Pattern pattern;
    private final String outputRoot;

    StringStripper(String outputRoot) {
      this.outputRoot = outputRoot;
      this.pattern = stripPathsPattern(outputRoot);
    }

    /**
     * Returns the regex to strip output paths from a string.
     *
     * <p>Supports strings with multiple output paths in arbitrary places. For example
     * "/path/to/compiler bazel-out/x86-fastbuild/foo src/my.src -Dbazel-out/arm-opt/bar".
     *
     * <p>Also supports special archived tree artifact paths containing colons (e.g.,
     * "bazel-out/:archived_tree_artifacts/k8-fastbuild/...").
     *
     * <p>Doesn't strip paths that would be non-existent without config prefixes. For example, these
     * are unchanged: "bazel-out/x86-fastbuild", "bazel-out;foo", "/path/to/compiler bazel-out".
     *
     * @param outputRoot root segment of output paths (e.g. "bazel-out")
     */
    private static Pattern stripPathsPattern(String outputRoot) {
      // Match "bazel-out" followed by a slash, an optional ":archived_tree_artifacts/" prefix group
      // captured in group 1, followed by the configuration segment (any combination of word
      // characters, "_", ".", and "-"), and another slash.
      return Pattern.compile(outputRoot + "/(" + ARCHIVED_TREE_ARTIFACTS_SEGMENT + "/)?[\\w_.-]+/");
    }

    String strip(String str) {
      return pattern.matcher(str).replaceAll(outputRoot + "/$1" + FIXED_CONFIG_SEGMENT + "/");
    }
  }

  /**
   * Is this a strippable path?
   *
   * @param artifact artifact whose path to check
   * @param outputRoot the output tree's execPath-relative root (e.g. "bazel-out")
   */
  private static boolean isOutputPath(ActionInput artifact, PathFragment outputRoot) {
    // We can't simply check for DerivedArtifact. Output paths can also appear, for example, in
    // ParamFileActionInput and ActionInputHelper.BasicActionInput.
    return isOutputPath(artifact.getExecPath(), outputRoot);
  }

  /** Private utility method: Is this a strippable path? */
  private static boolean isOutputPath(PathFragment pathFragment, PathFragment outputRoot) {
    return pathFragment.startsWith(outputRoot);
  }

  /**
   * Returns whether the given execution path belongs to an archived tree artifact.
   *
   * <p>Archived tree artifacts are compressed directory outputs stored in zip format that prepend a
   * virtual directory prefix segment {@value #ARCHIVED_TREE_ARTIFACTS_SEGMENT} immediately after
   * the output root (e.g., {@code bazel-out/:archived_tree_artifacts/k8-fastbuild/...}).
   */
  private static boolean isArchivedTreeArtifactPath(PathFragment execPath) {
    return execPath.segmentCount() > 2
        && execPath.getSegment(1).equals(ARCHIVED_TREE_ARTIFACTS_SEGMENT);
  }

  /**
   * Returns the segment index inside the execution path where the configuration prefix is located.
   *
   * <p>For standard output artifacts, the configuration segment is at index 1 (e.g. {@code
   * bazel-out/k8-fastbuild/bin/...}). For archived tree artifacts, the configuration segment is
   * shifted to index 2 due to the injected virtual prefix directory.
   */
  private static int getConfigSegmentIndex(PathFragment execPath) {
    return isArchivedTreeArtifactPath(execPath) ? 2 : 1;
  }

  /**
   * Is this action safe to strip?
   *
   * <p>This is distinct from whether we <b>should</b> strip it. An action is stripped if a) the
   * action is explicitly supported (see {@link PathMappers#SUPPORTED_MNEMONICS}) and b) it's safe
   * to do that (for example, the action doesn't have two inputs in different configurations that
   * would resolve to the same path if prefixes were removed and have different content).
   *
   * <p>Collisions between inputs from different configurations that map to the same root-relative
   * path are allowed if they have the same file digest.
   *
   * <p>This method checks b).
   *
   * @param packageMappedOutputs the outputs subject to output package path mapping if it is
   *     enabled, otherwise null
   */
  @VisibleForTesting
  static boolean isPathStrippable(
      Iterable<? extends ActionInput> actionInputs,
      PathFragment outputRoot,
      @Nullable PackageMappedOutputs packageMappedOutputs,
      @Nullable InputMetadataProvider inputMetadataProvider) {
    HashMap<PathFragment, ActionInput> mappedPaths = new HashMap<>();
    for (ActionInput input : actionInputs) {
      if (!isOutputPath(input, outputRoot)) {
        continue;
      }
      PathFragment mappedPath = mapOutputPath(input.getExecPath(), packageMappedOutputs);
      ActionInput previous = mappedPaths.computeIfAbsent(mappedPath, k -> input);
      if (!previous.equals(input) && !haveSameDigest(previous, input, inputMetadataProvider)) {
        return false;
      }
    }
    return true;
  }

  /** Returns true if the two inputs have the same file digest. */
  private static boolean haveSameDigest(
      ActionInput a, ActionInput b, @Nullable InputMetadataProvider inputMetadataProvider) {
    if (inputMetadataProvider == null) {
      return false;
    }
    try {
      FileArtifactValue metadataA = inputMetadataProvider.getInputMetadata(a);
      FileArtifactValue metadataB = inputMetadataProvider.getInputMetadata(b);
      if (metadataA == null || metadataB == null) {
        return false;
      }
      byte[] digestA = metadataA.getDigest();
      byte[] digestB = metadataB.getDigest();
      if (digestA == null || digestB == null) {
        return false;
      }
      return Arrays.equals(digestA, digestB);
    } catch (IOException e) {
      return false;
    }
  }

  private static PathFragment strip(PathFragment execPath) {
    int configIndex = getConfigSegmentIndex(execPath);
    return execPath
        .subFragment(0, configIndex)
        // Keep the config segment, but replace it with a fixed string to improve cacheability while
        // still preserving the general segment structure of the execpath.
        .getRelative(FIXED_CONFIG_SEGMENT)
        .getRelative(execPath.subFragment(configIndex + 1));
  }
}
