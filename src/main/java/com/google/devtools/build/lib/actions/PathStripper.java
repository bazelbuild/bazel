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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.List;
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
 *       its implementation logic checks {@code --experimental_output_paths=strip}, creates a {@link
 *       Spawn} with {@link Spawn#stripOutputPaths()} == true, and removes config prefixes from its
 *       command line with the help of {@link PathStripper#createForAction(boolean, String,
 *       PathFragment)}. Action logic should also check {@link PathStripper#isPathStrippable}: see
 *       that method's javadoc for why.
 *   <li>A supporting executor strips paths from qualifying actions' inputs and outputs before
 *       staging for execution, with the help of {@link PathStripper#createForExecutor(Spawn,
 *       PathFragment)}.
 * </ol>
 *
 * <p>So an action is responsible for declaring that it strips paths and adjusting its command line
 * accordingly. The executor is responsible for remapping action inputs and outputs to match.
 *
 * <p>A lot of this work is handled generically in {@link CustomCommandLine} and related classes.
 * Simple actions may be able to opt into this behavior with little more than setting {@link
 * com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder#stripOutputPaths(boolean)}.
 * Starlark actions don't yet have API support: specific mnemonics are enabled by {@link
 * com.google.devtools.build.lib.analysis.actions.StarlarkAction.Builder#stripOutputPaths(String,
 * NestedSet, Artifact, BuildConfigurationValue)}.
 */
public final class PathStripper {
  /**
   * Creates a new {@link PathMapper} for action implementation logic to use.
   *
   * @param stripOutputPaths should this action strip config prefixes?
   * @param starlarkMnemonic this action's mnemonic if it's a Starlark action, else null
   * @param outputRoot the root path where outputs are written (e.g. "bazel-out"). Actions that
   *     don't strip outputs can set this to null.
   */
  public static PathMapper createForAction(
      boolean stripOutputPaths,
      @Nullable String starlarkMnemonic,
      @Nullable PathFragment outputRoot) {
    if (stripOutputPaths) {
      Preconditions.checkNotNull(outputRoot);
      Preconditions.checkState(outputRoot.isSingleSegment());
      Preconditions.checkState(!outputRoot.getPathString().contains("\\"));
    }
    return stripOutputPaths ? commandStripper(starlarkMnemonic, outputRoot) : PathMapper.NOOP;
  }

  /**
   * Creates a new {@link PathMapper} for executor implementation logic to use.
   *
   * @param spawn the action to stage. If {@link Spawn#stripOutputPaths()} is true, paths like
   *     "bazel-out/k8-fastbuild/bin/foo" are reduced to "bazel-out/bin/foo". Else they're
   *     unchanged.
   * @param outputRoot the root path where outputs are written (e.g. "bazel-out")
   */
  public static PathMapper createForExecutor(Spawn spawn, PathFragment outputRoot) {
    Preconditions.checkState(outputRoot.isSingleSegment());
    Preconditions.checkState(!outputRoot.getPathString().contains("\\"));
    return spawn.stripOutputPaths() ? actionStripper(outputRoot) : PathMapper.NOOP;
  }

  /** Instantiates a {@link PathMapper} that strips config prefixes from output paths. */
  private static PathMapper actionStripper(PathFragment outputRoot) {
    return execPath -> isOutputPath(execPath, outputRoot) ? PathStripper.strip(execPath) : execPath;
  }

  /** Instantiates a {@link PathMapper} that strips config prefixes from output paths. */
  private static PathMapper commandStripper(
      @Nullable String starlarkMnemonic, PathFragment outputRoot) {
    final StringStripper argStripper =
        starlarkMnemonic != null ? new StringStripper(outputRoot.getPathString()) : null;
    return new PathMapper() {
      @Override
      public String getMappedExecPathString(ActionInput artifact) {
        if (artifact instanceof DerivedArtifact) {
          return PathStripper.strip(artifact);
        } else {
          return artifact.getExecPathString();
        }
      }

      @Override
      public PathFragment map(PathFragment execPath) {
        return PathStripper.isOutputPath(execPath, outputRoot)
            ? PathStripper.strip(execPath)
            : execPath;
      }

      @Override
      public List<String> mapCustomStarlarkArgs(List<String> args) {
        // Add your favorite Starlark mnemonic that needs custom arg processing here.
        if (!starlarkMnemonic.contains("Android")
            && !starlarkMnemonic.equals("MergeManifests")
            && !starlarkMnemonic.equals("StarlarkRClassGenerator")
            && !starlarkMnemonic.equals("StarlarkAARGenerator")) {
          return args;
        }
        // Add your favorite arg to custom-process here. When Bazel finds one of these in the
        // argument list (an argument name), it strips output path prefixes from the following
        // argument (the argument value).
        ImmutableList<String> starlarkArgsToStrip =
            ImmutableList.of(
                "--mainData",
                "--primaryData",
                "--directData",
                "--data",
                "--resources",
                "--mergeeManifests",
                "--library");
        for (int i = 1; i < args.size(); i++) {
          if (starlarkArgsToStrip.contains(args.get(i - 1))) {
            args.set(i, argStripper.strip(args.get(i)));
          }
        }
        return args;
      }
    };
  }

  /**
   * Support for mapping config parts of exec paths in an action's command line as well as when
   * staging its inputs and outputs for execution.
   *
   * <p>Action implementation logic should use this to correctly set an action's command line. The
   * executor should use this to correctly stage an action for execution.
   */
  public interface PathMapper {
    /**
     * Returns the exec path of the input with the config part replaced if necessary.
     *
     * <p>If the action should be config-stripped ({@link PathStripper}), removes "k8-fastbuild"
     * from paths like "bazel-out/k8-fastbuild/foo/bar".
     *
     * <p>Else returns the artifact's original exec path.
     */
    default String getMappedExecPathString(ActionInput artifact) {
      return map(artifact.getExecPath()).getPathString();
    }

    /** Same as {@link #getMappedExecPathString(ActionInput)} but for a {@link PathFragment}. */
    PathFragment map(PathFragment execPath);

    /**
     * We don't yet have a Starlark API for mapping paths in command lines. Simple Starlark calls
     * like {@code args.add(arg_name, file_path} are automatically handled. But calls that involve
     * custom Starlark code require deeper API support that remains a TODO.
     *
     * <p>This method hard-codes support for specific command line entries for specific Starlark
     * actions that we know we want to apply stripping to.
     */
    default List<String> mapCustomStarlarkArgs(List<String> args) {
      return args;
    }

    /** Instantiates a {@link PathMapper} that doesn't change paths. */
    PathMapper NOOP = execPath -> execPath;
  }

  /**
   * Utility class to strip output path configuration prefixes from arbitrary strings.
   *
   * <p>Rules that support path stripping can use this to help their implementation logic.
   */
  public static class StringStripper {
    private final Pattern pattern;
    private final String outputRoot;

    public StringStripper(String outputRoot) {
      this.outputRoot = outputRoot;
      this.pattern = stripPathsPattern(outputRoot);
    }

    public String strip(String str) {
      return pattern.matcher(str).replaceAll(outputRoot + "/");
    }
  }

  /**
   * Returns the regex to strip output paths from a string.
   *
   * <p>Supports strings with multiple output paths in arbitrary places. For example
   * "/path/to/compiler bazel-out/x86-fastbuild/foo src/my.src -Dbazel-out/arm-opt/bar".
   *
   * <p>Doesn't strip paths that would be non-existent without config prefixes. For example, these
   * are unchanged: "bazel-out/x86-fastbuild", "bazel-out;foo", "/path/to/compiler bazel-out".
   *
   * @param outputRoot root segment of output paths (e.g. "bazel-out")
   */
  private static Pattern stripPathsPattern(String outputRoot) {
    // Match "bazel-out" followed by a slash followed by any combination of word characters, "_",
    // and "-", followed by another slash. This would miss substrings like "bazel-out/k8-fastbuild".
    // But those don't represent actual outputs (all outputs would have to have names beneath that
    // path). So we're not trying to replace those.
    return Pattern.compile(outputRoot + "/[\\w_-]+/");
  }

  /**
   * Is this a strippable path?
   *
   * @param artifact artifact whose path to check
   * @param outputRoot - the output tree's execPath-relative root (e.g. "bazel-out")
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
   * Is this action safe to strip?
   *
   * <p>This is distinct from whether we <b>should</b> strip it. An action is stripped if a) the
   * action logic declares it's strippable via {@link Spawn#stripOutputPaths()} and b) it's safe to
   * do that (for example, the action doesn't have two inputs in different configurations that would
   * resolve to the same path if prefixes were removed).
   *
   * <p>This method checks b). Action logic is responsible for considering this to set a) correctly.
   */
  public static boolean isPathStrippable(
      NestedSet<? extends ActionInput> actionInputs, PathFragment outputRoot) {
    // For qualifying action types, check that no inputs or outputs would clash if paths were
    // removed, e.g. "bazel-out/k8-fastbuild/foo" and "bazel-out/host/foo".
    //
    // A more clever algorithm could remap these with custom prefixes - "bazel-out/1/foo" and
    // "bazel-out/2/foo" - if experience shows that would help.
    //
    // Another approach could keep host paths intact (since the "host" path prefix doesn't vary
    // with configurations). While this would help more action instances qualify, it also blocks
    // caching the same action in host and target configurations. This could be mitigated by
    // stripping the host prefix *only* when the entire action is in the host configuration.
    HashSet<PathFragment> rootRelativePaths = new HashSet<>();
    for (ActionInput input : actionInputs.toList()) {
      if (!isOutputPath(input, outputRoot)) {
        continue;
      }
      // For "bazel-out/k8-fastbuild/foo/bar", get "foo/bar".
      if (!rootRelativePaths.add(input.getExecPath().subFragment(2))) {
        // TODO(bazel-team): don't fail on duplicate inputs, i.e. when the same exact exec path
        // (including config prefix) is included twice.
        return false;
      }
    }
    return true;
  }

  /*
   * Private utility method: strips the configuration prefix from an output artifact's exec path.
   */
  static PathFragment strip(PathFragment execPath) {
    return execPath.subFragment(0, 1).getRelative(execPath.subFragment(2));
  }

  /**
   * Private utility method: returns an output artifact's exec path with its configuration prefix
   * stripped.
   */
  static String strip(ActionInput artifact) {
    return strip(artifact.getExecPath()).getPathString();
  }

  private PathStripper() {}
}
