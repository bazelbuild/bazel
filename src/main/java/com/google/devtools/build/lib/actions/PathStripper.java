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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Pattern;

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
 * code and avoid complicating large swaths of the code base. "Qualifying" actions are determined by
 * {@code --experimental_path_agnostic_action}. As this feature stabilizes, rule definitions can
 * more directly own their integration, particularly declaring which of their actions qualify and
 * where paths appear in their command lines.
 */
public interface PathStripper {
  /**
   * Returns the exec path a running action should use to identify one of its inputs or outputs.
   *
   * <p>If the action should be config-stripped ({@link PathStripper}), removes "k8-fastbuild" from
   * paths like "bazel-out/k8-fastbuild/foo/bar".
   *
   * <p>Else returns the artifact's original exec path.
   */
  String getExecPathString(ActionInput artifact);

  /**
   * If this action strips paths, textually replaces "bazel-out/x86-fastbuild/foo/..." paths in a
   * command line argument with "bazel-out/foo/...". Else returns the input as-is.
   *
   * @param arg a string expected to be part of the action's command line.
   */
  String processCmdArg(String arg);

  /**
   * Adjusts a set of action inputs for possible config path stripping.
   *
   * <p>For artifacts, this is a no-op.
   *
   * <p>{@link com.google.devtools.build.lib.actions.cache.VirtualActionInput}s materialize on
   * demand, so their behavior may be different. {@link ParamFileActionInput}, particularly, stores
   * a command line in a file. So any paths in that command line changes that input's contents.
   * That's the use case this method handles.
   */
  List<ActionInput> processInputs(List<ActionInput> inputs);

  /** Instantiates a {@link PathStripper} that doesn't change paths. */
  static PathStripper noop() {
    return new PathStripper() {
      @Override
      public String getExecPathString(ActionInput artifact) {
        return artifact.getExecPathString();
      }

      @Override
      public String processCmdArg(String arg) {
        return arg;
      }

      @Override
      public List<ActionInput> processInputs(List<ActionInput> inputs) {
        return inputs;
      }
    };
  }

  /**
   * Instantiates a {@link PathStripper} for a spawn action.
   *
   * @param spawn the action to support
   * @param pathAgnosticActions mnemonics of actions that "qualify" for path stripping. Just because
   *     an action type qualifies doesn't mean its paths are stripped. See {@link
   *     #shouldStripPaths}.
   * @param outputRoot root of the output tree ("bazel-out").
   */
  static PathStripper create(
      Spawn spawn, Collection<String> pathAgnosticActions, PathFragment outputRoot) {
    Preconditions.checkState(outputRoot.isSingleSegment());
    Preconditions.checkState(!outputRoot.getPathString().contains("\\"));
    if (!shouldStripPaths(spawn, pathAgnosticActions, outputRoot)) {
      return noop();
    }
    Pattern stripPathsPattern =
        Pattern.compile("\\Q" + outputRoot.getPathString() + "\\E/[\\w-_]+(/|$)");

    return new PathStripper() {
      @Override
      public String getExecPathString(ActionInput artifact) {
        if (!isOutputPath(artifact, outputRoot)) {
          return artifact.getExecPathString();
        }
        PathFragment origExecPath = artifact.getExecPath();
        return outputRoot.getRelative(origExecPath.subFragment(2)).toString();
      }

      @Override
      public String processCmdArg(String arg) {
        // Note that we can't just split the input on a simple delimiter like " ". Output paths
        // can be prefixed by all kinds of characters. Examples:
        //   params files: "@bazel-out/..."
        //   AndroidResourceCompiler: "--resource some_path#bazel-out/x86-opt/some-other-path"
        //   busybox.bzl (lots of Android actions): paths prefixed with ",", ":", and more
        //
        // A deeper and probably more sustainable design would be for all actions to store path
        // references as structured inputs (e.g. an Artifact object instead of a path string).
        // Then lazily instantiate their paths in the executor client. That's roughly what
        // processInputs(), below, models for ParamFileActionInputs. But that involves more API
        // modeling and rule logic cleanup. So we take the path of least resistance for now.
        return stripPathsPattern.matcher(arg).replaceAll(outputRoot.getPathString() + "/");
      }

      @Override
      public ImmutableList<ActionInput> processInputs(List<ActionInput> inputs) {
        return inputs.stream()
            .map(
                input ->
                    input instanceof ParamFileActionInput
                        ? ((ParamFileActionInput) input).withAdjustedArgs(this::processCmdArg)
                        : input)
            .collect(toImmutableList());
      }
    };
  }

  /**
   * Should this action have its paths stripped for execution?
   *
   * <p>Only true for actions that a) qualify, b) don't have distinct paths that would become
   * duplicates if their config prefixes were removed.
   */
  static boolean shouldStripPaths(
      Spawn spawn, Collection<String> pathAgnosticActions, PathFragment outputRoot) {
    String actionMnemonic = spawn.getMnemonic();
    if (!pathAgnosticActions.contains(actionMnemonic)) {
      return false;
    }

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
    for (ActionInput input : spawn.getInputFiles().toList()) {
      if (!isOutputPath(input, outputRoot)) {
        continue;
      }
      // For "bazel-out/k8-fastbuild/foo/bar", get "foo/bar".
      if (!rootRelativePaths.add(input.getExecPath().subFragment(2))) {
        return false;
      }
    }
    return true;
  }

  /** Is this a strippable path? */
  static boolean isOutputPath(ActionInput artifact, PathFragment outputRoot) {
    // We can't simply check for DerivedArtifact. Output paths can also appear, for example, in
    // ParamFileActionInput and ActionInputHelper.BasicActionInput.
    return artifact.getExecPath().startsWith(outputRoot);
  }
}
