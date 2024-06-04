// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.skyframe.ProjectFilesLookupValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.config.FlagSetValue;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * Container for reading project metadata.
 *
 * <p>A "project" is a set of related packages that support a common piece of software. For example,
 * "bazel" is a project that includes packages {@code src/main/java/com/google/devtools/build/lib},
 * {@code src/main/java/com/google/devtools/build/lib/analysis}, {@code src/test/cpp}, and more.
 *
 * <p>"Project metadata" is any useful information that might be associated with a project. Possible
 * consumers include <a
 * href="https://github.com/bazelbuild/bazel/commit/693215317a6732085731809266f63ff0e7fc31a5">
 * Skyfocus</a>> and project-sanctioned build flags (i.e. "these are the correct flags to use with
 * this project").
 *
 * <p>Projects are defined in .scl files that are checked into source control with BUILD files and
 * code. scl stands for "Starlark configuration language". This is a limited subset of Starlark
 * intended to model generic configuration without requiring Bazel to parse it (similar to JSON).
 *
 * <p>This is not the same as {@link com.google.devtools.build.lib.runtime.ProjectFile}. That's an
 * older implementation of the same idea that was built before .scl and .bzl existed. The code here
 * is a rejuvenation of these ideas with more modern APIs.
 */
// TODO: b/324127050 - Make the co-existence of this and ProjectFile less confusing. ProjectFile is
//   an outdated API that should be removed.
public final class Project {
  private Project() {}

  /** Thrown when project data can't be read. */
  public static class ProjectParseException extends Exception {
    ProjectParseException(String msg, Throwable cause) {
      super(msg, cause);
    }
  }

  /**
   * Finds and returns the project files for a set of build targets.
   *
   * <p>This walks up each target's package path looking for {@link
   * com.google.devtools.build.lib.skyframe.ProjectFilesLookupFunction#PROJECT_FILE_NAME} files. For
   * example, for {@code //foo/bar/baz:mytarget}, this might look in {@code foo/bar/baz}, {@code
   * foo/bar}, and {@code foo} ("might" because it skips directories that don't have BUILD files -
   * those directories aren't packages).
   *
   * @return a map from each target to its set of project files, ordered by reverse package depth.
   *     So a project file in {@code foo/bar} appears before a project file in {@code foo}.
   */
  // TODO: b/324127050 - Document resolution semantics when this is less experimental.
  public static ImmutableMultimap<Label, Label> findProjectFiles(
      Collection<Label> targets,
      SkyframeExecutor skyframeExecutor,
      ExtendedEventHandler eventHandler)
      throws ProjectParseException {
    // TODO: b/324127050 - Support other repos.
    ImmutableMap<Label, ProjectFilesLookupValue.Key> targetsToSkyKeys =
        targets.stream()
            .collect(
                toImmutableMap(
                    target -> target,
                    target -> ProjectFilesLookupValue.key(target.getPackageIdentifier())));
    var evalResult =
        skyframeExecutor.evaluateSkyKeys(
            eventHandler, targetsToSkyKeys.values(), /* keepGoing= */ false);
      if (evalResult.hasError()) {
        throw new ProjectParseException(
            "Error finding project files", evalResult.getError().getException());
      }

    ImmutableMultimap.Builder<Label, Label> ans = ImmutableMultimap.builder();
    for (var entry : targetsToSkyKeys.entrySet()) {
      ProjectFilesLookupValue containingProjects =
          (ProjectFilesLookupValue) evalResult.get(entry.getValue());
      ans.putAll(entry.getKey(), containingProjects.getProjectFiles());
    }
    return ans.build();
  }

  /**
   * applies {@link CoreOptions.sclConfig} to the top-level {@link BuildOptions}
   *
   * <p>given an existing PROJECT.scl file and an {@link CoreOptions.sclConfig}, the method creates
   * a {@link SkyKey} containing the {@link PathFragment} of the scl file and the config name which
   * is evaluated by the {@link FlagSetFunction}
   *
   * @return {@link FlagSetValue} which has the effective top-level {@link BuildOptions} after
   *     project file resolution.
   */
  public static FlagSetValue modifyBuildOptionsWithFlagSets(
      Label projectFile,
      BuildOptions targetOptions,
      ExtendedEventHandler eventHandler,
      SkyframeExecutor skyframeExecutor)
      throws InvalidConfigurationException {

    FlagSetValue.Key flagSetKey =
        FlagSetValue.Key.create(
            projectFile, targetOptions.get(CoreOptions.class).sclConfig, targetOptions);

    EvaluationResult<SkyValue> result =
        skyframeExecutor.evaluateSkyKeys(
            eventHandler, ImmutableList.of(flagSetKey), /* keepGoing= */ false);
    if (result.hasError()) {
      throw new InvalidConfigurationException("Cannot parse options", Code.INVALID_BUILD_OPTIONS);
    }
    return (FlagSetValue) result.get(flagSetKey);
  }
}
