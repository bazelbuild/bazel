// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;

/** Module to handle various Skymeld checks. */
public class SkymeldModule extends BlazeModule {
  @Override
  public void beforeCommand(CommandEnvironment env) {
    env.setMergedAnalysisAndExecution(determineIfMergingAnalysisExecution(env));
  }

  boolean determineIfMergingAnalysisExecution(CommandEnvironment env) {
    String commandName = env.getCommandName();
    PathPackageLocator packageLocator = env.getPackageLocator();
    BuildRequestOptions buildRequestOptions =
        env.getOptions().getOptions(BuildRequestOptions.class);
    boolean effectiveValue = getPlainValueFromFlag(buildRequestOptions);

    // --nobuild means no execution will be carried out, hence it doesn't make sense to interleave
    // analysis and execution in that case and --experimental_merged_skyframe_analysis_execution
    // should be ignored.
    if (effectiveValue && !buildRequestOptions.performExecutionPhase) {
      // Aquery and Cquery implicitly set --nobuild, so there's no need to have a warning here: it
      // makes no different from the users' perspective.
      if (!(commandName.equals("aquery") || commandName.equals("cquery"))) {
        env.getReporter()
            .handle(
                Event.warn(
                    "--experimental_merged_skyframe_analysis_execution is incompatible with"
                        + " --nobuild and will be ignored."));
      }
      effectiveValue = false;
    }
    // TODO(b/245922903): Make --explain compatible with Skymeld.
    if (effectiveValue && buildRequestOptions.explanationPath != null) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_merged_skyframe_analysis_execution is incompatible with --explain"
                      + " and will be ignored."));
      effectiveValue = false;
    }

    boolean havingMultiPackagePath =
        packageLocator != null && packageLocator.getPathEntries().size() > 1;
    // TODO(b/246324830): Skymeld and multi-package_path are incompatible.
    if (effectiveValue && havingMultiPackagePath) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_merged_skyframe_analysis_execution is "
                      + "incompatible with multiple --package_path ("
                      + packageLocator.getPathEntries()
                      + ") and its value will be ignored."));
      effectiveValue = false;
    }

    if (effectiveValue
        && env.getOptions().getOptions(AnalysisOptions.class) != null
        && env.getOptions().getOptions(AnalysisOptions.class).cpuHeavySkyKeysThreadPoolSize <= 0) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_merged_skyframe_analysis_execution is incompatible with a"
                      + " non-positive --experimental_skyframe_cpu_heavy_skykeys_thread_pool_size"
                      + " and its value will be ignored."));
      effectiveValue = false;
    }

    if (effectiveValue
        && (buildRequestOptions.aqueryDumpAfterBuildFormat != null
            || buildRequestOptions.aqueryDumpAfterBuildOutputFile != null)) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_merged_skyframe_analysis_execution is incompatible with"
                      + " generating an aquery dump after builds and its value will be ignored."));
      effectiveValue = false;
    }

    // TODO(b/245873370) --check_licenses is going away.
    if (effectiveValue
        && env.getOptions().getOptions(CoreOptions.class) != null
        && env.getOptions().getOptions(CoreOptions.class).checkLicenses) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_merged_skyframe_analysis_execution is incompatible with"
                      + " --check_licenses and its value will be ignored."));
      effectiveValue = false;
    }

    return effectiveValue;
  }

  static boolean getPlainValueFromFlag(BuildRequestOptions buildRequestOptions) {
    return buildRequestOptions != null
        && buildRequestOptions.mergedSkyframeAnalysisExecutionDoNotUseDirectly;
  }
}
