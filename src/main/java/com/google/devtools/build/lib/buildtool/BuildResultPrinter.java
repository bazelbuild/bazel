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

package com.google.devtools.build.lib.buildtool;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.util.io.OutErr;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Handles --show_result and --experimental_show_artifacts.
 */
class BuildResultPrinter {
  private final CommandEnvironment env;

  BuildResultPrinter(CommandEnvironment env) {
    this.env = env;
  }

  /**
   * Shows the result of the build. Information includes the list of up-to-date
   * and failed targets and list of output artifacts for successful targets
   *
   * <p>This corresponds to the --show_result flag.
   */
  public void showBuildResult(
      BuildRequest request,
      BuildResult result,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<AspectValue> aspects) {
    // NOTE: be careful what you print!  We don't want to create a consistency
    // problem where the summary message and the exit code disagree.  The logic
    // here is already complex.

    Collection<ConfiguredTarget> targetsToPrint = filterTargetsToPrint(configuredTargets);
    Collection<AspectValue> aspectsToPrint = filterAspectsToPrint(aspects);

    // Filter the targets we care about into two buckets:
    Collection<ConfiguredTarget> succeeded = new ArrayList<>();
    Collection<ConfiguredTarget> failed = new ArrayList<>();
    for (ConfiguredTarget target : targetsToPrint) {
      Collection<ConfiguredTarget> successfulTargets = result.getSuccessfulTargets();
      (successfulTargets.contains(target) ? succeeded : failed).add(target);
    }

    // Suppress summary if --show_result value is exceeded:
    if (succeeded.size() + failed.size() + aspectsToPrint.size()
        > request.getBuildOptions().maxResultTargets) {
      return;
    }

    OutErr outErr = request.getOutErr();

    TopLevelArtifactContext context = request.getTopLevelArtifactContext();
    for (ConfiguredTarget target : succeeded) {
      Label label = target.getLabel();
      // For up-to-date targets report generated artifacts, but only
      // if they have associated action and not middleman artifacts.
      boolean headerFlag = true;
      for (Artifact artifact :
          TopLevelArtifactHelper.getAllArtifactsToBuild(target, context).getImportantArtifacts()) {
        if (shouldPrint(artifact)) {
          if (headerFlag) {
            outErr.printErr("Target " + label + " up-to-date:\n");
            headerFlag = false;
          }
          outErr.printErrLn(formatArtifactForShowResults(artifact, request));
        }
      }
      if (headerFlag) {
        outErr.printErr("Target " + label + " up-to-date (nothing to build)\n");
      }
    }

    for (AspectValue aspect : aspectsToPrint) {
      Label label = aspect.getLabel();
      String aspectName = aspect.getConfiguredAspect().getName();
      boolean headerFlag = true;
      NestedSet<Artifact> importantArtifacts =
          TopLevelArtifactHelper.getAllArtifactsToBuild(aspect, context).getImportantArtifacts();
      for (Artifact importantArtifact : importantArtifacts) {
        if (headerFlag) {
          outErr.printErr("Aspect " + aspectName + " of " + label + " up-to-date:\n");
          headerFlag = false;
        }
        if (shouldPrint(importantArtifact)) {
          outErr.printErrLn(formatArtifactForShowResults(importantArtifact, request));
        }
      }
      if (headerFlag) {
        outErr.printErr(
            "Aspect " + aspectName + " of " + label + " up-to-date (nothing to build)\n");
      }
    }

    for (ConfiguredTarget target : failed) {
      outErr.printErr("Target " + target.getLabel() + " failed to build\n");

      // For failed compilation, it is still useful to examine temp artifacts,
      // (ie, preprocessed and assembler files).
      OutputGroupProvider topLevelProvider =
          target.getProvider(OutputGroupProvider.class);
      String productName = env.getRuntime().getProductName();
      if (topLevelProvider != null) {
        for (Artifact temp : topLevelProvider.getOutputGroup(OutputGroupProvider.TEMP_FILES)) {
          if (temp.getPath().exists()) {
            outErr.printErrLn("  See temp at "
                + OutputDirectoryLinksUtils.getPrettyPath(temp.getPath(),
                    env.getWorkspaceName(),
                    env.getWorkspace(),
                    request.getBuildOptions().getSymlinkPrefix(productName)));
          }
        }
      }
    }
    if (!failed.isEmpty() && !request.getOptions(ExecutionOptions.class).verboseFailures) {
      outErr.printErr("Use --verbose_failures to see the command lines of failed build steps.\n");
    }
  }

  private boolean shouldPrint(Artifact artifact) {
    return !artifact.isSourceArtifact() && !artifact.isMiddlemanArtifact();
  }

  private String formatArtifactForShowResults(Artifact artifact, BuildRequest request) {
    String productName = env.getRuntime().getProductName();
    return "  " + OutputDirectoryLinksUtils.getPrettyPath(artifact.getPath(),
        env.getWorkspaceName(), env.getWorkspace(),
        request.getBuildOptions().getSymlinkPrefix(productName));
  }

  /**
   * Prints a flat list of all artifacts built by the passed top-level targets.
   *
   * <p>This corresponds to the --experimental_show_artifacts flag.
   */
  public void showArtifacts(
      BuildRequest request,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<AspectValue> aspects) {

    TopLevelArtifactContext context = request.getTopLevelArtifactContext();
    Collection<ConfiguredTarget> targetsToPrint = filterTargetsToPrint(configuredTargets);
    Collection<AspectValue> aspectsToPrint = filterAspectsToPrint(aspects);

    NestedSetBuilder<Artifact> artifactsBuilder = NestedSetBuilder.stableOrder();
    for (ConfiguredTarget target : targetsToPrint) {
      artifactsBuilder.addTransitive(
          TopLevelArtifactHelper.getAllArtifactsToBuild(target, context).getImportantArtifacts());
    }

    for (AspectValue aspect : aspectsToPrint) {
      artifactsBuilder.addTransitive(
          TopLevelArtifactHelper.getAllArtifactsToBuild(aspect, context).getImportantArtifacts());
    }

    OutErr outErr = request.getOutErr();
    outErr.printErrLn("Build artifacts:");

    NestedSet<Artifact> artifacts = artifactsBuilder.build();
    for (Artifact artifact : artifacts) {
      if (!artifact.isSourceArtifact()) {
        outErr.printErrLn(">>>" + artifact.getPath());
      }
    }
  }

  /**
   * Returns a list of configured targets that should participate in printing.
   *
   * <p>Hidden rules and other inserted targets are ignored.
   */
  private Collection<ConfiguredTarget> filterTargetsToPrint(
      Collection<ConfiguredTarget> configuredTargets) {
    ImmutableList.Builder<ConfiguredTarget> result = ImmutableList.builder();
    for (ConfiguredTarget target : configuredTargets) {
      // TODO(bazel-team): this is quite ugly. Add a marker provider for this check.
      if (target instanceof InputFileConfiguredTarget) {
        // Suppress display of source files (because we do no work to build them).
        continue;
      }
      if (target.getTarget() instanceof Rule) {
        Rule rule = (Rule) target.getTarget();
        if (rule.getRuleClass().contains("$")) {
          // Suppress display of hidden rules
          continue;
        }
      }
      if (target instanceof OutputFileConfiguredTarget) {
        // Suppress display of generated files (because they appear underneath
        // their generating rule), EXCEPT those ones which are not part of the
        // filesToBuild of their generating rule (e.g. .par, _deploy.jar
        // files), OR when a user explicitly requests an output file but not
        // its rule.
        TransitiveInfoCollection generatingRule =
            ((OutputFileConfiguredTarget) target).getGeneratingRule();
        if (CollectionUtils.containsAll(
            generatingRule.getProvider(FileProvider.class).getFilesToBuild(),
            target.getProvider(FileProvider.class).getFilesToBuild())
            && configuredTargets.contains(generatingRule)) {
          continue;
        }
      }

      result.add(target);
    }
    return result.build();
  }

  private Collection<AspectValue> filterAspectsToPrint(Collection<AspectValue> aspects) {
    return aspects;
  }
}
