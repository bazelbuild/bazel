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
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
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
      Collection<ConfiguredTarget> configuredTargetsToSkip,
      Collection<AspectValue> aspects) {
    // NOTE: be careful what you print!  We don't want to create a consistency
    // problem where the summary message and the exit code disagree.  The logic
    // here is already complex.

    BlazeRuntime runtime = env.getRuntime();
    String productName = runtime.getProductName();
    PathPrettyPrinter prettyPrinter =
        OutputDirectoryLinksUtils.getPathPrettyPrinter(
            runtime.getRuleClassProvider().getSymlinkDefinitions(),
            request.getBuildOptions().getSymlinkPrefix(productName),
            productName,
            env.getWorkspace(),
            request.getBuildOptions().printWorkspaceInOutputPathsIfNeeded
                ? env.getWorkingDirectory()
                : env.getWorkspace(),
            request.getBuildOptions().experimentalNoProductNameOutSymlink);
    OutErr outErr = request.getOutErr();
    Collection<ConfiguredTarget> targetsToPrint = filterTargetsToPrint(configuredTargets);
    Collection<AspectValue> aspectsToPrint = filterAspectsToPrint(aspects);
    final boolean success;
    if (aspectsToPrint.isEmpty()) {
      // Suppress summary if --show_result value is exceeded:
      if (targetsToPrint.size() > request.getBuildOptions().maxResultTargets) {
        return;
      }
      // Filter the targets we care about into two buckets:
      Collection<ConfiguredTarget> succeeded = new ArrayList<>();
      Collection<ConfiguredTarget> failed = new ArrayList<>();
      Collection<ConfiguredTarget> successfulTargets = result.getSuccessfulTargets();
      for (ConfiguredTarget target : targetsToPrint) {
        (successfulTargets.contains(target) ? succeeded : failed).add(target);
      }

      // TODO(bazel-team): convert these to a new "SKIPPED" status when ready: b/62191890.
      failed.addAll(configuredTargetsToSkip);

      TopLevelArtifactContext context = request.getTopLevelArtifactContext();
      for (ConfiguredTarget target : succeeded) {
        Label label = target.getLabel();
        // For up-to-date targets report generated artifacts, but only
        // if they have associated action and not middleman artifacts.
        boolean headerFlag = true;
        for (Artifact artifact :
            TopLevelArtifactHelper.getAllArtifactsToBuild(target, context)
                .getImportantArtifacts()
                .toList()) {
          if (shouldPrint(artifact)) {
            if (headerFlag) {
              outErr.printErr("Target " + label + " up-to-date:\n");
              headerFlag = false;
            }
            outErr.printErrLn(formatArtifactForShowResults(prettyPrinter, artifact));
          }
        }
        if (headerFlag) {
          outErr.printErr("Target " + label + " up-to-date (nothing to build)\n");
        }
      }
      for (ConfiguredTarget target : failed) {
        outErr.printErr("Target " + target.getLabel() + " failed to build\n");

        // For failed compilation, it is still useful to examine temp artifacts,
        // (ie, preprocessed and assembler files).
        OutputGroupInfo topLevelProvider = OutputGroupInfo.get(target);
        if (topLevelProvider != null) {
          for (Artifact temp :
              topLevelProvider.getOutputGroup(OutputGroupInfo.TEMP_FILES).toList()) {
            if (temp.getPath().exists()) {
              outErr.printErrLn("  See temp at " + prettyPrinter.getPrettyPath(temp.getPath()));
            }
          }
        }
      }
      success = failed.isEmpty();
    } else {
      // Suppress summary if --show_result value is exceeded:
      if (aspectsToPrint.size() > request.getBuildOptions().maxResultTargets) {
        return;
      }
      // Filter the targets we care about into two buckets:
      Collection<AspectValue> succeeded = new ArrayList<>();
      Collection<AspectValue> failed = new ArrayList<>();
      Collection<AspectValue> successfulAspects = result.getSuccessfulAspects();
      for (AspectValue aspect : aspectsToPrint) {
        (successfulAspects.contains(aspect) ? succeeded : failed).add(aspect);
      }
      TopLevelArtifactContext context = request.getTopLevelArtifactContext();
      for (AspectValue aspect : succeeded) {
        Label label = aspect.getLabel();
        String aspectName = aspect.getConfiguredAspect().getName();
        boolean headerFlag = true;
        NestedSet<Artifact> importantArtifacts =
            TopLevelArtifactHelper.getAllArtifactsToBuild(aspect, context).getImportantArtifacts();
        for (Artifact importantArtifact : importantArtifacts.toList()) {
          if (headerFlag) {
            outErr.printErr("Aspect " + aspectName + " of " + label + " up-to-date:\n");
            headerFlag = false;
          }
          if (shouldPrint(importantArtifact)) {
            outErr.printErrLn(formatArtifactForShowResults(prettyPrinter, importantArtifact));
          }
        }
        if (headerFlag) {
          outErr.printErr(
              "Aspect " + aspectName + " of " + label + " up-to-date (nothing to build)\n");
        }
      }
      for (AspectValue aspect : failed) {
        Label label = aspect.getLabel();
        String aspectName = aspect.getConfiguredAspect().getName();
        outErr.printErr("Aspect " + aspectName + " of " + label + " failed to build\n");
      }
      success = failed.isEmpty();
    }
    if (!success && !request.getOptions(ExecutionOptions.class).verboseFailures) {
      outErr.printErr("Use --verbose_failures to see the command lines of failed build steps.\n");
    }
  }

  private boolean shouldPrint(Artifact artifact) {
    return !artifact.isSourceArtifact() && !artifact.isMiddlemanArtifact();
  }

  private String formatArtifactForShowResults(PathPrettyPrinter prettyPrinter, Artifact artifact) {
    return "  " + prettyPrinter.getPrettyPath(artifact.getPath());
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
    for (Artifact artifact : artifacts.toList()) {
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
    for (ConfiguredTarget configuredTarget : configuredTargets) {
      // TODO(bazel-team): this is quite ugly. Add a marker provider for this check.
      if (configuredTarget instanceof InputFileConfiguredTarget) {
        // Suppress display of source files (because we do no work to build them).
        continue;
      }
      if (configuredTarget instanceof RuleConfiguredTarget) {
        RuleConfiguredTarget ruleCt = (RuleConfiguredTarget) configuredTarget;
        if (ruleCt.getRuleClassString().contains("$")) {
          // Suppress display of hidden rules
          continue;
        }
      }
      if (configuredTarget instanceof OutputFileConfiguredTarget) {
        // Suppress display of generated files (because they appear underneath
        // their generating rule), EXCEPT those ones which are not part of the
        // filesToBuild of their generating rule (e.g. .par, _deploy.jar
        // files), OR when a user explicitly requests an output file but not
        // its rule.
        TransitiveInfoCollection generatingRule =
            ((OutputFileConfiguredTarget) configuredTarget).getGeneratingRule();
        if (CollectionUtils.containsAll(
                generatingRule.getProvider(FileProvider.class).getFilesToBuild().toList(),
                configuredTarget.getProvider(FileProvider.class).getFilesToBuild().toList())
            && configuredTargets.contains(generatingRule)) {
          continue;
        }
      }

      result.add(configuredTarget);
    }
    return result.build();
  }

  private Collection<AspectValue> filterAspectsToPrint(Collection<AspectValue> aspects) {
    return aspects;
  }
}
