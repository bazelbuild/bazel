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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
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
   * Shows the result of the build. Information includes the list of up-to-date and failed targets
   * and list of output artifacts for successful targets
   *
   * <p>This corresponds to the --show_result flag.
   */
  void showBuildResult(
      BuildRequest request,
      BuildResult result,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<ConfiguredTarget> configuredTargetsToSkip,
      ImmutableMap<AspectKey, ConfiguredAspect> aspects) {
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
                : env.getWorkspace());
    OutErr outErr = request.getOutErr();

    // Produce output as if validation aspect didn't exist; instead we'll consult validation aspect
    // if we end up printing targets below. Note that in the presence of other aspects, we may print
    // success messages for them but the overall build will still fail if validation aspects (or
    // targets) failed.
    Collection<AspectKey> aspectsToPrint = aspects.keySet();
    if (request.useValidationAspect()) {
      aspectsToPrint =
          aspectsToPrint.stream()
              .filter(
                  k -> !BuildRequest.VALIDATION_ASPECT_NAME.equals(k.getAspectClass().getName()))
              .collect(ImmutableList.toImmutableList());
    }
    final boolean success;
    if (aspectsToPrint.isEmpty()) {
      // Suppress summary if --show_result value is exceeded:
      Collection<ConfiguredTarget> targetsToPrint = filterTargetsToPrint(configuredTargets);
      if (targetsToPrint.size() > request.getBuildOptions().maxResultTargets) {
        return;
      }

      // Filter the targets we care about into three buckets. Targets are only considered successful
      // if they and their validation aspects succeeded. Note we determined above that all aspects
      // are validation aspects, so just use the full keySet() here.
      ImmutableMap<ConfiguredTargetKey, Boolean> validated =
          aspects.keySet().stream()
              .collect(
                  ImmutableMap.toImmutableMap(
                      AspectKey::getBaseConfiguredTargetKey,
                      k -> result.getSuccessfulAspects().contains(k),
                      Boolean::logicalAnd));

      Collection<ConfiguredTarget> succeeded = new ArrayList<>();
      Collection<ConfiguredTarget> failed = new ArrayList<>();
      Collection<ConfiguredTarget> skipped = new ArrayList<>();
      Collection<ConfiguredTarget> successfulTargets = result.getSuccessfulTargets();
      for (ConfiguredTarget target : targetsToPrint) {
        if (configuredTargetsToSkip.contains(target)) {
          skipped.add(target);
        } else if (successfulTargets.contains(target)
            && validated.getOrDefault(
                ConfiguredTargetKey.builder().setConfiguredTarget(target).build(), Boolean.TRUE)) {
          succeeded.add(target);
        } else {
          failed.add(target);
        }
      }

      for (ConfiguredTarget target : skipped) {
        outErr.printErr("Target " + target.getLabel() + " was skipped\n");
      }

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
              outErr.printErrLn(
                  "  See temp at " + prettyPrinter.getPrettyPath(temp.getPath().asFragment()));
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
      Collection<AspectKey> succeeded = new ArrayList<>();
      Collection<AspectKey> failed = new ArrayList<>();
      ImmutableSet<AspectKey> successfulAspects = result.getSuccessfulAspects();
      for (AspectKey aspect : aspectsToPrint) {
        (successfulAspects.contains(aspect) ? succeeded : failed).add(aspect);
      }
      TopLevelArtifactContext context = request.getTopLevelArtifactContext();
      for (AspectKey aspect : succeeded) {
        Label label = aspect.getLabel();
        ConfiguredAspect configuredAspect = aspects.get(aspect);
        String aspectName = aspect.getAspectClass().getName();
        boolean headerFlag = true;
        NestedSet<Artifact> importantArtifacts =
            TopLevelArtifactHelper.getAllArtifactsToBuild(configuredAspect, context)
                .getImportantArtifacts();
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
      for (AspectKey aspect : failed) {
        Label label = aspect.getLabel();
        String aspectName = aspect.getAspectClass().getName();
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
    return "  " + prettyPrinter.getPrettyPath(artifact.getPath().asFragment());
  }

  /**
   * Prints a flat list of all artifacts built by the passed top-level targets.
   *
   * <p>This corresponds to the --experimental_show_artifacts flag.
   */
  void showArtifacts(
      BuildRequest request,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<ConfiguredAspect> aspects) {

    TopLevelArtifactContext context = request.getTopLevelArtifactContext();
    Collection<ConfiguredTarget> targetsToPrint = filterTargetsToPrint(configuredTargets);

    NestedSetBuilder<Artifact> artifactsBuilder = NestedSetBuilder.stableOrder();
    targetsToPrint.forEach(
        t ->
            artifactsBuilder.addTransitive(
                TopLevelArtifactHelper.getAllArtifactsToBuild(t, context).getImportantArtifacts()));

    aspects.forEach(
        a ->
            artifactsBuilder.addTransitive(
                TopLevelArtifactHelper.getAllArtifactsToBuild(a, context).getImportantArtifacts()));

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
        if (generatingRule
                .getProvider(FileProvider.class)
                .getFilesToBuild()
                .toSet()
                .containsAll(
                    configuredTarget.getProvider(FileProvider.class).getFilesToBuild().toList())
            && configuredTargets.contains(generatingRule)) {
          continue;
        }
      }

      result.add(configuredTarget);
    }
    return result.build();
  }
}
