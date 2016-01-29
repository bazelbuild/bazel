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

package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.devtools.build.lib.analysis.RunfilesProvider.withData;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;
import java.util.Map;

/**
 * An implementation of genrule.
 */
public class GenRule implements RuleConfiguredTargetFactory {

  private Artifact getExecutable(RuleContext ruleContext, NestedSet<Artifact> filesToBuild) {
    if (Iterables.size(filesToBuild) == 1) {
      Artifact out = Iterables.getOnlyElement(filesToBuild);
      if (ruleContext.attributes().get("executable", Type.BOOLEAN)) {
        return out;
      }
    }
    return null;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    final List<Artifact> resolvedSrcs = Lists.newArrayList();

    final NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.wrap(Order.STABLE_ORDER, ruleContext.getOutputArtifacts());
    if (filesToBuild.isEmpty()) {
      ruleContext.attributeError("outs", "Genrules without outputs don't make sense");
    }
    if (ruleContext.attributes().get("executable", Type.BOOLEAN)
        && Iterables.size(filesToBuild) > 1) {
      ruleContext.attributeError("executable",
          "if genrules produce executables, they are allowed only one output. "
          + "If you need the executable=1 argument, then you should split this genrule into "
          + "genrules producing single outputs");
    }

    ImmutableMap.Builder<Label, Iterable<Artifact>> labelMap = ImmutableMap.builder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("srcs", Mode.TARGET)) {
      Iterable<Artifact> files = dep.getProvider(FileProvider.class).getFilesToBuild();
      Iterables.addAll(resolvedSrcs, files);
      labelMap.put(dep.getLabel(), files);
    }

    CommandHelper commandHelper =
        new CommandHelper(
            ruleContext, ruleContext.getPrerequisites("tools", Mode.HOST), labelMap.build());

    if (ruleContext.hasErrors()) {
      return null;
    }

    String baseCommand = commandHelper.resolveCommandAndExpandLabels(
        ruleContext.attributes().get("heuristic_label_expansion", Type.BOOLEAN), false);

    // Adds the genrule environment setup script before the actual shell command
    String command = String.format("source %s; %s",
        ruleContext.getPrerequisiteArtifact("$genrule_setup", Mode.HOST).getExecPath(),
        baseCommand);

    command = resolveCommand(ruleContext, command, resolvedSrcs, filesToBuild);

    String message = ruleContext.attributes().get("message", Type.STRING);
    if (message.isEmpty()) {
      message = "Executing genrule";
    }

    ImmutableMap<String, String> env = ruleContext.getConfiguration().getLocalShellEnvironment();

    Map<String, String> executionInfo = Maps.newLinkedHashMap();
    executionInfo.putAll(TargetUtils.getExecutionInfo(ruleContext.getRule()));

    if (ruleContext.attributes().get("local", Type.BOOLEAN)) {
      executionInfo.put("local", "");
    }

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.addAll(resolvedSrcs);
    inputs.addAll(commandHelper.getResolvedTools());
    FilesToRunProvider genruleSetup =
        ruleContext.getPrerequisite("$genrule_setup", Mode.HOST, FilesToRunProvider.class);
    inputs.addAll(genruleSetup.getFilesToRun());
    List<String> argv = commandHelper.buildCommandLine(command, inputs, ".genrule_script.sh");

    if (ruleContext.attributes().get("stamp", Type.BOOLEAN)) {
      inputs.add(ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact());
      inputs.add(ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact());
    }

    ruleContext.registerAction(
        new GenRuleAction(
            ruleContext.getActionOwner(),
            ImmutableList.copyOf(commandHelper.getResolvedTools()),
            inputs.build(),
            filesToBuild,
            argv,
            env,
            ImmutableMap.copyOf(executionInfo),
            ImmutableMap.copyOf(commandHelper.getRemoteRunfileManifestMap()),
            message + ' ' + ruleContext.getLabel()));

    RunfilesProvider runfilesProvider = withData(
        // No runfiles provided if not a data dependency.
        Runfiles.EMPTY,
        // We only need to consider the outputs of a genrule
        // No need to visit the dependencies of a genrule. They cross from the target into the host
        // configuration, because the dependencies of a genrule are always built for the host
        // configuration.
        new Runfiles.Builder(ruleContext.getWorkspaceName()).addTransitiveArtifacts(filesToBuild)
            .build());

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(null, getExecutable(ruleContext, filesToBuild))
        .addProvider(RunfilesProvider.class, runfilesProvider)
        .build();
  }

  private String resolveCommand(final RuleContext ruleContext, final String command,
      final List<Artifact> resolvedSrcs, final NestedSet<Artifact> filesToBuild) {
    return ruleContext.expandMakeVariables("cmd", command, new ConfigurationMakeVariableContext(
        ruleContext.getRule().getPackage(), ruleContext.getConfiguration()) {
          @Override
          public String lookupMakeVariable(String name) throws ExpansionException {
            if (name.equals("SRCS")) {
              return Artifact.joinExecPaths(" ", resolvedSrcs);
            } else if (name.equals("<")) {
              return expandSingletonArtifact(resolvedSrcs, "$<", "input file");
            } else if (name.equals("OUTS")) {
              return Artifact.joinExecPaths(" ", filesToBuild);
            } else if (name.equals("@")) {
              return expandSingletonArtifact(filesToBuild, "$@", "output file");
            } else if (name.equals("@D")) {
              // The output directory. If there is only one filename in outs,
              // this expands to the directory containing that file. If there are
              // multiple filenames, this variable instead expands to the
              // package's root directory in the genfiles tree, even if all the
              // generated files belong to the same subdirectory!
              if (Iterables.size(filesToBuild) == 1) {
                Artifact outputFile = Iterables.getOnlyElement(filesToBuild);
                PathFragment relativeOutputFile = outputFile.getExecPath();
                if (relativeOutputFile.segmentCount() <= 1) {
                  // This should never happen, since the path should contain at
                  // least a package name and a file name.
                  throw new IllegalStateException("$(@D) for genrule " + ruleContext.getLabel()
                      + " has less than one segment");
                }
                return relativeOutputFile.getParentDirectory().getPathString();
              } else {
                PathFragment dir;
                if (ruleContext.getRule().hasBinaryOutput()) {
                  dir = ruleContext.getConfiguration().getBinFragment();
                } else {
                  dir = ruleContext.getConfiguration().getGenfilesFragment();
                }
                PathFragment relPath = ruleContext.getRule().getLabel().getPackageFragment();
                return dir.getRelative(relPath).getPathString();
              }
            } else {
              return super.lookupMakeVariable(name);
            }
          }
        }
    );
  }

  // Returns the path of the sole element "artifacts", generating an exception
  // with an informative error message iff the set is not a singleton.
  //
  // Used to expand "$<", "$@"
  private String expandSingletonArtifact(Iterable<Artifact> artifacts,
                                         String variable,
                                         String artifactName)
      throws ExpansionException {
    if (Iterables.isEmpty(artifacts)) {
      throw new ExpansionException("variable '" + variable
                                   + "' : no " + artifactName);
    } else if (Iterables.size(artifacts) > 1) {
      throw new ExpansionException("variable '" + variable
                                   + "' : more than one " + artifactName);
    }
    return Iterables.getOnlyElement(artifacts).getExecPathString();
  }
}
