// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genrule;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.CommandConstructor;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;

/**
 * A base implementation of genrule, to be used by specific implementing rules which can change some
 * of the semantics around when the execution info and inputs are changed.
 */
public abstract class GenRuleBase implements RuleConfiguredTargetFactory {

  /**
   * Returns {@code true} if the rule should be stamped.
   *
   * <p>Genrule implementations can set this based on the rule context, including by defining their
   * own attributes over and above what is present in {@link GenRuleBaseRule}.
   */
  protected abstract boolean isStampingEnabled(RuleContext ruleContext);

  /**
   * Updates the {@link RuleConfiguredTargetBuilder} that is used for this rule.
   *
   * <p>GenRule implementations can override this method to enhance and update the builder without
   * needing to entirely override the {@link com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory#create} method.
   */
  protected RuleConfiguredTargetBuilder updateBuilder(
      RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext,
      NestedSet<Artifact> filesToBuild) {
    return builder;
  }

  enum CommandType {
    BASH,
    WINDOWS_BATCH,
    WINDOWS_POWERSHELL,
  }

  private static Pair<CommandType, String> determineCommandTypeAndAttribute(
      RuleContext ruleContext) {
    AttributeMap attributeMap = ruleContext.attributes();
    // TODO(pcloudy): This should match the execution platform instead of using OS.getCurrent()
    if (OS.getCurrent() == OS.WINDOWS) {
      if (attributeMap.isAttributeValueExplicitlySpecified("cmd_ps")) {
        return Pair.of(CommandType.WINDOWS_POWERSHELL, "cmd_ps");
      }
      if (attributeMap.isAttributeValueExplicitlySpecified("cmd_bat")) {
        return Pair.of(CommandType.WINDOWS_BATCH, "cmd_bat");
      }
    }
    if (attributeMap.isAttributeValueExplicitlySpecified("cmd_bash")) {
      return Pair.of(CommandType.BASH, "cmd_bash");
    }
    if (attributeMap.isAttributeValueExplicitlySpecified("cmd")) {
      return Pair.of(CommandType.BASH, "cmd");
    }
    ruleContext.attributeError(
        "cmd",
        "missing value for `cmd` attribute, you can also set `cmd_ps` or `cmd_bat` on"
            + " Windows and `cmd_bash` on other platforms.");
    return null;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.wrap(Order.STABLE_ORDER, ruleContext.getOutputArtifacts());
    NestedSetBuilder<Artifact> resolvedSrcsBuilder = NestedSetBuilder.stableOrder();

    if (filesToBuild.isEmpty()) {
      ruleContext.attributeError("outs", "Genrules without outputs don't make sense");
    }
    if (ruleContext.attributes().get("executable", Type.BOOLEAN)
        && !filesToBuild.isEmpty()
        && !filesToBuild.isSingleton()) {
      ruleContext.attributeError(
          "executable",
          "if genrules produce executables, they are allowed only one output. "
              + "If you need the executable=1 argument, then you should split this genrule into "
              + "genrules producing single outputs");
    }

    Pair<CommandType, String> cmdTypeAndAttr = determineCommandTypeAndAttribute(ruleContext);

    ImmutableMap.Builder<Label, Iterable<Artifact>> labelMap = ImmutableMap.builder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("srcs")) {
      // This target provides specific types of files for genrules.
      GenRuleSourcesProvider provider = dep.getProvider(GenRuleSourcesProvider.class);
      NestedSet<Artifact> files = (provider != null)
          ? provider.getGenruleFiles()
          : dep.getProvider(FileProvider.class).getFilesToBuild();
      resolvedSrcsBuilder.addTransitive(files);
      // The CommandHelper class makes an explicit copy of this in the constructor, so flattening
      // here should be benign.
      labelMap.put(AliasProvider.getDependencyLabel(dep), files.toList());
    }
    NestedSet<Artifact> resolvedSrcs = resolvedSrcsBuilder.build();

    CommandHelper commandHelper =
        commandHelperBuilder(ruleContext).addLabelMap(labelMap.build()).build();

    if (ruleContext.hasErrors()) {
      return null;
    }

    CommandType cmdType = cmdTypeAndAttr.first;
    String cmdAttr = cmdTypeAndAttr.second;
    boolean expandToWindowsPath = cmdType == CommandType.WINDOWS_BATCH;

    String baseCommand = ruleContext.attributes().get(cmdAttr, Type.STRING);

    // Expand template variables and functions.
    ImmutableList.Builder<MakeVariableSupplier> makeVariableSuppliers =
        new ImmutableList.Builder<>();
    CommandResolverContext commandResolverContext =
        new CommandResolverContext(
            ruleContext,
            resolvedSrcs,
            filesToBuild,
            makeVariableSuppliers.build(),
            expandToWindowsPath);
    String command =
        ruleContext
            .getExpander(commandResolverContext)
            .withExecLocations(commandHelper.getLabelMap(), expandToWindowsPath)
            .expand(cmdAttr, baseCommand);

    // Heuristically expand things that look like labels.
    if (ruleContext.attributes().get("heuristic_label_expansion", Type.BOOLEAN)) {
      command = commandHelper.expandLabelsHeuristically(command);
    }

    if (cmdType == CommandType.BASH) {
      // Add the genrule environment setup script before the actual shell command.
      command =
          String.format(
              "source %s; %s",
              ruleContext.getPrerequisiteArtifact("$genrule_setup").getExecPath(), command);
    }

    String messageAttr = ruleContext.attributes().get("message", Type.STRING);
    String message = messageAttr.isEmpty() ? "Executing genrule" : messageAttr;
    Label label = ruleContext.getLabel();
    LazyString progressMessage =
        new LazyString() {
          @Override
          public String toString() {
            return message + " " + label;
          }
        };

    Map<String, String> executionInfo = Maps.newLinkedHashMap();
    executionInfo.putAll(TargetUtils.getExecutionInfo(ruleContext.getRule()));

    if (ruleContext.attributes().get("local", Type.BOOLEAN)) {
      executionInfo.put("local", "");
    }

    ruleContext.getConfiguration().modifyExecutionInfo(executionInfo, GenRuleAction.MNEMONIC);

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.addTransitive(resolvedSrcs);
    inputs.addTransitive(commandHelper.getResolvedTools());
    if (cmdType == CommandType.BASH) {
      FilesToRunProvider genruleSetup =
          ruleContext.getPrerequisite("$genrule_setup", FilesToRunProvider.class);
      inputs.addTransitive(genruleSetup.getFilesToRun());
    }
    if (ruleContext.hasErrors()) {
      return null;
    }

    CommandConstructor constructor;
    switch (cmdType) {
      case WINDOWS_BATCH:
        constructor = CommandHelper.buildWindowsBatchCommandConstructor(".genrule_script.bat");
        break;
      case WINDOWS_POWERSHELL:
        constructor = CommandHelper.buildWindowsPowershellCommandConstructor(".genrule_script.ps1");
        break;
      case BASH:
      default:
        PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
        constructor =
            CommandHelper.buildBashCommandConstructor(
                executionInfo, shExecutable, ".genrule_script.sh");
    }
    List<String> argv = commandHelper.buildCommandLine(command, inputs, constructor);

    if (isStampingEnabled(ruleContext)) {
      inputs.add(ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact());
      inputs.add(ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact());
    }

    ruleContext.registerAction(
        new GenRuleAction(
            ruleContext.getActionOwner(),
            commandHelper.getResolvedTools(),
            inputs.build(),
            filesToBuild.toSet(),
            CommandLines.of(argv),
            ruleContext.getConfiguration().getActionEnvironment(),
            ImmutableMap.copyOf(executionInfo),
            CompositeRunfilesSupplier.fromSuppliers(commandHelper.getToolsRunfilesSuppliers()),
            progressMessage));

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(
        // No runfiles provided if not a data dependency.
        Runfiles.EMPTY,
        // We only need to consider the outputs of a genrule
        // No need to visit the dependencies of a genrule. They cross from the target into the host
        // configuration, because the dependencies of a genrule are always built for the host
        // configuration.
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles())
            .addTransitiveArtifacts(filesToBuild)
            .build());

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(null, getExecutable(ruleContext, filesToBuild))
        .addProvider(RunfilesProvider.class, runfilesProvider);

    builder = updateBuilder(builder, ruleContext, filesToBuild);
    return builder.build();
  }

  protected CommandHelper.Builder commandHelperBuilder(RuleContext ruleContext) {
    return CommandHelper.builder(ruleContext)
        .addHostToolDependencies("tools")
        .addToolDependencies("exec_tools")
        .addHostToolDependencies("toolchains");
  }

  /**
   * Returns the executable artifact, if the rule is marked as executable and there is only one
   * artifact.
   */
  private static Artifact getExecutable(RuleContext ruleContext, NestedSet<Artifact> filesToBuild) {
    if (!ruleContext.attributes().get("executable", Type.BOOLEAN)) {
      return null;
    }
    return filesToBuild.isSingleton() ? filesToBuild.getSingleton() : null;
  }

  /**
   * Implementation of {@link ConfigurationMakeVariableContext} used to expand variables in a
   * genrule command string.
   */
  protected static class CommandResolverContext extends ConfigurationMakeVariableContext {

    private final RuleContext ruleContext;
    private final NestedSet<Artifact> resolvedSrcs;
    private final NestedSet<Artifact> filesToBuild;
    private final boolean windowsPath;

    public CommandResolverContext(
        RuleContext ruleContext,
        NestedSet<Artifact> resolvedSrcs,
        NestedSet<Artifact> filesToBuild,
        Iterable<? extends MakeVariableSupplier> makeVariableSuppliers,
        boolean windowsPath) {
      super(
          ruleContext,
          ruleContext.getRule().getPackage(),
          ruleContext.getConfiguration(),
          makeVariableSuppliers);
      this.ruleContext = ruleContext;
      this.resolvedSrcs = resolvedSrcs;
      this.filesToBuild = filesToBuild;
      this.windowsPath = windowsPath;
    }

    public RuleContext getRuleContext() {
      return ruleContext;
    }

    @Override
    public String lookupVariable(String variableName) throws ExpansionException {
      String val = lookupVariableImpl(variableName);
      if (windowsPath) {
        return val.replace('/', '\\');
      }
      return val;
    }

    private String lookupVariableImpl(String variableName) throws ExpansionException {
      if (variableName.equals("SRCS")) {
        return Artifact.joinExecPaths(" ", resolvedSrcs.toList());
      }

      if (variableName.equals("<")) {
        return expandSingletonArtifact(resolvedSrcs, "$<", "input file");
      }

      if (variableName.equals("OUTS")) {
        return Artifact.joinExecPaths(" ", filesToBuild.toList());
      }

      if (variableName.equals("@")) {
        return expandSingletonArtifact(filesToBuild, "$@", "output file");
      }

      PathFragment ruleDirPackagePath = ruleContext.getPackageDirectory();
      PathFragment ruleDirExecPath =
          ruleContext.getBinOrGenfilesDirectory().getExecPath().getRelative(ruleDirPackagePath);

      if (variableName.equals("RULEDIR")) {
        // The output root directory. This variable expands to the package's root directory
        // in the genfiles tree.
        return ruleDirExecPath.getPathString();
      }

      if (variableName.equals("@D")) {
        // The output directory. If there is only one filename in outs,
        // this expands to the directory containing that file. If there are
        // multiple filenames, this variable instead expands to the
        // package's root directory in the genfiles tree, even if all the
        // generated files belong to the same subdirectory!
        if (filesToBuild.isSingleton()) {
          Artifact outputFile = filesToBuild.getSingleton();
          PathFragment relativeOutputFile = outputFile.getExecPath();
          if (relativeOutputFile.segmentCount() <= 1) {
            // This should never happen, since the path should contain at
            // least a package name and a file name.
            throw new IllegalStateException(
                "$(@D) for genrule " + ruleContext.getLabel() + " has less than one segment");
          }
          return relativeOutputFile.getParentDirectory().getPathString();
        } else {
          return ruleDirExecPath.getPathString();
        }
      }

      return super.lookupVariable(variableName);
    }

    /**
     * Returns the path of the sole element "artifacts", generating an exception with an informative
     * error message iff the set is not a singleton. Used to expand "$<", "$@".
     */
    private static final String expandSingletonArtifact(
        NestedSet<Artifact> artifacts, String variable, String artifactName)
        throws ExpansionException {
      if (artifacts.isEmpty()) {
        throw new ExpansionException("variable '" + variable
            + "' : no " + artifactName);
      } else if (!artifacts.isSingleton()) {
        throw new ExpansionException("variable '" + variable
            + "' : more than one " + artifactName);
      }
      return artifacts.getSingleton().getExecPathString();
    }
  }
}
