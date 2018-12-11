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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CompositeRunfilesSupplier;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
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
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * A base implementation of genrule, to be used by specific implementing rules which can change some
 * of the semantics around when the execution info and inputs are changed.
 */
public abstract class GenRuleBase implements RuleConfiguredTargetFactory {

  private static final ImmutableList<String> CROSSTOOL_MAKE_VARIABLES = ImmutableList.of("CC",
      "CC_FLAGS", "AR", "NM", "OBJCOPY", "STRIP", "GCOVTOOL");

  private static Pattern matchesMakeVariables(Iterable<String> variables) {
    return Pattern.compile("\\$\\((" + Joiner.on("|").join(variables) + ")\\)");
  }

  private static final Pattern CROSSTOOL_MAKE_VARIABLE_PATTERN =
      matchesMakeVariables(CROSSTOOL_MAKE_VARIABLES);

  protected static boolean requiresCrosstool(String command) {
    return CROSSTOOL_MAKE_VARIABLE_PATTERN.matcher(command).find();
  }

  /**
   * Returns a {@link Map} of execution info, which will be used in later processing to construct
   * the actual command line that will be executed.
   *
   * <p>GenRule implementations can override this method to include additional specific information
   * needed.
   */
  protected Map<String, String> getExtraExecutionInfo(RuleContext ruleContext, String command) {
    return ImmutableMap.of();
  }

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
        && Iterables.size(filesToBuild) > 1) {
      ruleContext.attributeError(
          "executable",
          "if genrules produce executables, they are allowed only one output. "
              + "If you need the executable=1 argument, then you should split this genrule into "
              + "genrules producing single outputs");
    }

    ImmutableMap.Builder<Label, NestedSet<Artifact>> labelMap = ImmutableMap.builder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("srcs", Mode.TARGET)) {
      // This target provides specific types of files for genrules.
      GenRuleSourcesProvider provider = dep.getProvider(GenRuleSourcesProvider.class);
      NestedSet<Artifact> files = (provider != null)
          ? provider.getGenruleFiles()
          : dep.getProvider(FileProvider.class).getFilesToBuild();
      resolvedSrcsBuilder.addTransitive(files);
      labelMap.put(AliasProvider.getDependencyLabel(dep), files);
    }
    NestedSet<Artifact> resolvedSrcs = resolvedSrcsBuilder.build();

    CommandHelper commandHelper =
        CommandHelper.builder(ruleContext)
            .addHostToolDependencies("tools")
            .addHostToolDependencies("toolchains")
            .addLabelMap(labelMap.build())
            .build();

    if (ruleContext.hasErrors()) {
      return null;
    }

    String baseCommand = ruleContext.attributes().get("cmd", Type.STRING);
    // Expand template variables and functions.
    ImmutableList.Builder<MakeVariableSupplier> makeVariableSuppliers =
        new ImmutableList.Builder<>();
    if (GenRuleBaseRule.enableCcToolchain(ruleContext.getConfiguration())) {
      makeVariableSuppliers.add(new CcFlagsSupplier(ruleContext));
    }
    CommandResolverContext commandResolverContext =
        new CommandResolverContext(
            ruleContext, resolvedSrcs, filesToBuild, makeVariableSuppliers.build());
    String command =
        ruleContext
            .getExpander(commandResolverContext)
            .withExecLocations(commandHelper.getLabelMap())
            .expand("cmd", baseCommand);

    // Heuristically expand things that look like labels.
    if (ruleContext.attributes().get("heuristic_label_expansion", Type.BOOLEAN)) {
      command = commandHelper.expandLabelsHeuristically(command);
    }

    // Add the genrule environment setup script before the actual shell command.
    command = String.format("source %s; %s",
        ruleContext.getPrerequisiteArtifact("$genrule_setup", Mode.HOST).getExecPath(),
        command);

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

    executionInfo.putAll(getExtraExecutionInfo(ruleContext, baseCommand));
    ruleContext.getConfiguration().modifyExecutionInfo(executionInfo, GenRuleAction.MNEMONIC);

    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.addTransitive(resolvedSrcs);
    inputs.addTransitive(commandHelper.getResolvedTools());
    FilesToRunProvider genruleSetup =
        ruleContext.getPrerequisite("$genrule_setup", Mode.HOST, FilesToRunProvider.class);
    inputs.addTransitive(genruleSetup.getFilesToRun());
    PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }
    List<String> argv =
        commandHelper.buildCommandLine(
            shExecutable,
            command,
            inputs,
            ".genrule_script.sh",
            ImmutableMap.copyOf(executionInfo));

    // TODO(bazel-team): Make the make variable expander pass back a list of these.
    if (GenRuleBaseRule.enableCcToolchain(ruleContext.getConfiguration())
        && requiresCrosstool(baseCommand)) {
      // If cc is used, silently throw in the crosstool filegroup as a dependency.
      inputs.addTransitive(
          CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext)
              .getCrosstoolMiddleman());
    }

    if (isStampingEnabled(ruleContext)) {
      inputs.add(ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact());
      inputs.add(ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact());
    }

    ruleContext.registerAction(
        new GenRuleAction(
            ruleContext.getActionOwner(),
            ImmutableList.copyOf(commandHelper.getResolvedTools()),
            inputs.build(),
            filesToBuild,
            CommandLines.of(argv),
            ruleContext.getConfiguration().getActionEnvironment(),
            ImmutableMap.copyOf(executionInfo),
            new CompositeRunfilesSupplier(commandHelper.getToolsRunfilesSuppliers()),
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

  /**
   * Returns the executable artifact, if the rule is marked as executable and there is only one
   * artifact.
   */
  private static Artifact getExecutable(RuleContext ruleContext, NestedSet<Artifact> filesToBuild) {
    if (!ruleContext.attributes().get("executable", Type.BOOLEAN)) {
      return null;
    }
    if (Iterables.size(filesToBuild) == 1) {
      return Iterables.getOnlyElement(filesToBuild);
    }
    return null;
  }

  /**
   * Implementation of {@link ConfigurationMakeVariableContext} used to expand variables in a
   * genrule command string.
   */
  protected static class CommandResolverContext extends ConfigurationMakeVariableContext {

    private final RuleContext ruleContext;
    private final NestedSet<Artifact> resolvedSrcs;
    private final NestedSet<Artifact> filesToBuild;

    public CommandResolverContext(
        RuleContext ruleContext,
        NestedSet<Artifact> resolvedSrcs,
        NestedSet<Artifact> filesToBuild,
        Iterable<? extends MakeVariableSupplier> makeVariableSuppliers) {
      super(
          ruleContext,
          ruleContext.getRule().getPackage(),
          ruleContext.getConfiguration(),
          makeVariableSuppliers);
      this.ruleContext = ruleContext;
      this.resolvedSrcs = resolvedSrcs;
      this.filesToBuild = filesToBuild;
    }

    public RuleContext getRuleContext() {
      return ruleContext;
    }

    @Override
    public String lookupVariable(String variableName) throws ExpansionException {
      if (variableName.equals("SRCS")) {
        return Artifact.joinExecPaths(" ", resolvedSrcs);
      }

      if (variableName.equals("<")) {
        return expandSingletonArtifact(resolvedSrcs, "$<", "input file");
      }

      if (variableName.equals("OUTS")) {
        return Artifact.joinExecPaths(" ", filesToBuild);
      }

      if (variableName.equals("@")) {
        return expandSingletonArtifact(filesToBuild, "$@", "output file");
      }

      if (variableName.equals("@D")) {
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
            throw new IllegalStateException(
                "$(@D) for genrule " + ruleContext.getLabel() + " has less than one segment");
          }
          return relativeOutputFile.getParentDirectory().getPathString();
        } else {
          PathFragment dir;
          if (ruleContext.getRule().hasBinaryOutput()) {
            dir = ruleContext.getConfiguration().getBinFragment();
          } else {
            dir = ruleContext.getConfiguration().getGenfilesFragment();
          }
          PathFragment relPath =
              ruleContext.getRule().getLabel().getPackageIdentifier().getSourceRoot();
          return dir.getRelative(relPath).getPathString();
        }
      }

      return super.lookupVariable(variableName);
    }

    /**
     * Returns the path of the sole element "artifacts", generating an exception with an informative
     * error message iff the set is not a singleton. Used to expand "$<", "$@".
     */
    private final String expandSingletonArtifact(Iterable<Artifact> artifacts,
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
}
