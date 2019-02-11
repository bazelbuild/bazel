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
package com.google.devtools.build.lib.analysis.skylark;

import static java.util.stream.Collectors.toList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SingleStringArgFormatter;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.skylark.SkylarkCustomCommandLine.ScalarArg;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FunctionSignature.Shape;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkMutable;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import javax.annotation.Nullable;

/** Provides a Skylark interface for all action creation needs. */
public class SkylarkActionFactory implements SkylarkActionFactoryApi {
  private final SkylarkRuleContext context;
  private final SkylarkSemantics skylarkSemantics;
  private RuleContext ruleContext;
  /** Counter for actions.run_shell helper scripts. Every script must have a unique name. */
  private int runShellOutputCounter = 0;

  public SkylarkActionFactory(
      SkylarkRuleContext context,
      SkylarkSemantics skylarkSemantics,
      RuleContext ruleContext) {
    this.context = context;
    this.skylarkSemantics = skylarkSemantics;
    this.ruleContext = ruleContext;
  }

  ArtifactRoot newFileRoot() throws EvalException {
    return context.isForAspect()
        ? ruleContext.getConfiguration().getBinDirectory(ruleContext.getRule().getRepository())
        : ruleContext.getBinOrGenfilesDirectory();
  }

  /**
   * Returns a {@link ActionRegistry} object to register actions using this action factory.
   *
   * @throws EvalException if actions cannot be registered with this object
   */
  public ActionRegistry asActionRegistry(
      final Location location, SkylarkActionFactory skylarkActionFactory) throws EvalException {
    validateActionCreation(location);
    return new ActionRegistry() {

      @Override
      public void registerAction(ActionAnalysisMetadata... actions) {
        ruleContext.registerAction(actions);
      }

      @Override
      public ArtifactOwner getOwner() {
        return skylarkActionFactory
            .getActionConstructionContext()
            .getAnalysisEnvironment()
            .getOwner();
      }
    };
  }

  @Override
  public Artifact declareFile(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_file");
    if (Runtime.NONE.equals(sibling)) {
      return ruleContext.getPackageRelativeArtifact(filename, newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      return ruleContext.getDerivedArtifact(fragment, newFileRoot());
    }
  }

  @Override
  public Artifact declareDirectory(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_directory");
    Artifact result;
    if (Runtime.NONE.equals(sibling)) {
      result =
          ruleContext.getPackageRelativeTreeArtifact(PathFragment.create(filename), newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      result = ruleContext.getTreeArtifact(fragment, newFileRoot());
    }
    if (!result.isTreeArtifact()) {
      throw new EvalException(
          null,
          String.format(
              "'%s' has already been declared as a regular file, not directory.", filename));
    }
    return result;
  }

  @Override
  public void doNothing(String mnemonic, Object inputs, Location location) throws EvalException {
    context.checkMutable("actions.do_nothing");
    NestedSet<Artifact> inputSet = inputs instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) inputs).getSet(Artifact.class)
        : NestedSetBuilder.<Artifact>compileOrder()
            .addAll(((SkylarkList) inputs).getContents(Artifact.class, "inputs"))
            .build();
    Action action =
        new PseudoAction<>(
            UUID.nameUUIDFromBytes(
                String.format("empty action %s", ruleContext.getLabel())
                    .getBytes(StandardCharsets.UTF_8)),
            ruleContext.getActionOwner(),
            inputSet,
            ImmutableList.of(PseudoAction.getDummyOutput(ruleContext)),
            mnemonic,
            SPAWN_INFO,
            SpawnInfo.newBuilder().build());
    registerAction(location, action);
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final GeneratedMessage.GeneratedExtension<ExtraActionInfo, SpawnInfo> SPAWN_INFO =
      SpawnInfo.spawnInfo;

  @Override
  public void write(FileApi output, Object content, Boolean isExecutable, Location location)
      throws EvalException {
    context.checkMutable("actions.write");
    final Action action;
    if (content instanceof String) {
      action =
          FileWriteAction.create(ruleContext, (Artifact) output, (String) content, isExecutable);
    } else if (content instanceof Args) {
      Args args = (Args) content;
      action =
          new ParameterFileWriteAction(
              ruleContext.getActionOwner(),
              skylarkSemantics.incompatibleExpandDirectories()
                  ? args.getDirectoryArtifacts()
                  : ImmutableList.of(),
              (Artifact) output,
              args.build(),
              args.parameterFileType,
              StandardCharsets.UTF_8);
    } else {
      throw new AssertionError("Unexpected type: " + content.getClass().getSimpleName());
    }
    registerAction(location, action);
  }

  @Override
  public void run(
      SkylarkList outputs,
      Object inputs,
      Object executableUnchecked,
      Object toolsUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location location)
      throws EvalException {
    context.checkMutable("actions.run");
    SpawnAction.Builder builder = new SpawnAction.Builder();

    SkylarkList argumentsList = ((SkylarkList) arguments);
    buildCommandLine(builder, argumentsList);
    if (executableUnchecked instanceof Artifact) {
      Artifact executable = (Artifact) executableUnchecked;
      FilesToRunProvider provider = context.getExecutableRunfiles(executable);
      if (provider == null) {
        builder.setExecutable(executable);
      } else {
        builder.setExecutable(provider);
      }
    } else if (executableUnchecked instanceof String) {
      builder.setExecutable(PathFragment.create((String) executableUnchecked));
    } else if (executableUnchecked instanceof FilesToRunProvider) {
      builder.setExecutable((FilesToRunProvider) executableUnchecked);
    } else {
      // Should have been verified by Starlark before this function is called
      throw new IllegalStateException();
    }
    registerSpawnAction(
        outputs,
        inputs,
        toolsUnchecked,
        mnemonicUnchecked,
        progressMessage,
        useDefaultShellEnv,
        envUnchecked,
        executionRequirementsUnchecked,
        inputManifestsUnchecked,
        location,
        builder);
  }

  private void validateActionCreation(Location location) throws EvalException {
    if (ruleContext.getRule().isAnalysisTest()) {
      throw new EvalException(
          location,
          "implementation function of a rule with "
              + "analysis_test=true may not register actions. Analysis test rules may only return "
              + "success/failure information via AnalysisTestResultInfo.");
    }
  }

  /**
   * Registers actions in the context of this {@link SkylarkActionFactory}.
   *
   * <p>Use {@link #getActionConstructionContext()} to obtain the context required to create those
   * actions.
   */
  public void registerAction(Location location, ActionAnalysisMetadata... actions)
      throws EvalException {
    validateActionCreation(location);
    ruleContext.registerAction(actions);
  }

  /**
   * Returns information needed to construct actions that can be registered with {@link
   * #registerAction(Location, ActionAnalysisMetadata...)}.
   */
  public ActionConstructionContext getActionConstructionContext() {
    return ruleContext;
  }

  @Override
  public void runShell(
      SkylarkList outputs,
      Object inputs,
      Object toolsUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location location)
      throws EvalException {
    context.checkMutable("actions.run_shell");

    SkylarkList argumentList = (SkylarkList) arguments;
    SpawnAction.Builder builder = new SpawnAction.Builder();
    buildCommandLine(builder, argumentList);

    if (commandUnchecked instanceof String) {
      Map<String, String> executionInfo =
          ImmutableMap.copyOf(TargetUtils.getExecutionInfo(ruleContext.getRule()));
      String helperScriptSuffix = String.format(".run_shell_%d.sh", runShellOutputCounter++);
      String command = (String) commandUnchecked;
      Artifact helperScript =
          CommandHelper.shellCommandHelperScriptMaybe(
              ruleContext, command, helperScriptSuffix, executionInfo);
      PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
      if (helperScript == null) {
        builder.setShellCommand(shExecutable, command);
      } else {
        builder.setShellCommand(shExecutable, helperScript.getExecPathString());
        builder.addInput(helperScript);
        FilesToRunProvider provider = context.getExecutableRunfiles(helperScript);
        if (provider != null) {
          builder.addTool(provider);
        }
      }
    } else if (commandUnchecked instanceof SkylarkList) {
      SkylarkList commandList = (SkylarkList) commandUnchecked;
      if (argumentList.size() > 0) {
        throw new EvalException(location,
            "'arguments' must be empty if 'command' is a sequence of strings");
      }
      @SuppressWarnings("unchecked")
      List<String> command = commandList.getContents(String.class, "command");
      builder.setShellCommand(command);
    } else {
      throw new EvalException(
          null,
          "expected string or list of strings for command instead of "
              + EvalUtils.getDataTypeName(commandUnchecked));
    }
    if (argumentList.size() > 0) {
      // When we use a shell command, add an empty argument before other arguments.
      //   e.g.  bash -c "cmd" '' 'arg1' 'arg2'
      // bash will use the empty argument as the value of $0 (which we don't care about).
      // arg1 and arg2 will be $1 and $2, as a user expects.
      builder.addExecutableArguments("");
    }
    registerSpawnAction(
        outputs,
        inputs,
        toolsUnchecked,
        mnemonicUnchecked,
        progressMessage,
        useDefaultShellEnv,
        envUnchecked,
        executionRequirementsUnchecked,
        inputManifestsUnchecked,
        location,
        builder);
  }

  private void buildCommandLine(SpawnAction.Builder builder, SkylarkList argumentsList)
      throws EvalException {
    List<String> stringArgs = new ArrayList<>();
    for (Object value : argumentsList) {
      if (value instanceof String) {
        stringArgs.add((String) value);
      } else if (value instanceof Args) {
        if (!stringArgs.isEmpty()) {
          builder.addCommandLine(CommandLine.of(stringArgs));
          stringArgs = new ArrayList<>();
        }
        Args args = (Args) value;
        ParamFileInfo paramFileInfo = null;
        if (args.flagFormatString != null) {
          paramFileInfo =
              ParamFileInfo.builder(args.parameterFileType)
                  .setFlagFormatString(args.flagFormatString)
                  .setUseAlways(args.useAlways)
                  .setCharset(StandardCharsets.UTF_8)
                  .build();
        }
        builder.addCommandLine(args.commandLine.build(), paramFileInfo);
      } else {
        throw new EvalException(
            null,
            "expected list of strings or ctx.actions.args() for arguments instead of "
                + EvalUtils.getDataTypeName(value));
      }
    }
    if (!stringArgs.isEmpty()) {
      builder.addCommandLine(CommandLine.of(stringArgs));
    }
  }

  /**
   * Setup for spawn actions common between {@link #run} and {@link #runShell}.
   *
   * <p>{@code builder} should have either executable or a command set.
   */
  private void registerSpawnAction(
      SkylarkList outputs,
      Object inputs,
      Object toolsUnchecked,
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location location,
      SpawnAction.Builder builder)
      throws EvalException {
    Iterable<Artifact> inputArtifacts;
    if (inputs instanceof SkylarkList) {
      inputArtifacts = ((SkylarkList) inputs).getContents(Artifact.class, "inputs");
      builder.addInputs(inputArtifacts);
    } else {
      NestedSet<Artifact> inputSet = ((SkylarkNestedSet) inputs).getSet(Artifact.class);
      builder.addTransitiveInputs(inputSet);
      inputArtifacts = inputSet;
    }
    builder.addOutputs(outputs.getContents(Artifact.class, "outputs"));

    if (toolsUnchecked != Runtime.UNBOUND) {
      @SuppressWarnings("unchecked")
      Iterable<Object> toolsIterable;
      if (toolsUnchecked instanceof SkylarkList) {
        toolsIterable = ((SkylarkList<Object>) toolsUnchecked).getContents(Object.class, "tools");
      } else {
        toolsIterable = ((SkylarkNestedSet) toolsUnchecked).getSet(Object.class);
      }
      for (Object toolUnchecked : toolsIterable) {
        if (toolUnchecked instanceof Artifact) {
          Artifact artifact = (Artifact) toolUnchecked;
          builder.addInput(artifact);
          FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
          if (provider != null) {
            builder.addTool(provider);
          }
        } else if (toolUnchecked instanceof FilesToRunProvider) {
          builder.addTool((FilesToRunProvider) toolUnchecked);
        } else {
          throw new EvalException(
              null,
              "expected value of type 'File or FilesToRunProvider' for "
                  + "a member of parameter 'tools' but got "
                  + EvalUtils.getDataTypeName(toolUnchecked)
                  + " instead");
        }
      }
    } else {
      // Users didn't pass 'tools', kick in compatibility modes
      if (skylarkSemantics.incompatibleNoSupportToolsInActionInputs()) {
        // In this mode we error out if we find any tools among the inputs
        List<Artifact> tools = null;
        for (Artifact artifact : inputArtifacts) {
          FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
          if (provider != null) {
            tools = tools != null ? tools : new ArrayList<>(1);
            tools.add(artifact);
          }
        }
        if (tools != null) {
          String toolsAsString =
              Joiner.on(", ")
                  .join(
                      tools
                          .stream()
                          .map(Artifact::getExecPathString)
                          .map(s -> "'" + s + "'")
                          .collect(toList()));
          throw new EvalException(
              location,
              String.format(
                  "Found tool(s) %s in inputs. "
                      + "A tool is an input with executable=True set. "
                      + "All tools should be passed using the 'tools' "
                      + "argument instead of 'inputs' in order to make their runfiles available "
                      + "to the action. This safety check will not be performed once the action "
                      + "is modified to take a 'tools' argument. "
                      + "To temporarily disable this check, "
                      + "set --incompatible_no_support_tools_in_action_inputs=false.",
                  toolsAsString));
        }
      } else {
        // Full legacy support -- add tools from inputs
        for (Artifact artifact : inputArtifacts) {
          FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
          if (provider != null) {
            builder.addTool(provider);
          }
        }
      }
    }

    String mnemonic = getMnemonic(mnemonicUnchecked);
    builder.setMnemonic(mnemonic);
    if (envUnchecked != Runtime.NONE) {
      builder.setEnvironment(
          ImmutableMap.copyOf(
              SkylarkDict.castSkylarkDictOrNoneToDict(
                  envUnchecked, String.class, String.class, "env")));
    }
    if (progressMessage != Runtime.NONE) {
      builder.setProgressMessageNonLazy((String) progressMessage);
    }
    if (EvalUtils.toBoolean(useDefaultShellEnv)) {
      builder.useDefaultShellEnvironment();
    }
    if (executionRequirementsUnchecked != Runtime.NONE) {
      builder.setExecutionInfo(
          TargetUtils.filter(
              SkylarkDict.castSkylarkDictOrNoneToDict(
                  executionRequirementsUnchecked,
                  String.class,
                  String.class,
                  "execution_requirements")));
    }
    if (inputManifestsUnchecked != Runtime.NONE) {
      for (RunfilesSupplier supplier : SkylarkList.castSkylarkListOrNoneToList(
          inputManifestsUnchecked, RunfilesSupplier.class, "runfiles suppliers")) {
        builder.addRunfilesSupplier(supplier);
      }
    }
    // Always register the action
    registerAction(location, builder.build(ruleContext));
  }

  private String getMnemonic(Object mnemonicUnchecked) {
    String mnemonic =
        mnemonicUnchecked == Runtime.NONE ? "SkylarkAction" : (String) mnemonicUnchecked;
    if (ruleContext.getConfiguration().getReservedActionMnemonics().contains(mnemonic)) {
      mnemonic = mangleMnemonic(mnemonic);
    }
    return mnemonic;
  }

  private static String mangleMnemonic(String mnemonic) {
    return mnemonic + "FromSkylark";
  }

  @Override
  public void expandTemplate(
      FileApi template,
      FileApi output,
      SkylarkDict<?, ?> substitutionsUnchecked,
      Boolean executable,
      Location location)
      throws EvalException {
    context.checkMutable("actions.expand_template");
    ImmutableList.Builder<Substitution> substitutionsBuilder = ImmutableList.builder();
    for (Map.Entry<String, String> substitution :
        substitutionsUnchecked
            .getContents(String.class, String.class, "substitutions")
            .entrySet()) {
      // ParserInputSource.create(Path) uses Latin1 when reading BUILD files, which might
      // contain UTF-8 encoded symbols as part of template substitution.
      // As a quick fix, the substitution values are corrected before being passed on.
      // In the long term, fixing ParserInputSource.create(Path) would be a better approach.
      substitutionsBuilder.add(
          Substitution.of(
              substitution.getKey(), convertLatin1ToUtf8(substitution.getValue())));
    }
    TemplateExpansionAction action =
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            (Artifact) template,
            (Artifact) output,
            substitutionsBuilder.build(),
            executable);
    registerAction(location, action);
  }

  /**
   * Returns the proper UTF-8 representation of a String that was erroneously read using Latin1.
   *
   * @param latin1 Input string
   * @return The input string, UTF8 encoded
   */
  private static String convertLatin1ToUtf8(String latin1) {
    return new String(latin1.getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8);
  }

  /** Args module. */
  @VisibleForTesting
  public static class Args extends SkylarkMutable implements CommandLineArgsApi {
    private final Mutability mutability;
    private final SkylarkSemantics skylarkSemantics;
    private final SkylarkCustomCommandLine.Builder commandLine;
    private List<NestedSet<Object>> potentialDirectoryArtifacts = new ArrayList<>();
    private final Set<Artifact> directoryArtifacts = new HashSet<>();
    private ParameterFileType parameterFileType = ParameterFileType.SHELL_QUOTED;
    private String flagFormatString;
    private boolean useAlways;

    @Override
    public CommandLineArgsApi addArgument(
        Object argNameOrValue,
        Object value,
        Object format,
        Object beforeEach,
        Object joinWith,
        Object mapFn,
        Location loc)
        throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      final String argName;
      if (value == Runtime.UNBOUND) {
        value = argNameOrValue;
        argName = null;
      } else {
        validateArgName(argNameOrValue, loc);
        argName = (String) argNameOrValue;
      }
      if (argName != null) {
        commandLine.add(argName);
      }
      if (value instanceof SkylarkNestedSet || value instanceof SkylarkList) {
        if (skylarkSemantics.incompatibleDisallowOldStyleArgsAdd()) {
          throw new EvalException(
              loc,
              "Args#add no longer accepts vectorized arguments when "
                  + "--incompatible_disallow_old_style_args_add is set. "
                  + "Please use Args#add_all or Args#add_joined.");
        }
        addVectorArg(
            value,
            /* argName= */ null,
            mapFn != Runtime.NONE ? (BaseFunction) mapFn : null,
            /* mapEach= */ null,
            format != Runtime.NONE ? (String) format : null,
            beforeEach != Runtime.NONE ? (String) beforeEach : null,
            joinWith != Runtime.NONE ? (String) joinWith : null,
            /* formatJoined= */ null,
            /* omitIfEmpty= */ false,
            /* uniquify= */ false,
            skylarkSemantics.incompatibleExpandDirectories(),
            /* terminateWith= */ null,
            loc);

      } else {
        if (mapFn != Runtime.NONE && skylarkSemantics.incompatibleDisallowOldStyleArgsAdd()) {
          throw new EvalException(
              loc,
              "Args#add no longer accepts map_fn when"
                  + "--incompatible_disallow_old_style_args_add is set. "
                  + "Please eagerly map the value.");
        }
        if (beforeEach != Runtime.NONE) {
          throw new EvalException(null, "'before_each' is not supported for scalar arguments");
        }
        if (joinWith != Runtime.NONE) {
          throw new EvalException(null, "'join_with' is not supported for scalar arguments");
        }
        addScalarArg(
            value,
            format != Runtime.NONE ? (String) format : null,
            mapFn != Runtime.NONE ? (BaseFunction) mapFn : null,
            loc);
      }
      return this;
    }

    @Override
    public CommandLineArgsApi addAll(
        Object argNameOrValue,
        Object values,
        Object mapEach,
        Object formatEach,
        Object beforeEach,
        Boolean omitIfEmpty,
        Boolean uniquify,
        Object expandDirectories,
        Object terminateWith,
        Location loc)
        throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      final String argName;
      if (values == Runtime.UNBOUND) {
        values = argNameOrValue;
        validateValues(values, loc);
        argName = null;
      } else {
        validateArgName(argNameOrValue, loc);
        argName = (String) argNameOrValue;
      }
      addVectorArg(
          values,
          argName,
          /* mapAll= */ null,
          mapEach != Runtime.NONE ? (BaseFunction) mapEach : null,
          formatEach != Runtime.NONE ? (String) formatEach : null,
          beforeEach != Runtime.NONE ? (String) beforeEach : null,
          /* joinWith= */ null,
          /* formatJoined= */ null,
          omitIfEmpty,
          uniquify,
          expandDirectories == Runtime.UNBOUND
              ? skylarkSemantics.incompatibleExpandDirectories()
              : (Boolean) expandDirectories,
          terminateWith != Runtime.NONE ? (String) terminateWith : null,
          loc);
      return this;
    }

    @Override
    public CommandLineArgsApi addJoined(
        Object argNameOrValue,
        Object values,
        String joinWith,
        Object mapEach,
        Object formatEach,
        Object formatJoined,
        Boolean omitIfEmpty,
        Boolean uniquify,
        Object expandDirectories,
        Location loc)
        throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      final String argName;
      if (values == Runtime.UNBOUND) {
        values = argNameOrValue;
        validateValues(values, loc);
        argName = null;
      } else {
        validateArgName(argNameOrValue, loc);
        argName = (String) argNameOrValue;
      }
      addVectorArg(
          values,
          argName,
          /* mapAll= */ null,
          mapEach != Runtime.NONE ? (BaseFunction) mapEach : null,
          formatEach != Runtime.NONE ? (String) formatEach : null,
          /* beforeEach= */ null,
          joinWith,
          formatJoined != Runtime.NONE ? (String) formatJoined : null,
          omitIfEmpty,
          uniquify,
          expandDirectories == Runtime.UNBOUND
              ? skylarkSemantics.incompatibleExpandDirectories()
              : (Boolean) expandDirectories,
          /* terminateWith= */ null,
          loc);
      return this;
    }

    private void addVectorArg(
        Object value,
        String argName,
        BaseFunction mapAll,
        BaseFunction mapEach,
        String formatEach,
        String beforeEach,
        String joinWith,
        String formatJoined,
        boolean omitIfEmpty,
        boolean uniquify,
        boolean expandDirectories,
        String terminateWith,
        Location loc)
        throws EvalException {
      SkylarkCustomCommandLine.VectorArg.Builder vectorArg;
      if (value instanceof SkylarkNestedSet) {
        SkylarkNestedSet skylarkNestedSet = ((SkylarkNestedSet) value);
        NestedSet<Object> nestedSet = skylarkNestedSet.getSet(Object.class);
        if (expandDirectories) {
          potentialDirectoryArtifacts.add(nestedSet);
        }
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(nestedSet);
      } else {
        @SuppressWarnings("unchecked")
        SkylarkList<Object> skylarkList = (SkylarkList<Object>) value;
        if (expandDirectories) {
          scanForDirectories(skylarkList);
        }
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(skylarkList);
      }
      validateMapEach(mapEach, loc);
      validateFormatString("format_each", formatEach, loc);
      validateFormatString("format_joined", formatJoined, loc);
      vectorArg
          .setLocation(loc)
          .setArgName(argName)
          .setExpandDirectories(expandDirectories)
          .setMapAll(mapAll)
          .setFormatEach(formatEach)
          .setBeforeEach(beforeEach)
          .setJoinWith(joinWith)
          .setFormatJoined(formatJoined)
          .omitIfEmpty(omitIfEmpty)
          .uniquify(uniquify)
          .setTerminateWith(terminateWith)
          .setMapEach(mapEach);
      commandLine.add(vectorArg);
    }

    private void validateArgName(Object argName, Location loc) throws EvalException {
      if (!(argName instanceof String)) {
        throw new EvalException(
            loc,
            String.format(
                "expected value of type 'string' for arg name, got '%s'",
                argName.getClass().getSimpleName()));
      }
    }

    private void validateValues(Object values, Location loc) throws EvalException {
      if (!(values instanceof SkylarkList || values instanceof SkylarkNestedSet)) {
        throw new EvalException(
            loc,
            String.format(
                "expected value of type 'sequence or depset' for values, got '%s'",
                values.getClass().getSimpleName()));
      }
    }

    private void validateMapEach(@Nullable BaseFunction mapEach, Location loc)
        throws EvalException {
      if (mapEach == null) {
        return;
      }
      Shape shape = mapEach.getSignature().getSignature().getShape();
      boolean valid =
          shape.getMandatoryPositionals() == 1
              && shape.getOptionalPositionals() == 0
              && shape.getMandatoryNamedOnly() == 0
              && shape.getOptionalPositionals() == 0;
      if (!valid) {
        throw new EvalException(
            loc, "map_each must be a function that accepts a single positional argument");
      }
    }

    private void validateFormatString(String argumentName, @Nullable String formatStr, Location loc)
        throws EvalException {
      if (formatStr != null
          && skylarkSemantics.incompatibleDisallowOldStyleArgsAdd()
          && !SingleStringArgFormatter.isValid(formatStr)) {
        throw new EvalException(
            loc,
            String.format(
                "Invalid value for parameter \"%s\": Expected string with a single \"%%s\"",
                argumentName));
      }
    }

    private void addScalarArg(Object value, String format, BaseFunction mapFn, Location loc)
        throws EvalException {
      validateNoDirectory(value, loc);
      validateFormatString("format", format, loc);
      if (format == null && mapFn == null) {
        commandLine.add(value);
      } else {
        ScalarArg.Builder scalarArg =
            new ScalarArg.Builder(value).setLocation(loc).setFormat(format).setMapFn(mapFn);
        commandLine.add(scalarArg);
      }
    }

    private void validateNoDirectory(Object value, Location loc) throws EvalException {
      if (skylarkSemantics.incompatibleExpandDirectories() && isDirectory(value)) {
        throw new EvalException(
            loc,
            "Cannot add directories to Args#add since they may expand to multiple values. "
                + "Either use Args#add_all (if you want expansion) "
                + "or args.add(directory.path) (if you do not).");
      }
    }

    private static boolean isDirectory(Object object) {
      return ((object instanceof Artifact) && ((Artifact) object).isDirectory());
    }

    @Override
    public CommandLineArgsApi useParamsFile(String paramFileArg, Boolean useAlways)
        throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      if (!SingleStringArgFormatter.isValid(paramFileArg)) {
        throw new EvalException(
            null,
            String.format(
                "Invalid value for parameter \"param_file_arg\": "
                    + "Expected string with a single \"%s\"",
                paramFileArg));
      }
      this.flagFormatString = paramFileArg;
      this.useAlways = useAlways;
      return this;
    }

    @Override
    public CommandLineArgsApi setParamFileFormat(String format) throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      final ParameterFileType parameterFileType;
      switch (format) {
        case "shell":
          parameterFileType = ParameterFileType.SHELL_QUOTED;
          break;
        case "multiline":
          parameterFileType = ParameterFileType.UNQUOTED;
          break;
        default:
          throw new EvalException(
              null,
              "Invalid value for parameter \"format\": Expected one of \"shell\", \"multiline\"");
      }
      this.parameterFileType = parameterFileType;
      return this;
    }

    private Args(@Nullable Mutability mutability, SkylarkSemantics skylarkSemantics) {
      this.mutability = mutability != null ? mutability : Mutability.IMMUTABLE;
      this.skylarkSemantics = skylarkSemantics;
      this.commandLine = new SkylarkCustomCommandLine.Builder(skylarkSemantics);
    }

    public SkylarkCustomCommandLine build() {
      return commandLine.build();
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("context.args() object");
    }

    @Override
    public void debugPrint(SkylarkPrinter printer) {
      try {
        printer.append(Joiner.on(" ").join(commandLine.build().arguments()));
      } catch (CommandLineExpansionException e) {
        printer.append("Cannot expand command line: " + e.getMessage());
      }
    }

    ImmutableSet<Artifact> getDirectoryArtifacts() {
      for (Iterable<Object> collection : potentialDirectoryArtifacts) {
        scanForDirectories(collection);
      }
      potentialDirectoryArtifacts.clear();
      return ImmutableSet.copyOf(directoryArtifacts);
    }

    private void scanForDirectories(Iterable<?> objects) {
      for (Object object : objects) {
        if (isDirectory(object)) {
          directoryArtifacts.add((Artifact) object);
        }
      }
    }
  }

  @Override
  public Args args(Environment env) {
    return new Args(env.mutability(), skylarkSemantics);
  }

  @Override
  public boolean isImmutable() {
    return context.isImmutable();
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("actions for");
    context.repr(printer);
  }

  void nullify() {
    ruleContext = null;
  }
}
