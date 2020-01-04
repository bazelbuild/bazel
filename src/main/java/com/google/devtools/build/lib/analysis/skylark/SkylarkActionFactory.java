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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.BashCommandConstructor;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.StarlarkAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/** Provides a Skylark interface for all action creation needs. */
public class SkylarkActionFactory implements SkylarkActionFactoryApi {
  private final SkylarkRuleContext context;
  private final StarlarkSemantics starlarkSemantics;
  private RuleContext ruleContext;
  /** Counter for actions.run_shell helper scripts. Every script must have a unique name. */
  private int runShellOutputCounter = 0;

  public SkylarkActionFactory(
      SkylarkRuleContext context, StarlarkSemantics starlarkSemantics, RuleContext ruleContext) {
    this.context = context;
    this.starlarkSemantics = starlarkSemantics;
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
      public ActionLookupValue.ActionLookupKey getOwner() {
        return skylarkActionFactory
            .getActionConstructionContext()
            .getAnalysisEnvironment()
            .getOwner();
      }
    };
  }

  @Override
  public Artifact declareFile(String filename, Object sibling, Location loc) throws EvalException {
    context.checkMutable("actions.declare_file");

    PathFragment fragment;
    if (Starlark.NONE.equals(sibling)) {
      fragment = ruleContext.getPackageDirectory().getRelative(PathFragment.create(filename));
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      fragment = original.replaceName(filename);
    }

    if (!fragment.startsWith(ruleContext.getPackageDirectory())) {
      throw new EvalException(
          loc,
          String.format(
              "the output artifact '%s' is not under package directory '%s' for target '%s'",
              fragment, ruleContext.getPackageDirectory(), ruleContext.getLabel()));
    }
    return ruleContext.getDerivedArtifact(fragment, newFileRoot());
  }

  @Override
  public Artifact declareDirectory(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_directory");
    PathFragment fragment;

    if (Starlark.NONE.equals(sibling)) {
      fragment = ruleContext.getPackageDirectory().getRelative(PathFragment.create(filename));
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      fragment = original.replaceName(filename);
    }

    if (!fragment.startsWith(ruleContext.getPackageDirectory())) {
      throw new EvalException(
          String.format(
              "the output directory '%s' is not under package directory '%s' for target '%s'",
              fragment, ruleContext.getPackageDirectory(), ruleContext.getLabel()));
    }

    Artifact result = ruleContext.getTreeArtifact(fragment, newFileRoot());
    if (!result.isTreeArtifact()) {
      throw new EvalException(
          null,
          String.format(
              "'%s' has already been declared as a regular file, not directory.", filename));
    }
    return result;
  }

  @Override
  public Artifact declareSymlink(String filename, Object sibling, Location location)
      throws EvalException {
    context.checkMutable("actions.declare_symlink");

    if (!ruleContext.getConfiguration().allowUnresolvedSymlinks()) {
      throw new EvalException(
          location,
          "actions.declare_symlink() is not allowed; "
              + "use the --experimental_allow_unresolved_symlinks command line option");
    }

    Artifact result;
    PathFragment rootRelativePath;
    if (Starlark.NONE.equals(sibling)) {
      rootRelativePath = ruleContext.getPackageDirectory().getRelative(filename);
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      rootRelativePath = original.replaceName(filename);
    }

    result =
        ruleContext.getAnalysisEnvironment().getSymlinkArtifact(rootRelativePath, newFileRoot());

    if (!result.isSymlink()) {
      throw new EvalException(
          location,
          String.format(
              "'%s' has already been declared as something other than a symlink.", filename));
    }

    return result;
  }

  @Override
  public void doNothing(String mnemonic, Object inputs, Location location) throws EvalException {
    context.checkMutable("actions.do_nothing");
    NestedSet<Artifact> inputSet =
        inputs instanceof Depset
            ? ((Depset) inputs).getSetFromParam(Artifact.class, "inputs")
            : NestedSetBuilder.<Artifact>compileOrder()
                .addAll(((Sequence<?>) inputs).getContents(Artifact.class, "inputs"))
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
  public void symlink(FileApi output, String path, Location location) throws EvalException {
    context.checkMutable("actions.symlink");

    if (!ruleContext.getConfiguration().allowUnresolvedSymlinks()) {
      throw new EvalException(
          null,
          "actions.symlink() is not allowed; "
              + "use the --experimental_allow_unresolved_symlinks command line option");
    }

    PathFragment targetPath = PathFragment.create(path);
    Artifact outputArtifact = (Artifact) output;
    if (!outputArtifact.isSymlink()) {
      throw new EvalException(
          location, "output of symlink action must be created by declare_symlink()");
    }

    Action action =
        SymlinkAction.createUnresolved(
            ruleContext.getActionOwner(),
            outputArtifact,
            targetPath,
            "creating symlink " + ((Artifact) output).getRootRelativePathString());
    registerAction(location, action);
  }

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
              NestedSetBuilder.wrap(Order.STABLE_ORDER, args.getDirectoryArtifacts()),
              (Artifact) output,
              args.build(),
              args.getParameterFileType(),
              StandardCharsets.UTF_8);
    } else {
      throw new AssertionError("Unexpected type: " + content.getClass().getSimpleName());
    }
    registerAction(location, action);
  }

  @Override
  public void run(
      Sequence<?> outputs,
      Object inputs,
      Object unusedInputsList,
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
    StarlarkAction.Builder builder = new StarlarkAction.Builder();

    Sequence<?> argumentsList = ((Sequence) arguments);
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
    registerStarlarkAction(
        outputs,
        inputs,
        unusedInputsList,
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

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  @Override
  public void runShell(
      Sequence<?> outputs,
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
      Location location,
      StarlarkThread thread)
      throws EvalException {
    context.checkMutable("actions.run_shell");

    Sequence<?> argumentList = (Sequence) arguments;
    StarlarkAction.Builder builder = new StarlarkAction.Builder();
    buildCommandLine(builder, argumentList);

    if (commandUnchecked instanceof String) {
      Map<String, String> executionInfo =
          ImmutableMap.copyOf(TargetUtils.getExecutionInfo(ruleContext.getRule()));
      String helperScriptSuffix = String.format(".run_shell_%d.sh", runShellOutputCounter++);
      String command = (String) commandUnchecked;
      PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);
      BashCommandConstructor constructor =
          CommandHelper.buildBashCommandConstructor(
              executionInfo, shExecutable, helperScriptSuffix);
      Artifact helperScript =
          CommandHelper.commandHelperScriptMaybe(ruleContext, command, constructor);
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
    } else if (commandUnchecked instanceof Sequence) {
      if (thread.getSemantics().incompatibleRunShellCommandString()) {
        throw new EvalException(
            location,
            "'command' must be of type string. passing a sequence of strings as 'command'"
                + " is deprecated. To temporarily disable this check,"
                + " set --incompatible_objc_framework_cleanup=false.");
      }
      Sequence<?> commandList = (Sequence) commandUnchecked;
      if (argumentList.size() > 0) {
        throw new EvalException(location,
            "'arguments' must be empty if 'command' is a sequence of strings");
      }
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
    registerStarlarkAction(
        outputs,
        inputs,
        /*unusedInputsList=*/ Starlark.NONE,
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

  private static void buildCommandLine(SpawnAction.Builder builder, Sequence<?> argumentsList)
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
        ParamFileInfo paramFileInfo = args.getParamFileInfo();
        builder.addCommandLine(args.build(), paramFileInfo);
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
  private void registerStarlarkAction(
      Sequence<?> outputs,
      Object inputs,
      Object unusedInputsList,
      Object toolsUnchecked,
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location location,
      StarlarkAction.Builder builder)
      throws EvalException {
    Iterable<Artifact> inputArtifacts;
    if (inputs instanceof Sequence) {
      inputArtifacts = ((Sequence<?>) inputs).getContents(Artifact.class, "inputs");
      builder.addInputs(inputArtifacts);
    } else {
      NestedSet<Artifact> inputSet = ((Depset) inputs).getSetFromParam(Artifact.class, "inputs");
      builder.addTransitiveInputs(inputSet);
      inputArtifacts = inputSet.toList();
    }

    List<Artifact> outputArtifacts = outputs.getContents(Artifact.class, "outputs");
    if (outputArtifacts.isEmpty()) {
      throw new EvalException(location, "param 'outputs' may not be empty");
    }
    builder.addOutputs(outputArtifacts);

    if (unusedInputsList != Starlark.NONE) {
      if (!starlarkSemantics.experimentalStarlarkUnusedInputsList()) {
        throw new EvalException(
            location,
            "'unused_inputs_list' attribute is experimental and disabled by default. "
                + "This API is in development and subject to change at any time. "
                + "Use --experimental_starlark_unused_inputs_list to use this experimental API.");
      }
      if (unusedInputsList instanceof Artifact) {
        builder.setUnusedInputsList(Optional.of((Artifact) unusedInputsList));
      } else {
        throw new EvalException(
            location,
            "expected value of type 'File' for "
                + "a member of parameter 'unused_inputs_list' but got "
                + EvalUtils.getDataTypeName(unusedInputsList)
                + " instead");
      }
    }

    if (toolsUnchecked != Starlark.UNBOUND) {
      Iterable<?> toolsIterable;
      if (toolsUnchecked instanceof Sequence) {
        toolsIterable = ((Sequence<?>) toolsUnchecked).getContents(Object.class, "tools");
      } else {
        toolsIterable = ((Depset) toolsUnchecked).getSet().toList();
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
      if (starlarkSemantics.incompatibleNoSupportToolsInActionInputs()) {
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
    if (envUnchecked != Starlark.NONE) {
      builder.setEnvironment(
          ImmutableMap.copyOf(
              Dict.castSkylarkDictOrNoneToDict(envUnchecked, String.class, String.class, "env")));
    }
    if (progressMessage != Starlark.NONE) {
      builder.setProgressMessageNonLazy((String) progressMessage);
    }
    if (Starlark.truth(useDefaultShellEnv)) {
      builder.useDefaultShellEnvironment();
    }

    ImmutableMap<String, String> executionInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked,
            ruleContext.getRule(),
            starlarkSemantics.experimentalAllowTagsPropagation());
    builder.setExecutionInfo(executionInfo);

    if (inputManifestsUnchecked != Starlark.NONE) {
      for (RunfilesSupplier supplier :
          Sequence.castSkylarkListOrNoneToList(
              inputManifestsUnchecked, RunfilesSupplier.class, "runfiles suppliers")) {
        builder.addRunfilesSupplier(supplier);
      }
    }
    // Always register the action
    registerAction(location, builder.build(ruleContext));
  }

  private String getMnemonic(Object mnemonicUnchecked) {
    String mnemonic = mnemonicUnchecked == Starlark.NONE ? "Action" : (String) mnemonicUnchecked;
    if (ruleContext.getConfiguration().getReservedActionMnemonics().contains(mnemonic)) {
      mnemonic = mangleMnemonic(mnemonic);
    }
    return mnemonic;
  }

  private static String mangleMnemonic(String mnemonic) {
    return mnemonic + "FromStarlark";
  }

  @Override
  public void expandTemplate(
      FileApi template,
      FileApi output,
      Dict<?, ?> substitutionsUnchecked,
      Boolean executable,
      Location location)
      throws EvalException {
    context.checkMutable("actions.expand_template");
    ImmutableList.Builder<Substitution> substitutionsBuilder = ImmutableList.builder();
    for (Map.Entry<String, String> substitution :
        substitutionsUnchecked
            .getContents(String.class, String.class, "substitutions")
            .entrySet()) {
      // ParserInput.create(Path) uses Latin1 when reading BUILD files, which might
      // contain UTF-8 encoded symbols as part of template substitution.
      // As a quick fix, the substitution values are corrected before being passed on.
      // In the long term, fixing ParserInput.create(Path) would be a better approach.
      substitutionsBuilder.add(
          Substitution.of(substitution.getKey(), convertLatin1ToUtf8(substitution.getValue())));
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

  @Override
  public Args args(StarlarkThread thread) {
    return Args.newArgs(thread.mutability(), starlarkSemantics);
  }

  @Override
  public boolean isImmutable() {
    return context.isImmutable();
  }

  @Override
  public void repr(Printer printer) {
    printer.append("actions for");
    context.repr(printer);
  }

  void nullify() {
    ruleContext = null;
  }
}
