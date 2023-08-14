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
package com.google.devtools.build.lib.analysis.starlark;

import static com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext.PRIVATE_STARLARKIFICATION_ALLOWLIST;
import static com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.ResourceSetOrBuilder;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.BashCommandConstructor;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.BuildInfoFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.StarlarkAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.TemplateDictApi;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** Provides a Starlark interface for all action creation needs. */
public class StarlarkActionFactory implements StarlarkActionFactoryApi {
  private final StarlarkRuleContext context;
  /** Counter for actions.run_shell helper scripts. Every script must have a unique name. */
  private int runShellOutputCounter = 0;

  private static final ResourceSet DEFAULT_RESOURCE_SET = ResourceSet.createWithRamCpu(250, 1);
  private static final Set<String> validResources =
      new HashSet<>(Arrays.asList("cpu", "memory", "local_test"));

  // TODO(gnish): This is a temporary allowlist while new BuildInfo API becomes stable enough to
  // become public.
  // After at least some of the builtin rules have been switched to the new API delete this.
  private static final ImmutableSet<PackageIdentifier> PRIVATE_BUILDINFO_API_ALLOWLIST =
      ImmutableSet.of(
          PackageIdentifier.createInMainRepo("test"), // for tests
          PackageIdentifier.createInMainRepo("tools/build_defs/build_info"));

  public StarlarkActionFactory(StarlarkRuleContext context) {
    this.context = context;
  }

  ArtifactRoot newFileRoot() {
    return context.isForAspect()
        ? getRuleContext().getBinDirectory()
        : getRuleContext().getBinOrGenfilesDirectory();
  }

  /**
   * Returns a {@link ActionRegistry} object to register actions using this action factory.
   *
   * @throws EvalException if actions cannot be registered with this object
   */
  public ActionRegistry asActionRegistry(StarlarkActionFactory starlarkActionFactory)
      throws EvalException {
    validateActionCreation();
    return new ActionRegistry() {

      @Override
      public void registerAction(ActionAnalysisMetadata action) {
        getRuleContext().registerAction(action);
      }

      @Override
      public ActionLookupKey getOwner() {
        return starlarkActionFactory
            .getActionConstructionContext()
            .getAnalysisEnvironment()
            .getOwner();
      }
    };
  }

  private void checkToolchainParameterIsSet(Object toolchainUnchecked) throws EvalException {
    if (toolchainUnchecked == Starlark.UNBOUND) {
      throw Starlark.errorf(
          "Couldn't identify if tools are from implicit dependencies or a toolchain. Please"
              + " set the toolchain parameter. If you're not using a toolchain, set it to 'None'.");
    }
  }

  @Override
  public Artifact declareFile(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_file");
    RuleContext ruleContext = getRuleContext();

    PathFragment fragment;
    if (Starlark.NONE.equals(sibling)) {
      fragment = ruleContext.getPackageDirectory().getRelative(PathFragment.create(filename));
    } else {
      PathFragment original =
          ((Artifact) sibling)
              .getOutputDirRelativePath(
                  getSemantics().getBool(EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT));
      fragment = original.replaceName(filename);
    }

    if (!fragment.startsWith(ruleContext.getPackageDirectory())) {
      throw Starlark.errorf(
          "the output artifact '%s' is not under package directory '%s' for target '%s'",
          fragment, ruleContext.getPackageDirectory(), ruleContext.getLabel());
    }
    return ruleContext.getDerivedArtifact(fragment, newFileRoot());
  }

  @Override
  public Artifact declareDirectory(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_directory");
    RuleContext ruleContext = getRuleContext();
    PathFragment fragment;

    if (Starlark.NONE.equals(sibling)) {
      fragment = ruleContext.getPackageDirectory().getRelative(PathFragment.create(filename));
    } else {
      PathFragment original =
          ((Artifact) sibling)
              .getOutputDirRelativePath(
                  getSemantics().getBool(EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT));
      fragment = original.replaceName(filename);
    }

    if (!fragment.startsWith(ruleContext.getPackageDirectory())) {
      throw Starlark.errorf(
          "the output directory '%s' is not under package directory '%s' for target '%s'",
          fragment, ruleContext.getPackageDirectory(), ruleContext.getLabel());
    }

    Artifact result = ruleContext.getTreeArtifact(fragment, newFileRoot());
    if (!result.isTreeArtifact()) {
      throw Starlark.errorf(
          "'%s' has already been declared as a regular file, not directory.", filename);
    }
    return result;
  }

  @Override
  public Artifact declareSymlink(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_symlink");
    RuleContext ruleContext = getRuleContext();

    if (!ruleContext.getConfiguration().allowUnresolvedSymlinks()) {
      throw Starlark.errorf(
          "actions.declare_symlink() is not allowed; "
              + "use the --allow_unresolved_symlinks command line option");
    }

    Artifact result;
    PathFragment rootRelativePath;
    if (Starlark.NONE.equals(sibling)) {
      rootRelativePath = ruleContext.getPackageDirectory().getRelative(filename);
    } else {
      PathFragment original =
          ((Artifact) sibling)
              .getOutputDirRelativePath(
                  getSemantics().getBool(EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT));
      rootRelativePath = original.replaceName(filename);
    }

    result =
        ruleContext.getAnalysisEnvironment().getSymlinkArtifact(rootRelativePath, newFileRoot());

    if (!result.isSymlink()) {
      throw Starlark.errorf(
          "'%s' has already been declared as something other than a symlink.", filename);
    }

    return result;
  }

  @Override
  public void doNothing(String mnemonic, Object inputs) throws EvalException {
    context.checkMutable("actions.do_nothing");
    RuleContext ruleContext = getRuleContext();

    NestedSet<Artifact> inputSet =
        inputs instanceof Depset
            ? Depset.cast(inputs, Artifact.class, "inputs")
            : NestedSetBuilder.<Artifact>compileOrder()
                .addAll(Sequence.cast(inputs, Artifact.class, "inputs"))
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
    registerAction(action);
  }

  @SerializationConstant @AutoCodec.VisibleForSerialization
  static final GeneratedMessage.GeneratedExtension<ExtraActionInfo, SpawnInfo> SPAWN_INFO =
      SpawnInfo.spawnInfo;

  @Override
  public void symlink(
      FileApi output,
      Object /* Artifact or None */ targetFile,
      Object /* String or None */ targetPath,
      Boolean isExecutable,
      Object /* String or None */ progressMessageUnchecked)
      throws EvalException {
    context.checkMutable("actions.symlink");
    RuleContext ruleContext = getRuleContext();

    if ((targetFile == Starlark.NONE) == (targetPath == Starlark.NONE)) {
      throw Starlark.errorf("Exactly one of \"target_file\" or \"target_path\" is required");
    }

    Artifact outputArtifact = (Artifact) output;
    String progressMessage =
        (progressMessageUnchecked != Starlark.NONE)
            ? (String) progressMessageUnchecked
            : "Creating symlink %{output}";

    Action action;
    if (targetFile != Starlark.NONE) {
      Artifact inputArtifact = (Artifact) targetFile;
      if (outputArtifact.isSymlink()) {
        throw Starlark.errorf(
            "symlink() with \"target_file\" param requires that \"output\" be declared as a "
                + "file or directory, not a symlink (did you mean to use declare_file() or "
                + "declare_directory() instead of declare_symlink()?)");
      }

      if (inputArtifact.isDirectory() != outputArtifact.isDirectory()) {
        String inputType = inputArtifact.isDirectory() ? "directory" : "file";
        String outputType = outputArtifact.isDirectory() ? "directory" : "file";
        throw Starlark.errorf(
            "symlink() with \"target_file\" %s param requires that \"output\" be declared as a %s "
                + "(did you mean to use declare_%s() instead of declare_%s()?)",
            inputType, inputType, inputType, outputType);
      }

      if (isExecutable) {
        if (outputArtifact.isTreeArtifact()) {
          throw Starlark.errorf("symlink() with \"output\" directory param cannot be executable");
        }
        action =
            SymlinkAction.toExecutable(
                ruleContext.getActionOwner(), inputArtifact, outputArtifact, progressMessage);
      } else {
        action =
            SymlinkAction.toArtifact(
                ruleContext.getActionOwner(), inputArtifact, outputArtifact, progressMessage);
      }
    } else {
      if (!outputArtifact.isSymlink()) {
        throw Starlark.errorf(
            "symlink() with \"target_path\" param requires that \"output\" be declared as a "
                + "symlink, not a file or directory (did you mean to use declare_symlink() instead "
                + "of declare_file() or declare_directory()?)");
      }

      if (isExecutable) {
        throw Starlark.errorf("\"is_executable\" cannot be True when using \"target_path\"");
      }

      action =
          UnresolvedSymlinkAction.create(
              ruleContext.getActionOwner(), outputArtifact, (String) targetPath, progressMessage);
    }
    registerAction(action);
  }

  @Override
  public void write(FileApi output, Object content, Boolean isExecutable) throws EvalException {
    context.checkMutable("actions.write");
    RuleContext ruleContext = getRuleContext();

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
              args.getParameterFileType());
    } else {
      throw new AssertionError("Unexpected type: " + content.getClass().getSimpleName());
    }
    registerAction(action);
  }

  @Override
  public void run(
      Sequence<?> outputs,
      Object inputs,
      Object unusedInputsList,
      Object executableUnchecked,
      Object toolsUnchecked,
      Sequence<?> arguments,
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Object execGroupUnchecked,
      Object shadowedActionUnchecked,
      Object resourceSetUnchecked,
      Object toolchainUnchecked)
      throws EvalException {
    context.checkMutable("actions.run");

    RuleContext ruleContext = getRuleContext();
    boolean useAutoExecGroups = ruleContext.useAutoExecGroups();

    StarlarkAction.Builder builder = new StarlarkAction.Builder();
    buildCommandLine(builder, arguments);
    if (executableUnchecked instanceof Artifact) {
      Artifact executable = (Artifact) executableUnchecked;
      FilesToRunProvider provider = context.getExecutableRunfiles(executable);
      if (provider == null) {
        if (useAutoExecGroups && execGroupUnchecked == Starlark.NONE) {
          checkToolchainParameterIsSet(toolchainUnchecked);
        }
        builder.setExecutable(executable);
      } else {
        builder.setExecutable(provider);
      }
    } else if (executableUnchecked instanceof String) {
      // Normalise if needed and then pass as a String; this keeps the reference when PathFragment
      // is passed from native to Starlark
      builder.setExecutableAsString(
          PathFragment.create((String) executableUnchecked).getPathString());
    } else if (executableUnchecked instanceof FilesToRunProvider) {
      if (useAutoExecGroups
          && !context.areRunfilesFromDeps((FilesToRunProvider) executableUnchecked)
          && execGroupUnchecked == Starlark.NONE) {
        checkToolchainParameterIsSet(toolchainUnchecked);
      }
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
        execGroupUnchecked,
        shadowedActionUnchecked,
        resourceSetUnchecked,
        toolchainUnchecked,
        builder);
  }

  @Override
  public Artifact transformVersionFile(
      Object transformFuncObject,
      Object templateObject,
      String outputFileName,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    BuiltinRestriction.failIfCalledOutsideAllowlist(thread, PRIVATE_BUILDINFO_API_ALLOWLIST);
    return transformBuildInfoFile(
        transformFuncObject, templateObject, outputFileName, true, thread);
  }

  @Override
  public Artifact transformInfoFile(
      Object transformFuncObject,
      Object templateObject,
      String outputFileName,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    BuiltinRestriction.failIfCalledOutsideAllowlist(thread, PRIVATE_BUILDINFO_API_ALLOWLIST);
    return transformBuildInfoFile(
        transformFuncObject, templateObject, outputFileName, false, thread);
  }

  private Artifact transformBuildInfoFile(
      Object transformFuncObject,
      Object templateObject,
      String outputFileName,
      boolean isVolatile,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    RuleContext ruleContext = getRuleContext();
    Artifact templateFile = (Artifact) templateObject;
    PathFragment fragment =
        ruleContext.getPackageDirectory().getRelative(PathFragment.create(outputFileName));
    Artifact buildInfoFile =
        isVolatile
            ? ruleContext
                .getAnalysisEnvironment()
                .getConstantMetadataArtifact(fragment, newFileRoot())
            : ruleContext.getDerivedArtifact(fragment, newFileRoot());
    StarlarkFunction translationFunc = (StarlarkFunction) transformFuncObject;
    BuildInfoFileWriteAction action =
        new BuildInfoFileWriteAction(
            ruleContext.getActionOwner(),
            isVolatile
                ? ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact()
                : ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact(),
            buildInfoFile,
            translationFunc,
            templateFile,
            isVolatile,
            thread.getSemantics());
    registerAction(action);
    return buildInfoFile;
  }

  private void validateActionCreation() throws EvalException {
    if (getRuleContext().getRule().isAnalysisTest()) {
      throw Starlark.errorf(
          "implementation function of a rule with "
              + "analysis_test=true may not register actions. Analysis test rules may only return "
              + "success/failure information via AnalysisTestResultInfo.");
    }
  }

  /**
   * Registers action in the context of this {@link StarlarkActionFactory}.
   *
   * <p>Use {@link #getActionConstructionContext()} to obtain the context required to create this
   * action.
   */
  public void registerAction(ActionAnalysisMetadata action) throws EvalException {
    validateActionCreation();
    getRuleContext().registerAction(action);
  }

  /**
   * Returns information needed to construct actions that can be registered with {@link
   * #registerAction}.
   */
  public ActionConstructionContext getActionConstructionContext() {
    return context.getRuleContext();
  }

  public RuleContext getRuleContext() {
    return context.getRuleContext();
  }

  private StarlarkSemantics getSemantics() {
    return context.getStarlarkSemantics();
  }

  private void verifyExecGroupExists(String execGroup, RuleContext ctx) throws EvalException {
    if (!ctx.hasToolchainContext(execGroup)) {
      throw Starlark.errorf("Action declared for non-existent exec group '%s'.", execGroup);
    }
  }

  private void verifyAutomaticExecGroupExists(String execGroup, RuleContext ruleContext)
      throws EvalException {
    if (!ruleContext.hasToolchainContext(execGroup)) {
      throw Starlark.errorf("Action declared for non-existent toolchain '%s'.", execGroup);
    }
  }

  private void checkValidGroupName(String execGroup) throws EvalException {
    if (!StarlarkExecGroupCollection.isValidGroupName(execGroup)) {
      throw Starlark.errorf("Invalid name for exec group '%s'.", execGroup);
    }
  }

  private PlatformInfo getExecutionPlatform(Object execGroupUnchecked, RuleContext ctx)
      throws EvalException {
    if (execGroupUnchecked == Starlark.NONE) {
      return ctx.getExecutionPlatform(ExecGroup.DEFAULT_EXEC_GROUP_NAME);
    } else {
      String execGroup = (String) execGroupUnchecked;
      verifyExecGroupExists(execGroup, ctx);
      checkValidGroupName(execGroup);
      return ctx.getExecutionPlatform((String) execGroupUnchecked);
    }
  }

  @Override
  public void runShell(
      Sequence<?> outputs,
      Object inputs,
      Object toolsUnchecked,
      Sequence<?> arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Object execGroupUnchecked,
      Object shadowedActionUnchecked,
      Object resourceSetUnchecked,
      Object toolchainUnchecked)
      throws EvalException {
    context.checkMutable("actions.run_shell");
    RuleContext ruleContext = getRuleContext();

    StarlarkAction.Builder builder = new StarlarkAction.Builder();
    buildCommandLine(builder, arguments);

    // When we use a shell command, add an empty argument before other arguments.
    //   e.g.  bash -c "cmd" '' 'arg1' 'arg2'
    // bash will use the empty argument as the value of $0 (which we don't care about).
    // arg1 and arg2 will be $1 and $2, as a user expects.
    boolean pad = !arguments.isEmpty();

    if (commandUnchecked instanceof String) {
      ImmutableMap<String, String> executionInfo =
          ImmutableMap.copyOf(TargetUtils.getExecutionInfo(ruleContext.getRule()));
      String helperScriptSuffix = String.format(".run_shell_%d.sh", runShellOutputCounter++);
      String command = (String) commandUnchecked;
      PathFragment shExecutable =
          ShToolchain.getPathForPlatform(
              ruleContext.getConfiguration(),
              getExecutionPlatform(execGroupUnchecked, ruleContext));
      BashCommandConstructor constructor =
          CommandHelper.buildBashCommandConstructor(
              executionInfo, shExecutable, helperScriptSuffix);
      Artifact helperScript =
          CommandHelper.commandHelperScriptMaybe(ruleContext, command, constructor);
      if (helperScript == null) {
        builder.setShellCommand(shExecutable, command, pad);
      } else {
        builder.setShellCommand(shExecutable, helperScript.getExecPathString(), pad);
        builder.addInput(helperScript);
      }
    } else if (commandUnchecked instanceof Sequence) {
      if (getSemantics().getBool(BuildLanguageOptions.INCOMPATIBLE_RUN_SHELL_COMMAND_STRING)) {
        throw Starlark.errorf(
            "'command' must be of type string. passing a sequence of strings as 'command'"
                + " is deprecated. To temporarily disable this check,"
                + " set --incompatible_run_shell_command_string=false.");
      }
      Sequence<?> commandList = (Sequence) commandUnchecked;
      if (!arguments.isEmpty()) {
        throw Starlark.errorf("'arguments' must be empty if 'command' is a sequence of strings");
      }
      List<String> command = Sequence.cast(commandList, String.class, "command");
      builder.setShellCommand(command, pad);
    } else {
      throw Starlark.errorf(
          "expected string or list of strings for command instead of %s",
          Starlark.type(commandUnchecked));
    }
    registerStarlarkAction(
        outputs,
        inputs,
        /* unusedInputsList= */ Starlark.NONE,
        toolsUnchecked,
        mnemonicUnchecked,
        progressMessage,
        useDefaultShellEnv,
        envUnchecked,
        executionRequirementsUnchecked,
        inputManifestsUnchecked,
        execGroupUnchecked,
        shadowedActionUnchecked,
        resourceSetUnchecked,
        toolchainUnchecked,
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
        throw Starlark.errorf(
            "expected list of strings or ctx.actions.args() for arguments instead of %s",
            Starlark.type(value));
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
      Object execGroupUnchecked,
      Object shadowedActionUnchecked,
      Object resourceSetUnchecked,
      Object toolchainUnchecked,
      StarlarkAction.Builder builder)
      throws EvalException {
    if (inputs instanceof Sequence) {
      builder.addInputs(Sequence.cast(inputs, Artifact.class, "inputs"));
    } else {
      builder.addTransitiveInputs(Depset.cast(inputs, Artifact.class, "inputs"));
    }

    List<Artifact> outputArtifacts = Sequence.cast(outputs, Artifact.class, "outputs");
    if (outputArtifacts.isEmpty()) {
      throw Starlark.errorf("param 'outputs' may not be empty");
    }
    builder.addOutputs(outputArtifacts);

    if (unusedInputsList != Starlark.NONE) {
      if (unusedInputsList instanceof Artifact) {
        builder.setUnusedInputsList(Optional.of((Artifact) unusedInputsList));
      } else {
        throw Starlark.errorf(
            "expected value of type 'File' for a member of parameter 'unused_inputs_list' but got"
                + " %s instead",
            Starlark.type(unusedInputsList));
      }
    }

    RuleContext ruleContext = getRuleContext();
    boolean useAutoExecGroups = ruleContext.useAutoExecGroups();

    if (toolsUnchecked != Starlark.UNBOUND) {
      List<?> tools =
          toolsUnchecked instanceof Sequence
              ? Sequence.cast(toolsUnchecked, Object.class, "tools")
              : Depset.cast(toolsUnchecked, Object.class, "tools").toList();

      for (Object toolUnchecked : tools) {
        if (toolUnchecked instanceof Artifact) {
          Artifact artifact = (Artifact) toolUnchecked;
          builder.addTool(artifact);
          FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
          if (provider != null) {
            builder.addTool(provider);
          } else {
            if (useAutoExecGroups && execGroupUnchecked == Starlark.NONE) {
              checkToolchainParameterIsSet(toolchainUnchecked);
            }
          }
        } else if (toolUnchecked instanceof FilesToRunProvider) {
          if (useAutoExecGroups
              && !context.areRunfilesFromDeps((FilesToRunProvider) toolUnchecked)
              && execGroupUnchecked == Starlark.NONE) {
            checkToolchainParameterIsSet(toolchainUnchecked);
          }
          builder.addTool((FilesToRunProvider) toolUnchecked);
        } else if (toolUnchecked instanceof Depset) {
          try {
            if (useAutoExecGroups && execGroupUnchecked == Starlark.NONE) {
              checkToolchainParameterIsSet(toolchainUnchecked);
            }
            builder.addTransitiveTools(((Depset) toolUnchecked).getSet(Artifact.class));
          } catch (TypeException e) {
            throw Starlark.errorf(
                "expected value of type 'File, FilesToRunProvider or Depset of Files' for a member "
                    + "of parameter 'tools' but %s",
                e.getMessage());
          }
        } else {
          throw Starlark.errorf(
              "expected value of type 'File, FilesToRunProvider or Depset of Files' for a member of"
                  + " parameter 'tools' but got %s instead",
              Starlark.type(toolUnchecked));
        }
      }
    }

    String mnemonic = getMnemonic(mnemonicUnchecked);
    try {
      builder.setMnemonic(mnemonic);
    } catch (IllegalArgumentException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
    if (envUnchecked != Starlark.NONE) {
      builder.setEnvironment(
          ImmutableMap.copyOf(Dict.cast(envUnchecked, String.class, String.class, "env")));
    }
    if (progressMessage != Starlark.NONE) {
      builder.setProgressMessageFromStarlark((String) progressMessage);
    }
    if (Starlark.truth(useDefaultShellEnv)) {
      builder.useDefaultShellEnvironment();
    }

    ImmutableMap<String, String> executionInfo =
        TargetUtils.getFilteredExecutionInfo(
            executionRequirementsUnchecked,
            ruleContext.getRule(),
            getSemantics().getBool(BuildLanguageOptions.INCOMPATIBLE_ALLOW_TAGS_PROPAGATION));
    builder.setExecutionInfo(executionInfo);

    if (inputManifestsUnchecked != Starlark.NONE) {
      for (RunfilesSupplier supplier :
          Sequence.cast(inputManifestsUnchecked, RunfilesSupplier.class, "runfiles suppliers")) {
        builder.addRunfilesSupplier(supplier);
      }
    }

    Label toolchainLabel = null;
    if (toolchainUnchecked instanceof Label) {
      toolchainLabel = (Label) toolchainUnchecked;
    } else if (toolchainUnchecked instanceof String) {
      try {
        toolchainLabel =
            Label.parseWithPackageContext(
                (String) toolchainUnchecked, ruleContext.getPackageContext());
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }

    if (execGroupUnchecked != Starlark.NONE) {
      String execGroup = (String) execGroupUnchecked;
      verifyExecGroupExists(execGroup, ruleContext);
      checkValidGroupName(execGroup);

      // If toolchain and exec_groups are both defined, verify they are compatible.
      if (useAutoExecGroups && toolchainLabel != null) {
        if (ruleContext.getExecGroups().getExecGroup(execGroup).toolchainTypes().stream()
            .map(ToolchainTypeRequirement::toolchainType)
            .noneMatch(toolchainLabel::equals)) {
          throw Starlark.errorf(
              "`toolchain` and `exec_group` parameters inside actions.{run, run_shell} are not"
                  + " compatible; use one of them or define `toolchain` which is compatible with"
                  + " the exec_group (already exists inside the `exec_group`)");
        }
      }

      builder.setExecGroup(execGroup);
    } else if (useAutoExecGroups && toolchainLabel != null) {
      verifyAutomaticExecGroupExists(toolchainLabel.toString(), ruleContext);
      builder.setExecGroup(toolchainLabel.toString());
    } else {
      builder.setExecGroup(ExecGroup.DEFAULT_EXEC_GROUP_NAME);
    }

    if (shadowedActionUnchecked != Starlark.NONE) {
      builder.setShadowedAction(Optional.of((Action) shadowedActionUnchecked));
    }

    if (getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_ACTION_RESOURCE_SET)
        && resourceSetUnchecked != Starlark.NONE) {
      validateResourceSetBuilder(resourceSetUnchecked);
      builder.setResources(
          new StarlarkActionResourceSetBuilder(
              (StarlarkCallable) resourceSetUnchecked, mnemonic, getSemantics()));
    }

    // Always register the action
    registerAction(builder.build(ruleContext));
  }

  private static class StarlarkActionResourceSetBuilder implements ResourceSetOrBuilder {
    private final StarlarkCallable fn;
    private final String mnemonic;
    private final StarlarkSemantics semantics;

    public StarlarkActionResourceSetBuilder(
        StarlarkCallable fn, String mnemonic, StarlarkSemantics semantics) {
      this.fn = fn;
      this.mnemonic = mnemonic;
      this.semantics = semantics;
    }

    @Override
    public ResourceSet buildResourceSet(OS os, int inputsSize) throws ExecException {
      try (Mutability mu = Mutability.create("resource_set_builder_function")) {
        StarlarkThread thread = new StarlarkThread(mu, semantics);
        StarlarkInt inputInt = StarlarkInt.of(inputsSize);
        Object response =
            Starlark.call(
                thread,
                this.fn,
                ImmutableList.of(os.getCanonicalName(), inputInt),
                ImmutableMap.of());
        Map<String, Object> resourceSetMapRaw =
            Dict.cast(response, String.class, Object.class, "resource_set");

        if (!validResources.containsAll(resourceSetMapRaw.keySet())) {
          String message =
              String.format(
                  "Illegal resource keys: (%s)",
                  Joiner.on(",").join(Sets.difference(resourceSetMapRaw.keySet(), validResources)));
          throw new EvalException(message);
        }

        return ResourceSet.create(
            getNumericOrDefault(resourceSetMapRaw, "memory", DEFAULT_RESOURCE_SET.getMemoryMb()),
            getNumericOrDefault(resourceSetMapRaw, "cpu", DEFAULT_RESOURCE_SET.getCpuUsage()),
            (int)
                getNumericOrDefault(
                    resourceSetMapRaw,
                    "local_test",
                    (double) DEFAULT_RESOURCE_SET.getLocalTestCount()));
      } catch (EvalException e) {
        throw new UserExecException(
            FailureDetail.newBuilder()
                .setMessage(
                    String.format("Could not build resources for %s. %s", mnemonic, e.getMessage()))
                .setStarlarkAction(
                    FailureDetails.StarlarkAction.newBuilder()
                        .setCode(FailureDetails.StarlarkAction.Code.STARLARK_ACTION_UNKNOWN)
                        .build())
                .build());
      } catch (InterruptedException e) {
        throw new UserExecException(
            FailureDetail.newBuilder()
                .setMessage(e.getMessage())
                .setInterrupted(
                    Interrupted.newBuilder().setCode(Interrupted.Code.INTERRUPTED).build())
                .build());
      }
    }

    private static double getNumericOrDefault(
        Map<String, Object> resourceSetMap, String key, double defaultValue) throws EvalException {
      if (!resourceSetMap.containsKey(key)) {
        return defaultValue;
      }

      Object value = resourceSetMap.get(key);
      if (value instanceof StarlarkInt) {
        return ((StarlarkInt) value).toDouble();
      }

      if (value instanceof StarlarkFloat) {
        return ((StarlarkFloat) value).toDouble();
      }
      throw new EvalException(
          String.format(
              "Illegal resource value type for key %s: got %s, want int or float",
              key, Starlark.type(value)));
    }
  }

  private static StarlarkCallable validateResourceSetBuilder(Object fn) throws EvalException {
    if (!(fn instanceof StarlarkCallable)) {
      throw Starlark.errorf(
          "resource_set should be a Starlark-callable function, but got %s instead",
          Starlark.type(fn));
    }

    if (fn instanceof StarlarkFunction) {
      StarlarkFunction sfn = (StarlarkFunction) fn;

      // Reject non-global functions, because arbitrary closures may cause large
      // analysis-phase data structures to remain live into the execution phase.
      // We require that the function is "global" as opposed to "not a closure"
      // because a global function may be closure if it refers to load bindings.
      // This unfortunately disallows such trivially safe non-global
      // functions as "lambda x: x".
      // See https://github.com/bazelbuild/bazel/issues/12701.
      if (sfn.getModule().getGlobal(sfn.getName()) != sfn) {
        throw Starlark.errorf(
            "to avoid unintended retention of analysis data structures, "
                + "the resource_set function (declared at %s) must be declared "
                + "by a top-level def statement",
            sfn.getLocation());
      }
    }
    return (StarlarkCallable) fn;
  }

  private String getMnemonic(Object mnemonicUnchecked) {
    String mnemonic = mnemonicUnchecked == Starlark.NONE ? "Action" : (String) mnemonicUnchecked;
    if (getRuleContext().getConfiguration().getReservedActionMnemonics().contains(mnemonic)) {
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
      /* TemplateDict */ Object computedSubstitutions)
      throws EvalException {
    context.checkMutable("actions.expand_template");
    // We use a map to check for duplicate keys
    ImmutableMap.Builder<String, Substitution> substitutionsBuilder = ImmutableMap.builder();
    for (Map.Entry<String, String> substitution :
        Dict.cast(substitutionsUnchecked, String.class, String.class, "substitutions").entrySet()) {
      // Blaze calls ParserInput.fromLatin1 when reading BUILD files, which might
      // contain UTF-8 encoded symbols as part of template substitution.
      // As a quick fix, the substitution values are corrected before being passed on.
      // In the long term, avoiding ParserInput.fromLatin would be a better approach.
      substitutionsBuilder.put(
          substitution.getKey(),
          Substitution.of(substitution.getKey(), convertLatin1ToUtf8(substitution.getValue())));
    }
    if (!Starlark.UNBOUND.equals(computedSubstitutions)) {
      for (Substitution substitution : ((TemplateDict) computedSubstitutions).getAll()) {
        substitutionsBuilder.put(substitution.getKey(), substitution);
      }
    }
    ImmutableMap<String, Substitution> substitutionMap;
    try {
      substitutionMap = substitutionsBuilder.buildOrThrow();
    } catch (IllegalArgumentException e) {
      // user added duplicate keys, report the error, but the stack trace is not of use
      throw Starlark.errorf("%s", e.getMessage());
    }
    TemplateExpansionAction action =
        new TemplateExpansionAction(
            getRuleContext().getActionOwner(),
            (Artifact) template,
            (Artifact) output,
            substitutionMap.values().asList(),
            executable);
    registerAction(action);
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
    return Args.newArgs(thread.mutability(), getSemantics());
  }

  @Override
  public TemplateDictApi templateDict() {
    return TemplateDict.newDict();
  }

  @Override
  public FileApi createShareableArtifact(String path, Object artifactRoot, StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideAllowlist(thread, PRIVATE_STARLARKIFICATION_ALLOWLIST);
    ArtifactRoot root =
        artifactRoot == Starlark.UNBOUND
            ? getRuleContext().getBinDirectory()
            : (ArtifactRoot) artifactRoot;
    return getRuleContext().getShareableArtifact(PathFragment.create(path), root);
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
}
