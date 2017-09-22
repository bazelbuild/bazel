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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ParamFileInfo;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.skylark.SkylarkCustomCommandLine.ScalarArg;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Provides a Skylark interface for all action creation needs.
 */
@SkylarkModule(
    name = "actions",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Module providing functions to create actions."
)

public class SkylarkActionFactory implements SkylarkValue {
  private final SkylarkRuleContext context;
  private final SkylarkSemanticsOptions skylarkSemanticsOptions;
  private RuleContext ruleContext;
  /** Counter for actions.run_shell helper scripts. Every script must have a unique name. */
  private int runShellOutputCounter = 0;

  public SkylarkActionFactory(
      SkylarkRuleContext context,
      SkylarkSemanticsOptions skylarkSemanticsOptions,
      RuleContext ruleContext) {
    this.context = context;
    this.skylarkSemanticsOptions = skylarkSemanticsOptions;
    this.ruleContext = ruleContext;
  }

  Root newFileRoot() throws EvalException {
    return context.isForAspect()
        ? ruleContext.getConfiguration().getBinDirectory(ruleContext.getRule().getRepository())
        : ruleContext.getBinOrGenfilesDirectory();
  }

  @SkylarkCallable(
      name = "declare_file",
      doc =
          "Declares that rule or aspect creates a file with the given filename. "
              + "If <code>sibling</code> is not specified, file name is relative to "
              + "package directory, otherwise the file is in the same directory as "
              + "<code>sibling</code>. "
              + "You must create an action that generates the file. <br>"
              + "Files that are specified in rule's outputs do not need to be declared and are "
              + "available through <a href=\"ctx.html#outputs\">ctx.outputs</a>.",
      parameters = {
          @Param(
              name = "filename",
              type = String.class,
              doc =
                  "If no 'sibling' provided, path of the new file, relative "
                  + "to the current package. Otherwise a base name for a file "
                   + "('sibling' determines a directory)."
          ),
          @Param(
              name = "sibling",
              doc = "A file that lives in the same directory as the newly created file.",
              type = Artifact.class,
              noneable = true,
              positional = false,
              named = true,
              defaultValue = "None"
          )
      }
  )
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

  @SkylarkCallable(
      name = "declare_directory",
      doc =
          "Declares that rule or aspect create a directory with the given name, in the "
              + "current package. You must create an action that generates the directory.",
      parameters = {
          @Param(
              name = "filename",
              type = String.class,
              doc =
                  "If no 'sibling' provided, path of the new directory, relative "
                      + "to the current package. Otherwise a base name for a file "
                      + "('sibling' defines a directory)."
          ),
          @Param(
              name = "sibling",
              doc = "A file that lives in the same directory as the newly declared directory.",
              type = Artifact.class,
              noneable = true,
              positional = false,
              named = true,
              defaultValue = "None"
          )
      }
  )
  public Artifact declareDirectory(String filename, Object sibling) throws EvalException {
    context.checkMutable("actions.declare_directory");
    if (Runtime.NONE.equals(sibling)) {
      return ruleContext.getPackageRelativeTreeArtifact(
          PathFragment.create(filename), newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      return ruleContext.getTreeArtifact(fragment, newFileRoot());
    }
  }


  @SkylarkCallable(
      name = "do_nothing",
      doc =
          "Creates an empty action that neither executes a command nor produces any "
              + "output, but that is useful for inserting 'extra actions'.",
      parameters = {
          @Param(
              name = "mnemonic",
              type = String.class,
              named = true,
              positional = false,
              doc = "a one-word description of the action, e.g. CppCompile or GoLink."
          ),
          @Param(
              name = "inputs",
              allowedTypes = {
                  @ParamType(type = SkylarkList.class),
                  @ParamType(type = SkylarkNestedSet.class),
              },
              generic1 = Artifact.class,
              named = true,
              positional = false,
              defaultValue = "[]",
              doc = "list of the input files of the action."
          ),
      }
  )
  public void doNothing(String mnemonic, Object inputs) throws EvalException {
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
            SpawnInfo.spawnInfo,
            SpawnInfo.newBuilder().build());
    ruleContext.registerAction(action);
  }

  @SkylarkCallable(
    name = "write",
    doc =
        "Creates a file write action. When the action is executed, it will write the given content "
            + "to a file. This is used to generate files using information available in the "
            + "analysis phase. If the file is large and with a lot of static content, consider "
            + "using <a href=\"#expand_template\">expand_template</a>. "
            + "<a href=\"https://github.com/laurentlb/examples/blob/master/rules/executable/executable.bzl\">"
            + "See example of use</a>",
    parameters = {
      @Param(name = "output", type = Artifact.class, doc = "the output file.", named = true),
      @Param(
        name = "content",
        type = Object.class,
        allowedTypes = {@ParamType(type = String.class), @ParamType(type = Args.class)},
        doc =
            "the contents of the file. "
                + "May be a either a string or actions.args() object. "
                + "See <a href=\"actions.html#args\">ctx.actions.args()</a>.",
        named = true
      ),
      @Param(
        name = "is_executable",
        type = Boolean.class,
        defaultValue = "False",
        doc = "whether the output file should be executable.",
        named = true
      )
    }
  )
  public void write(Artifact output, Object content, Boolean isExecutable) throws EvalException {
    context.checkMutable("actions.write");
    final Action action;
    if (content instanceof String) {
      action = FileWriteAction.create(ruleContext, output, (String) content, isExecutable);
    } else if (content instanceof Args) {
      Args args = (Args) content;
      action =
          new ParameterFileWriteAction(
              ruleContext.getActionOwner(),
              output,
              args.build(),
              args.parameterFileType,
              StandardCharsets.UTF_8);
    } else {
      throw new AssertionError("Unexpected type: " + content.getClass().getSimpleName());
    }
    ruleContext.registerAction(action);
  }

  @SkylarkCallable(
    name = "run",
    doc = "Creates an action that runs an executable.",
    parameters = {
      @Param(
        name = "outputs",
        type = SkylarkList.class,
        generic1 = Artifact.class,
        named = true,
        positional = false,
        doc = "list of the output files of the action."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "list of the input files of the action."
      ),
      @Param(
        name = "executable",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = Artifact.class),
          @ParamType(type = String.class),
        },
        named = true,
        positional = false,
        doc = "the executable file to be called by the action."
      ),
      @Param(
        name = "arguments",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
        },
        defaultValue = "[]",
        named = true,
        positional = false,
        doc =
            "command line arguments of the action. "
                + "Must be a list of strings or actions.args() objects. "
                + "See <a href=\"actions.html#args\">ctx.actions.args()</a>."
      ),
      @Param(
        name = "mnemonic",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "a one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "progress_message",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "progress message to show to the user during the build, "
                + "e.g. \"Compiling foo.cc to create foo.o\"."
      ),
      @Param(
        name = "use_default_shell_env",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "whether the action should use the built in shell environment or not."
      ),
      @Param(
        name = "env",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "sets the dictionary of environment variables."
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "information for scheduling the action. See "
                + "<a href=\"/docs/be/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
      @Param(
        // TODO(bazel-team): The name here isn't accurate anymore.
        // This is technically experimental, so folks shouldn't be too attached,
        // but consider renaming to be more accurate/opaque.
        name = "input_manifests",
        type = SkylarkList.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "(Experimental) sets the input runfiles metadata; "
                + "they are typically generated by resolve_command."
      )
    }
  )
  public void run(
      SkylarkList outputs,
      Object inputs,
      Object executableUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked)
      throws EvalException {
    context.checkMutable("actions.run");
    SpawnAction.Builder builder = new SpawnAction.Builder();

    SkylarkList argumentsList = ((SkylarkList) arguments);
    buildCommandLine(builder, argumentsList);
    if (executableUnchecked instanceof Artifact) {
      Artifact executable = (Artifact) executableUnchecked;
      builder.addInput(executable);
      FilesToRunProvider provider = context.getExecutableRunfiles(executable);
      if (provider == null) {
        builder.setExecutable(executable);
      } else {
        builder.setExecutable(provider);
      }
    } else if (executableUnchecked instanceof String) {
      builder.setExecutable(PathFragment.create((String) executableUnchecked));
    } else {
      throw new EvalException(
          null,
          "expected file or string for "
              + "executable but got "
              + EvalUtils.getDataTypeName(executableUnchecked)
              + " instead");
    }
    registerSpawnAction(outputs, inputs, mnemonicUnchecked, progressMessage, useDefaultShellEnv,
        envUnchecked, executionRequirementsUnchecked, inputManifestsUnchecked, builder);
  }

  /**
   * Registers actions in the context of this {@link SkylarkActionFactory}.
   *
   * Use {@link #getActionConstructionContext()} to obtain the context required to
   * create those actions.
   */
  public void registerAction(ActionAnalysisMetadata... actions) {
    ruleContext.registerAction(actions);
  }

  /**
   * Returns information needed to construct actions that can be
   * registered with {@link #registerAction(ActionAnalysisMetadata...)}.
   */
  public ActionConstructionContext getActionConstructionContext() {
    return ruleContext;
  }

  @SkylarkCallable(
    name = "run_shell",
    doc = "Creates an action that runs a shell command.",
    parameters = {
      @Param(
        name = "outputs",
        type = SkylarkList.class,
        generic1 = Artifact.class,
        named = true,
        positional = false,
        doc = "list of the output files of the action."
      ),
      @Param(
        name = "inputs",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]",
        named = true,
        positional = false,
        doc = "list of the input files of the action."
      ),
      @Param(
        name = "arguments",
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
        },
        defaultValue = "[]",
        named = true,
        positional = false,
        doc =
            "command line arguments of the action. "
                + "Must be a list of strings or actions.args() objects.<br>"
                + "Blaze passes the elements in this attribute as arguments to the command."
                + "The command can access these arguments as <code>$1</code>, <code>$2</code>, "
                + "etc. See <a href=\"actions.html#args\">ctx.actions.args()</a>."
      ),
      @Param(
        name = "mnemonic",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "a one-word description of the action, e.g. CppCompile or GoLink."
      ),
      @Param(
        name = "command",
        type = Object.class,
        allowedTypes = {
          @ParamType(type = String.class),
          @ParamType(type = SkylarkList.class, generic1 = String.class),
          @ParamType(type = Runtime.NoneType.class),
        },
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "shell command to execute.<br><br>"
                + "<b>Passing a sequence of strings to this attribute is deprecated and Blaze may "
                + "stop accepting such values in the future.</b><br><br>"
                + "The command can access the elements of the <code>arguments</code> object via "
                + "<code>$1</code>, <code>$2</code>, etc.<br>"
                + "When this argument is a string, it must be a valid shell command. For example: "
                + "\"<code>echo foo > $1</code>\". Blaze uses the same shell to execute the "
                + "command as it does for genrules."
      ),
      @Param(
        name = "progress_message",
        type = String.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "progress message to show to the user during the build, "
                + "e.g. \"Compiling foo.cc to create foo.o\"."
      ),
      @Param(
        name = "use_default_shell_env",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "whether the action should use the built in shell environment or not."
      ),
      @Param(
        name = "env",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc = "sets the dictionary of environment variables."
      ),
      @Param(
        name = "execution_requirements",
        type = SkylarkDict.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "information for scheduling the action. See "
                + "<a href=\"/docs/be/common-definitions.html#common.tags\">tags</a> "
                + "for useful keys."
      ),
      @Param(
        // TODO(bazel-team): The name here isn't accurate anymore.
        // This is technically experimental, so folks shouldn't be too attached,
        // but consider renaming to be more accurate/opaque.
        name = "input_manifests",
        type = SkylarkList.class,
        noneable = true,
        defaultValue = "None",
        named = true,
        positional = false,
        doc =
            "(Experimental) sets the input runfiles metadata; "
                + "they are typically generated by resolve_command."
      )
    }
  )
  public void runShell(
      SkylarkList outputs,
      Object inputs,
      Object arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked)
      throws EvalException {
    context.checkMutable("actions.run_shell");

    // TODO(bazel-team): builder still makes unnecessary copies of inputs, outputs and args.
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
      if (helperScript == null) {
        builder.setShellCommand(command);
      } else {
        builder.setShellCommand(helperScript.getExecPathString());
        builder.addInput(helperScript);
        FilesToRunProvider provider = context.getExecutableRunfiles(helperScript);
        if (provider != null) {
          builder.addTool(provider);
        }
      }
    } else if (commandUnchecked instanceof SkylarkList) {
      SkylarkList commandList = (SkylarkList) commandUnchecked;
      if (commandList.size() < 3) {
        throw new EvalException(null, "'command' list has to be of size at least 3");
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
        mnemonicUnchecked,
        progressMessage,
        useDefaultShellEnv,
        envUnchecked,
        executionRequirementsUnchecked,
        inputManifestsUnchecked,
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
      Object mnemonicUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      SpawnAction.Builder builder)
      throws EvalException {
    // TODO(bazel-team): builder still makes unnecessary copies of inputs, outputs and args.
    Iterable<Artifact> inputArtifacts;
    if (inputs instanceof SkylarkList) {
      inputArtifacts = ((SkylarkList) inputs).getContents(Artifact.class, "inputs");
      builder.addInputs(inputArtifacts);
    } else {
      inputArtifacts = ((SkylarkNestedSet) inputs).toCollection(Artifact.class);
      builder.addInputs(((SkylarkNestedSet) inputs).getSet(Artifact.class));
    }
    builder.addOutputs(outputs.getContents(Artifact.class, "outputs"));

    // The actual command can refer to an executable from the inputs, which could require
    // some runfiles. Consequently, we add the runfiles of all inputs of this action manually.
    for (Artifact current : inputArtifacts) {
      FilesToRunProvider provider = context.getExecutableRunfiles(current);
      if (provider != null) {
        builder.addTool(provider);
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
          ImmutableMap.copyOf(
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
    ruleContext.registerAction(builder.build(ruleContext));
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

  @SkylarkCallable(
    name = "expand_template",
    doc =
        "Creates a template expansion action. When the action is executed, it will "
            + "generate a file based on a template. Parts of the template will be replaced "
            + "using the <code>substitutions</code> dictionary. Whenever a key of the "
            + "dictionary appears in the template, it is replaced with the associated value. "
            + "There is no special syntax for the keys. You may for example use curly braces "
            + "to avoid conflicts (e.g. <code>{KEY}</code>). "
            + "<a href=\"https://github.com/laurentlb/examples/blob/master/rules/expand_template/hello.bzl\">"
            + "See example of use</a>",
    parameters = {
      @Param(
        name = "template",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "the template file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "output",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "the output file, which is a UTF-8 encoded text file."
      ),
      @Param(
        name = "substitutions",
        type = SkylarkDict.class,
        named = true,
        positional = false,
        doc = "substitutions to make when expanding the template."
      ),
      @Param(
        name = "is_executable",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "whether the output file should be executable."
      )
    }
  )
  public void expandTemplate(
      Artifact template,
      Artifact output,
      SkylarkDict<?, ?> substitutionsUnchecked,
      Boolean executable)
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
            template,
            output,
            substitutionsBuilder.build(),
            executable);
    ruleContext.registerAction(action);
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

  @SkylarkModule(
    name = "Args",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "Module providing methods to build memory-efficient command lines.<br><br>"
            + "The command lines are memory-efficient because Blaze doesn't fully construct them"
            + " until just before executing the action. "
            + "See <a href=\"actions.html#run\">ctx.actions.run()</a> or "
            + "<a href=\"actions.html#run_shell\">ctx.actions.run_shell()</a>.<br>"
            + "Example:"
            + "<pre class=language-python>\n"
            + "# foo_deps and bar_deps are each a large depset of artifacts\n"
            + "args = ctx.actions.args()\n"
            + "args.add(\"--foo\")\n"
            + "args.add(foo_deps)\n"
            + "args.add(\"--bar\")\n"
            + "args.add(bar_deps, join_with=\",\")\n"
            + "ctx.run(\n"
            + "  arguments = [args],\n"
            + "  ...\n"
            + ")\n"
            + "# Expands to [\n"
            + "#   \"--foo\",\n"
            + "#   ...artifacts from foo_deps,\n"
            + "#   \"--bar\",\n"
            + "#   ...artifacts from bar_deps joined with ',',\n"
            + "# ]"
            + "</pre>"
  )
  static class Args implements SkylarkValue {

    private final SkylarkRuleContext context;
    private final SkylarkCustomCommandLine.Builder commandLine;
    private ParameterFileType parameterFileType = ParameterFileType.SHELL_QUOTED;
    private String flagFormatString;
    private boolean useAlways;

    @SkylarkSignature(
      name = "add",
      objectType = Args.class,
      returnType = NoneType.class,
      doc = "Adds an argument to be dynamically expanded at evaluation time.",
      parameters = {
        @Param(name = "self", type = Args.class, doc = "This args object."),
        @Param(
          name = "value",
          type = Object.class,
          doc =
              "the object to add to the argument list. "
                  + "If the object is scalar, the object's string representation is added. "
                  + "If it's a <a href=\"list.html\">list</a> or "
                  + "<a href=\"depset.html\">depset</a>, "
                  + "each element's string representation is added."
        ),
        @Param(
          name = "format",
          type = String.class,
          named = true,
          positional = false,
          defaultValue = "None",
          noneable = true,
          doc =
              "a format string used to format the object(s). "
                  + "The format string is as per pattern % tuple. "
                  + "Limitations: only %d %s %r %% are supported."
        ),
        @Param(
          name = "before_each",
          type = String.class,
          named = true,
          positional = false,
          defaultValue = "None",
          noneable = true,
          doc =
              "each object in the list is prepended by this string. "
                  + "Only supported for vector arguments."
        ),
        @Param(
          name = "join_with",
          type = String.class,
          named = true,
          positional = false,
          defaultValue = "None",
          noneable = true,
          doc =
              "each object in the list is joined with this string. "
                  + "Only supported for vector arguments."
        ),
        @Param(
          name = "map_fn",
          type = BaseFunction.class,
          named = true,
          positional = false,
          defaultValue = "None",
          noneable = true,
          doc =
              "The passed objects are passed through a map function. "
                  + "For vector args the function is given a list and is expected to return a list."
        )
      },
      useLocation = true
    )
    public static final BuiltinFunction add =
        new BuiltinFunction("add") {
          @SuppressWarnings("unused")
          public NoneType invoke(
              Args self,
              Object value,
              Object format,
              Object beforeEach,
              Object joinWith,
              Object mapFn,
              Location loc)
              throws EvalException {
            if (self.isImmutable()) {
              throw new EvalException(null, "cannot modify frozen value");
            }
            if (value instanceof SkylarkNestedSet || value instanceof SkylarkList) {
              self.addVectorArg(value, format, beforeEach, joinWith, mapFn, loc);
            } else {
              self.addScalarArg(value, format, beforeEach, joinWith, mapFn, loc);
            }
            return Runtime.NONE;
          }
        };

    private void addVectorArg(
        Object value, Object format, Object beforeEach, Object joinWith, Object mapFn, Location loc)
        throws EvalException {
      if (beforeEach != Runtime.NONE && joinWith != Runtime.NONE) {
        throw new EvalException(null, "cannot pass both 'before_each' and 'join_with'");
      }
      SkylarkCustomCommandLine.VectorArg.Builder vectorArg;
      if (value instanceof SkylarkNestedSet) {
        NestedSet<?> nestedSet = ((SkylarkNestedSet) value).getSet(Object.class);
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(nestedSet);
      } else {
        SkylarkList skylarkList = (SkylarkList) value;
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(skylarkList);
      }
      vectorArg.setLocation(loc);
      if (format != Runtime.NONE) {
        vectorArg.setFormat((String) format);
      }
      if (beforeEach != Runtime.NONE) {
        vectorArg.setBeforeEach((String) beforeEach);
      }
      if (joinWith != Runtime.NONE) {
        vectorArg.setJoinWith((String) joinWith);
      }
      if (mapFn != Runtime.NONE) {
        vectorArg.setMapFn((BaseFunction) mapFn);
      }
      commandLine.add(vectorArg);
    }

    private void addScalarArg(
        Object value, Object format, Object beforeEach, Object joinWith, Object mapFn, Location loc)
        throws EvalException {
      if (!EvalUtils.isImmutable(value)) {
        throw new EvalException(null, "arg must be an immutable type");
      }
      if (beforeEach != Runtime.NONE) {
        throw new EvalException(null, "'before_each' is not supported for scalar arguments");
      }
      if (joinWith != Runtime.NONE) {
        throw new EvalException(null, "'join_with' is not supported for scalar arguments");
      }
      if (format == Runtime.NONE && mapFn == Runtime.NONE) {
        commandLine.add(value);
      } else {
        ScalarArg.Builder scalarArg = new ScalarArg.Builder(value);
        scalarArg.setLocation(loc);
        if (format != Runtime.NONE) {
          scalarArg.setFormat((String) format);
        }
        if (mapFn != Runtime.NONE) {
          scalarArg.setMapFn((BaseFunction) mapFn);
        }
        commandLine.add(scalarArg);
      }
    }

    @SkylarkCallable(
      name = "use_param_file",
      doc =
          "Spills the args to a params file, replacing them with a pointer to the param file. "
              + "Use when your args may be too large for the system's command length limits ",
      parameters = {
        @Param(
          name = "param_file_arg",
          type = String.class,
          named = true,
          doc =
              "a format string with a single \"%s\". "
                  + "If the args are spilled to a params file then they are replaced "
                  + "with an argument consisting of this string formatted with"
                  + "the path of the params file."
        ),
        @Param(
          name = "use_always",
          type = Boolean.class,
          named = true,
          positional = false,
          defaultValue = "False",
          doc =
              "whether to always spill the args to a params file. If false, "
                  + "bazel will decide whether the arguments need to be spilled "
                  + "based on your system and arg length."
        )
      }
    )
    public void useParamsFile(String paramFileArg, Boolean useAlways) throws EvalException {
      if (isImmutable()) {
        throw new EvalException(null, "cannot modify frozen value");
      }
      if (!paramFileArg.contains("%s")) {
        throw new EvalException(
            null,
            "Invalid value for parameter \"param_file_arg\": Expected string with a single \"%s\"");
      }
      this.flagFormatString = paramFileArg;
      this.useAlways = useAlways;
    }

    @SkylarkCallable(
      name = "set_param_file_format",
      doc = "sets the format of the param file when written to disk",
      parameters = {
        @Param(
          name = "format",
          type = String.class,
          named = true,
          doc =
              "the format of the param file. Must be one of:<br>"
                  + "\"shell\": All arguments are shell quoted and separated by whitespace<br>"
                  + "\"multiline\": All arguments are unquoted and separated by newline characters"
                  + "The format defaults to \"shell\" if not called."
        )
      }
    )
    public void setParamFileFormat(String format) throws EvalException {
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
    }

    public Args(
        SkylarkRuleContext context,
        SkylarkSemanticsOptions skylarkSemantics,
        EventHandler eventHandler) {
      this.context = context;
      this.commandLine = new SkylarkCustomCommandLine.Builder(skylarkSemantics, eventHandler);
    }

    public SkylarkCustomCommandLine build() {
      return commandLine.build();
    }

    @Override
    public boolean isImmutable() {
      return context.isImmutable();
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append("context.args() object");
    }

    static {
      SkylarkSignatureProcessor.configureSkylarkFunctions(Args.class);
    }
  }

  @SkylarkCallable(
    name = "args",
    doc = "returns an Args object that can be used to build memory-efficient command lines."
  )
  public Args args() {
    return new Args(
        context, skylarkSemanticsOptions, ruleContext.getAnalysisEnvironment().getEventHandler());
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
