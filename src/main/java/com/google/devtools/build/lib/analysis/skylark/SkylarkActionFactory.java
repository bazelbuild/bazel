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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineItemSimpleFormatter;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
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
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.skylark.SkylarkCustomCommandLine.ScalarArg;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skylarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FunctionSignature.Shape;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkMutable;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
    if (Runtime.NONE.equals(sibling)) {
      return ruleContext.getPackageRelativeTreeArtifact(
          PathFragment.create(filename), newFileRoot());
    } else {
      PathFragment original = ((Artifact) sibling).getRootRelativePath();
      PathFragment fragment = original.replaceName(filename);
      return ruleContext.getTreeArtifact(fragment, newFileRoot());
    }
  }

  @Override
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

  @Override
  public void write(FileApi output, Object content, Boolean isExecutable) throws EvalException {
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
              (Artifact) output,
              args.build(),
              args.parameterFileType,
              StandardCharsets.UTF_8);
    } else {
      throw new AssertionError("Unexpected type: " + content.getClass().getSimpleName());
    }
    ruleContext.registerAction(action);
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
      final Iterable<Artifact> toolsIterable;
      if (toolsUnchecked instanceof SkylarkList) {
        toolsIterable = ((SkylarkList) toolsUnchecked).getContents(Artifact.class, "tools");
      } else {
        toolsIterable = ((SkylarkNestedSet) toolsUnchecked).getSet(Artifact.class);
      }
      for (Artifact artifact : toolsIterable) {
        builder.addInput(artifact);
        FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
        if (provider != null) {
          builder.addTool(provider);
        }
      }
    } else {
      // Users didn't pass 'tools', kick in compatibility modes
      // Full legacy support -- add tools from inputs
      for (Artifact artifact : inputArtifacts) {
        FilesToRunProvider provider = context.getExecutableRunfiles(artifact);
        if (provider != null) {
          builder.addTool(provider);
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

  @Override
  public void expandTemplate(
      FileApi template,
      FileApi output,
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
            (Artifact) template,
            (Artifact) output,
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

  /** Args module. */
  @SkylarkModule(
    name = "Args",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "An object that encapsulates, in a memory-efficient way, the data needed to build part or "
            + "all of a command line."
            + ""
            + "<p>It often happens that an action requires a large command line containing values "
            + "accumulated from transitive dependencies. For example, a linker command line might "
            + "list every object file needed by all of the libraries being linked. It is best "
            + "practice to store such transitive data in <a href='depset.html'><code>depset"
            + "</code></a>s, so that they can be shared by multiple targets. However, if the rule "
            + "author had to convert these depsets into lists of strings in order to construct an "
            + "action command line, it would defeat this memory-sharing optimization."
            + ""
            + "<p>For this reason, the action-constructing functions accept <code>Args</code> "
            + "objects in addition to strings. Each <code>Args</code> object represents a "
            + "concatenation of strings and depsets, with optional transformations for "
            + "manipulating the data. <code>Args</code> objects do not process the depsets they "
            + "encapsulate until the execution phase, when it comes time to calculate the command "
            + "line. This helps defer any expensive copying until after the analysis phase is "
            + "complete. See the <a href='../performance.$DOC_EXT'>Optimizing Performance</a> page "
            + "for more information."
            + ""
            + "<p><code>Args</code> are constructed by calling <a href='actions.html#args'><code>"
            + "ctx.actions.args()</code></a>. They can be passed as the <code>arguments</code> "
            + "parameter of <a href='actions.html#run'><code>ctx.actions.run()</code></a> or "
            + "<a href='actions.html#run_shell'><code>ctx.actions.run_shell()</code></a>. Each "
            + "mutation of an <code>Args</code> object appends values to the eventual command "
            + "line."
            + ""
            + "<p>The <code>map_each</code> feature allows you to customize how items are "
            + "transformed into strings. If you do not provide a <code>map_each</code> function, "
            + "the standard conversion is as follows: "
            + "<ul>"
            + "<li>Values that are already strings are left as-is."
            + "<li><a href='File.html'><code>File</code></a> objects are turned into their "
            + "    <code>File.path</code> values."
            + "<li>All other types are turned into strings in an <i>unspecified</i> manner. For "
            + "    this reason, you should avoid passing values that are not of string or "
            + "    <code>File</code> type to <code>add()</code>, and if you pass them to "
            + "    <code>add_all()</code> or <code>add_joined()</code> then you should provide a "
            + "    <code>map_each</code> function."
            + "</ul>"
            + ""
            + "<p>When using string formatting (<code>format</code>, <code>format_each</code>, and "
            + "<code>format_joined</code> params of the <code>add*()</code> methods), the format "
            + "template is interpreted in the same way as <code>%</code>-substitution on strings, "
            + "except that the template must have exactly one substitution placeholder and it must "
            + "be <code>%s</code>. Literal percents may be escaped as <code>%%</code>. Formatting "
            + "is applied after the value is converted to a string as per the above."
            + ""
            + "<p>Each of the <code>add*()</code> methods have an alternate form that accepts an "
            + "extra positional parameter, an \"arg name\" string to insert before the rest of the "
            + "arguments. For <code>add_all</code> and <code>add_joined</code> the extra string "
            + "will not be added if the sequence turns out to be empty. "
            + "For instance, the same usage can add either <code>--foo val1 val2 val3 --bar"
            + "</code> or just <code>--bar</code> to the command line, depending on whether the "
            + "given sequence contains <code>val1..val3</code> or is empty."
            + ""
            + "<p>If the size of the command line can grow longer than the maximum size allowed by "
            + "the system, the arguments can be spilled over into parameter files. See "
            + "<a href='#use_param_file'><code>use_param_file()</code></a> and "
            + "<a href='#set_param_file_format'><code>set_param_file_format()</code></a>."
            + ""
            + "<p>Example: Suppose we wanted to generate the command line: "
            + "<pre>\n"
            + "--foo foo1.txt foo2.txt ... fooN.txt --bar bar1.txt,bar2.txt,...,barM.txt --baz\n"
            + "</pre>"
            + "We could use the following <code>Args</code> object: "
            + "<pre class=language-python>\n"
            + "# foo_deps and bar_deps are depsets containing\n"
            + "# File objects for the foo and bar .txt files.\n"
            + "args = ctx.actions.args()\n"
            + "args.add_all(\"--foo\", foo_deps)\n"
            + "args.add_joined(\"--bar\", bar_deps, join_with=\",\")\n"
            + "args.add(\"--baz\")\n"
            + "ctx.actions.run(\n"
            + "  ...\n"
            + "  arguments = [args],\n"
            + "  ...\n"
            + ")\n"
            + "</pre>"
  )
  @VisibleForTesting
  public static class Args extends SkylarkMutable implements CommandLineArgsApi {
    private final Mutability mutability;
    private final SkylarkSemantics skylarkSemantics;
    private final SkylarkCustomCommandLine.Builder commandLine;
    private ParameterFileType parameterFileType = ParameterFileType.SHELL_QUOTED;
    private String flagFormatString;
    private boolean useAlways;

    @Override
    public NoneType addArgument(
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
      return Runtime.NONE;
    }

    @Override
    public NoneType addAll(
        Object argNameOrValue,
        Object values,
        Object mapEach,
        Object formatEach,
        Object beforeEach,
        Boolean omitIfEmpty,
        Boolean uniquify,
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
          terminateWith != Runtime.NONE ? (String) terminateWith : null,
          loc);
      return Runtime.NONE;
    }

    @Override
    public NoneType addJoined(
        Object argNameOrValue,
        Object values,
        String joinWith,
        Object mapEach,
        Object formatEach,
        Object formatJoined,
        Boolean omitIfEmpty,
        Boolean uniquify,
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
          /* terminateWith= */ null,
          loc);
      return Runtime.NONE;
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
        String terminateWith,
        Location loc)
        throws EvalException {
      SkylarkCustomCommandLine.VectorArg.Builder vectorArg;
      if (value instanceof SkylarkNestedSet) {
        NestedSet<?> nestedSet = ((SkylarkNestedSet) value).getSet(Object.class);
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(nestedSet);
      } else {
        SkylarkList skylarkList = (SkylarkList) value;
        vectorArg = new SkylarkCustomCommandLine.VectorArg.Builder(skylarkList);
      }
      validateMapEach(mapEach, loc);
      validateFormatString("format_each", formatEach);
      validateFormatString("format_joined", formatJoined);
      vectorArg
          .setLocation(loc)
          .setArgName(argName)
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

    private void validateFormatString(String argumentName, @Nullable String formatStr)
        throws EvalException {
      if (formatStr != null
          && skylarkSemantics.incompatibleDisallowOldStyleArgsAdd()
          && !CommandLineItemSimpleFormatter.isValid(formatStr)) {
        throw new EvalException(
            null,
            String.format(
                "Invalid value for parameter \"%s\": Expected string with a single \"%%s\"",
                argumentName));
      }
    }

    private void addScalarArg(Object value, String format, BaseFunction mapFn, Location loc)
        throws EvalException {
      validateFormatString("format", format);
      if (format == null && mapFn == null) {
        commandLine.add(value);
      } else {
        ScalarArg.Builder scalarArg =
            new ScalarArg.Builder(value).setLocation(loc).setFormat(format).setMapFn(mapFn);
        commandLine.add(scalarArg);
      }
    }

    @Override
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

    @Override
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
