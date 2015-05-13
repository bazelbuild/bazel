// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.syntax.SkylarkType.castList;
import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSignature;
import com.google.devtools.build.lib.syntax.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutionException;

// TODO(bazel-team): function argument names are often duplicated,
// figure out a nicely readable way to get rid of the duplications.
/**
 * A helper class to provide an easier API for Skylark rule implementations
 * and hide the original Java API. This is experimental code.
 */
public class SkylarkRuleImplementationFunctions {

  // TODO(bazel-team): add all the remaining parameters
  // TODO(bazel-team): merge executable and arguments
  /**
   * A Skylark built-in function to create and register a SpawnAction using a
   * dictionary of parameters:
   * createSpawnAction(
   *         inputs = [input1, input2, ...],
   *         outputs = [output1, output2, ...],
   *         executable = executable,
   *         arguments = [argument1, argument2, ...],
   *         mnemonic = 'Mnemonic',
   *         command = 'command',
   *     )
   */
  @SkylarkSignature(name = "action",
      doc = "Creates an action that runs an executable or a shell command.",
      objectType = SkylarkRuleContext.class,
      returnType = Environment.NoneType.class,
      mandatoryPositionals = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "This RuleContext.")},
      mandatoryNamedOnly = {
        @Param(name = "outputs", type = SkylarkList.class, generic1 = Artifact.class,
            doc = "list of the output files of the action")},
      optionalNamedOnly = {
        @Param(name = "inputs", type = SkylarkList.class, generic1 = Artifact.class,
            defaultValue = "[]", doc = "list of the input files of the action"),
        @Param(name = "executable", type = Object.class, // File or PathFragment or None
            defaultValue = "None",
            doc = "the executable file to be called by the action"),
        @Param(name = "arguments", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = "command line arguments of the action"),
        @Param(name = "mnemonic", type = String.class, noneable = true,
            defaultValue = "None",
            doc = "a one-word description of the action, e.g. CppCompile or GoLink"),
        @Param(name = "command", type = Object.class, // string or ListOf(string) or NoneType
            defaultValue = "None", doc = "shell command to execute"),
        @Param(name = "progress_message", type = String.class, noneable = true,
            defaultValue = "None",
            doc = "progress message to show to the user during the build, "
            + "e.g. \"Compiling foo.cc to create foo.o\""),
        @Param(name = "use_default_shell_env", type = Boolean.class, defaultValue = "False",
            doc = "whether the action should use the built in shell environment or not"),
        @Param(name = "env", type = Map.class, noneable = true, defaultValue = "None",
            doc = "sets the dictionary of environment variables"),
        @Param(name = "execution_requirements", type = Map.class, noneable = true,
            defaultValue = "None",
            doc = "information for scheduling the action"),
        @Param(name = "input_manifests", type = Map.class, noneable = true,
            defaultValue = "None",
            doc = "sets the map of input manifests files; "
            + "they are typicially generated by the command_helper")},
      useLocation = true)
  private static final BuiltinFunction createSpawnAction = new BuiltinFunction("action") {
    public Environment.NoneType invoke(
        SkylarkRuleContext ctx,
        SkylarkList outputs,
        SkylarkList inputs,
        Object executableO,
        SkylarkList arguments,
        Object mnemonicO,
        Object commandO,
        Object progressMessage,
        Boolean useDefaultShellEnv,
        Object envO,
        Object executionRequirementsO,
        Object inputManifestsO,
        Location loc) throws EvalException, ConversionException {
      SpawnAction.Builder builder = new SpawnAction.Builder();
      // TODO(bazel-team): builder still makes unnecessary copies of inputs, outputs and args.
      builder.addInputs(castList(inputs, Artifact.class));
      builder.addOutputs(castList(outputs, Artifact.class));
      if (commandO != Environment.NONE && arguments.size() > 0) {
        // When we use a shell command, add an empty argument before other arguments.
        //   e.g.  bash -c "cmd" '' 'arg1' 'arg2'
        // bash will use the empty argument as the value of $0 (which we don't care about).
        // arg1 and arg2 will be $1 and $2, as a user exects.
        builder.addArgument("");
      }
      builder.addArguments(castList(arguments, String.class));
      if (executableO != Environment.NONE) {
        if (executableO instanceof Artifact) {
          Artifact executable = (Artifact) executableO;
          builder.addInput(executable);
          FilesToRunProvider provider = ctx.getExecutableRunfiles(executable);
          if (provider == null) {
            builder.setExecutable(executable);
          } else {
            builder.setExecutable(provider);
          }
        } else if (executableO instanceof PathFragment) {
          builder.setExecutable((PathFragment) executableO);
        } else {
          throw new EvalException(loc, "expected file or PathFragment for "
              + "executable but got " + EvalUtils.getDataTypeName(executableO) + " instead");
        }
      }
      if ((commandO == Environment.NONE) == (executableO == Environment.NONE)) {
        throw new EvalException(loc, "You must specify either 'command' or 'executable' argument");
      }
      if (commandO != Environment.NONE) {
        if (commandO instanceof String) {
          builder.setShellCommand((String) commandO);
        } else if (commandO instanceof SkylarkList) {
          SkylarkList commandList = (SkylarkList) commandO;
          if (commandList.size() < 3) {
            throw new EvalException(loc, "'command' list has to be of size at least 3");
          }
          builder.setShellCommand(castList(commandList, String.class, "command"));
        } else {
          throw new EvalException(loc, "expected string or list of strings for "
              + "command instead of " + EvalUtils.getDataTypeName(commandO));
        }
      }
      if (mnemonicO != Environment.NONE) {
        builder.setMnemonic((String) mnemonicO);
      }
      if (envO != Environment.NONE) {
        builder.setEnvironment(ImmutableMap.copyOf(
            castMap(envO, String.class, String.class, "env")));
      }
      if (progressMessage != Environment.NONE) {
        builder.setProgressMessage((String) progressMessage);
      }
      if (EvalUtils.toBoolean(useDefaultShellEnv)) {
        builder.useDefaultShellEnvironment();
      }
      if (executionRequirementsO != Environment.NONE) {
        builder.setExecutionInfo(ImmutableMap.copyOf(castMap(
            executionRequirementsO, String.class, String.class, "execution_requirements")));
      }
      if (inputManifestsO != Environment.NONE) {
        for (Map.Entry<PathFragment, Artifact> entry : castMap(inputManifestsO,
            PathFragment.class, Artifact.class, "input manifest file map").entrySet()) {
          builder.addInputManifest(entry.getValue(), entry.getKey());
        }
      }
      // Always register the action
      ctx.getRuleContext().registerAction(builder.build(ctx.getRuleContext()));
      return Environment.NONE;
    }
  };

  // TODO(bazel-team): improve this method to be more memory friendly
  @SkylarkSignature(name = "file_action",
      doc = "Creates a file write action.",
      objectType = SkylarkRuleContext.class,
      returnType = FileWriteAction.class,
      mandatoryPositionals = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context"),
        @Param(name = "output", type = Artifact.class, doc = "the output file"),
        @Param(name = "content", type = String.class, doc = "the contents of the file")},
      optionalPositionals = {
        @Param(name = "executable", type = Boolean.class, defaultValue = "False",
            doc = "whether the output file should be executable (default is False)")})
  private static final BuiltinFunction createFileWriteAction =
      new BuiltinFunction("file_action") {
        public FileWriteAction invoke(SkylarkRuleContext ctx,
            Artifact output, String content, Boolean executable)
            throws EvalException, ConversionException {
          FileWriteAction action = new FileWriteAction(
              ctx.getRuleContext().getActionOwner(), output, content, executable);
          ctx.getRuleContext().registerAction(action);
          return action;
        }
      };

  @SkylarkSignature(name = "template_action",
      doc = "Creates a template expansion action.",
      objectType = SkylarkRuleContext.class,
      returnType = TemplateExpansionAction.class,
      mandatoryPositionals = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "this context")},
      mandatoryNamedOnly = {
        @Param(name = "template", type = Artifact.class,
            doc = "the template file"),
        @Param(name = "output", type = Artifact.class,
            doc = "the output file"),
        @Param(name = "substitutions", type = Map.class,
            doc = "substitutions to make when expanding the template")},
      optionalNamedOnly = {
        @Param(name = "executable", type = Boolean.class,
            doc = "whether the output file should be executable (default is False)")})
  private static final BuiltinFunction createTemplateAction =
      new BuiltinFunction("template_action", Arrays.<Object>asList(false)) {
        public TemplateExpansionAction invoke(SkylarkRuleContext ctx,
            Artifact template, Artifact output, Map<?, ?> substitutionsO, Boolean executable)
            throws EvalException, ConversionException {
          ImmutableList.Builder<Substitution> substitutions = ImmutableList.builder();
          for (Map.Entry<String, String> substitution : castMap(
              substitutionsO, String.class, String.class, "substitutions").entrySet()) {
            substitutions.add(Substitution.of(substitution.getKey(), substitution.getValue()));
          }
          TemplateExpansionAction action = new TemplateExpansionAction(
              ctx.getRuleContext().getActionOwner(),
              template,
              output,
              substitutions.build(),
              executable);
          ctx.getRuleContext().registerAction(action);
          return action;
        }
      };

  /**
   * A built in Skylark helper function to access the
   * Transitive info providers of Transitive info collections.
   */
  @SkylarkSignature(name = "provider",
      doc = "Returns the transitive info provider provided by the target.",
      mandatoryPositionals = {
        @Param(name = "target", type = TransitiveInfoCollection.class,
            doc = "the configured target which provides the provider"),
        @Param(name = "type", type = String.class, doc = "the class type of the provider")},
      useLocation = true)
  private static final BuiltinFunction provider = new BuiltinFunction("provider") {
      public Object invoke(TransitiveInfoCollection target, String type,
          Location loc) throws EvalException {
      try {
        Class<?> classType = SkylarkRuleContext.classCache.get(type);
        Class<? extends TransitiveInfoProvider> convertedClass =
            classType.asSubclass(TransitiveInfoProvider.class);
        Object result = target.getProvider(convertedClass);
        return result == null ? Environment.NONE : result;
      } catch (ExecutionException e) {
        throw new EvalException(loc, "Unknown class type " + type);
      } catch (ClassCastException e) {
        throw new EvalException(loc, "Not a TransitiveInfoProvider " + type);
      }
    }
  };

  // TODO(bazel-team): Remove runfile states from Skylark.
  @SkylarkSignature(name = "runfiles",
      doc = "Creates a runfiles object.",
      objectType = SkylarkRuleContext.class,
      returnType = Runfiles.class,
      mandatoryPositionals = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "This context.")},
      optionalPositionals = {
        @Param(name = "files", type = SkylarkList.class, generic1 = Artifact.class,
            defaultValue = "[]", doc = "The list of files to be added to the runfiles."),
        // TODO(bazel-team): If we have a memory efficient support for lazy list containing
        // NestedSets we can remove this and just use files = [file] + list(set)
        // Also, allow empty set for init
        @Param(name = "transitive_files", type = SkylarkNestedSet.class, generic1 = Artifact.class,
            noneable = true, defaultValue = "None",
            doc = "The (transitive) set of files to be added to the runfiles."),
        @Param(name = "collect_data", type = Boolean.class, defaultValue = "False",
            doc = "Whether to collect the data "
            + "runfiles from the dependencies in srcs, data and deps attributes."),
        @Param(name = "collect_default", type = Boolean.class, defaultValue = "False",
            doc = "Whether to collect the default "
            + "runfiles from the dependencies in srcs, data and deps attributes.")},
      useLocation = true)
  private static final BuiltinFunction runfiles = new BuiltinFunction("runfiles") {
    public Runfiles invoke(SkylarkRuleContext ctx, SkylarkList files, Object transitiveFiles,
        Boolean collectData, Boolean collectDefault,
        Location loc) throws EvalException, ConversionException {
      Runfiles.Builder builder = new Runfiles.Builder();
      if (EvalUtils.toBoolean(collectData)) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DATA_RUNFILES);
      }
      if (EvalUtils.toBoolean(collectDefault)) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
      }
      if (!files.isEmpty()) {
        builder.addArtifacts(castList(files, Artifact.class));
      }
      if (transitiveFiles != Environment.NONE) {
        builder.addTransitiveArtifacts(((SkylarkNestedSet) transitiveFiles).getSet(Artifact.class));
      }
      return builder.build();
    }
  };

  @SkylarkSignature(name = "command_helper", doc = "Experimental. Creates a command helper class.",
      objectType = SkylarkRuleContext.class,
      returnType = CommandHelper.class,
      mandatoryPositionals = {
        @Param(name = "self", type = SkylarkRuleContext.class, doc = "this RuleContext"),
        @Param(name = "tools", type = SkylarkList.class, generic1 = TransitiveInfoCollection.class,
            doc = "list of tools (list of targets)"),
        @Param(name = "label_dict", type = Map.class, defaultValue = "{}",
            doc = "dictionary of resolved labels and the corresponding list of Files "
            + "(a dict of Label : list of Files)")})
  private static final BuiltinFunction createCommandHelper = new BuiltinFunction("command_helper") {
      @SuppressWarnings("unchecked")
      // TODO(bazel-team): this cast to Map is unchecked and is not safe.
      // The best way to fix this probably is to convert CommandHelper to Skylark.
      public CommandHelper invoke(
          SkylarkRuleContext ctx, SkylarkList tools, Map<Label, Iterable<Artifact>> labelDict)
          throws ConversionException, EvalException {
        return new CommandHelper(ctx.getRuleContext(),
            AnalysisUtils.getProviders(
                castList(tools, TransitiveInfoCollection.class),
                FilesToRunProvider.class),
            ImmutableMap.copyOf(labelDict));
      }
    };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkRuleImplementationFunctions.class);
  }
}
