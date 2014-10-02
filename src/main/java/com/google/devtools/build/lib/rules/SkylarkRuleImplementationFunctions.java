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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.CommandHelper;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.FileWriteAction;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.lib.view.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.view.actions.TemplateExpansionAction.Substitution;

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
   *         mnemonic = 'mnemonic',
   *         command = 'command',
   *         register = 1
   *     )
   */
  @SkylarkBuiltin(name = "action",
      doc = "Creates an action that runs an executable or a shell command.",
      objectType = SkylarkRuleContext.class,
      returnType = Environment.NoneType.class,
      mandatoryParams = {
      @Param(name = "outputs", type = SkylarkList.class,
          doc = "list of the output files of the action")},
      optionalParams = {
      @Param(name = "inputs", type = SkylarkList.class,
          doc = "list of the input files of the action"),
      @Param(name = "executable", doc = "the executable file to be called by the action"),
      @Param(name = "arguments", type = SkylarkList.class,
          doc = "command line arguments of the action"),
      @Param(name = "mnemonic", type = String.class, doc = "mnemonic"),
      @Param(name = "command", doc = "shell command to execute"),
      @Param(name = "command_line", doc = "a command line to execute"),
      @Param(name = "progress_message", type = String.class,
          doc = "progress message to show to the user during the build"),
      @Param(name = "use_default_shell_env", type = Boolean.class,
          doc = "whether the action should use the built in shell environment or not"),
      @Param(name = "env", type = Map.class, doc = "sets the dictionary of environment variables")})
  private static final SkylarkFunction createSpawnAction =
      new SimpleSkylarkFunction("action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = (SkylarkRuleContext) params.get("self");
      SpawnAction.Builder builder = new SpawnAction.Builder(ctx.getRuleContext());
      builder.addInputs(castList(params.get("inputs"), Artifact.class, "inputs"));
      builder.addOutputs(castList(params.get("outputs"), Artifact.class, "outputs"));
      builder.addArguments(
          castList(params.get("arguments"), String.class, "arguments"));
      if (params.containsKey("executable")) {
        Object exe = params.get("executable");
        if (exe instanceof Artifact) {
          builder.addInput((Artifact) exe);
          builder.setExecutable((Artifact) exe);
        } else if (exe instanceof PathFragment) {
          builder.setExecutable((PathFragment) exe);
        } else {
          throw new EvalException(loc, "expected file or PathFragment for "
              + "executable but got " + EvalUtils.getDatatypeName(exe) + " instead");
        }
      }
      if (params.containsKey("command") == params.containsKey("executable")) {
        throw new EvalException(loc, "You must specify either 'command' or 'executable' argument");
      }
      if (params.containsKey("command")) {
        Object command = params.get("command");
        if (command instanceof String) {
          builder.setShellCommand((String) command);
        } else if (command instanceof SkylarkList) {
          builder.setShellCommand(castList(command, String.class, "command"));
        } else {
          throw new EvalException(loc, "expected string or list of strings for "
              + "command instead of " + EvalUtils.getDatatypeName(command));
        }
      }
      if (params.containsKey("command_line")) {
        builder.setCommandLine(CommandLine.ofCharSequences(ImmutableList.copyOf(castList(
            params.get("command_line"), CharSequence.class, "command line"))));
      }
      if (params.containsKey("mnemonic")) {
        builder.setMnemonic((String) params.get("mnemonic"));
      }
      if (params.containsKey("env")) {
        builder.setEnvironment(
            toMap(castMap(params.get("env"), String.class, String.class, "env")));
      }
      if (params.containsKey("progress_message")) {
        builder.setProgressMessage((String) params.get("progress_message"));
      }
      if (params.containsKey("use_default_shell_env")
          && EvalUtils.toBoolean(params.get("use_default_shell_env"))) {
        builder.useDefaultShellEnvironment();
      }
      // Always register the action
      builder.setRegisterSpawnAction(true);
      builder.build();
      return Environment.NONE;
    }
  };

  // TODO(bazel-team): improve this method to be more memory friendly
  @SkylarkBuiltin(name = "file_action",
      doc = "Creates a file write action.",
      objectType = SkylarkRuleContext.class,
      returnType = Environment.NoneType.class,
      optionalParams = {
        @Param(name = "executable", type = Boolean.class,
            doc = "whether the output file should be executable (default is False)"),
      },
      mandatoryParams = {
        @Param(name = "output", type = Artifact.class, doc = "the output file"),
        @Param(name = "content", type = String.class, doc = "the contents of the file")})
  private static final SkylarkFunction createFileWriteAction =
    new SimpleSkylarkFunction("file_action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = (SkylarkRuleContext) params.get("self");
      boolean executable = params.containsKey("executable")
          ? (Boolean) params.get("executable") : false;
      FileWriteAction action = new FileWriteAction(
          ctx.getRuleContext().getActionOwner(),
          (Artifact) params.get("output"),
          (String) params.get("content"),
          executable);
      ctx.getRuleContext().registerAction(action);
      return action;
    }
  };

  @SkylarkBuiltin(name = "template_action",
      doc = "Creates a template expansion action.",
      objectType = SkylarkRuleContext.class,
      returnType = Environment.NoneType.class,
      mandatoryParams = {
      @Param(name = "template", type = Artifact.class, doc = "the template file"),
      @Param(name = "output", type = Artifact.class, doc = "the output file"),
      @Param(name = "substitutions", type = Map.class,
             doc = "substitutions to make when expanding the template")},
      optionalParams = {
      @Param(name = "executable", type = Boolean.class,
          doc = "whether the output file should be executable (default is False)")})
  private static final SkylarkFunction createTemplateAction =
    new SimpleSkylarkFunction("template_action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = (SkylarkRuleContext) params.get("self");
      ImmutableList.Builder<Substitution> substitutions = ImmutableList.builder();
      for (Map.Entry<String, String> substitution
          : castMap(params.get("substitutions"), String.class, String.class, "substitutions")) {
        substitutions.add(Substitution.of(substitution.getKey(), substitution.getValue()));
      }

      boolean executable = params.containsKey("executable")
          ? (Boolean) params.get("executable") : false;
      TemplateExpansionAction action = new TemplateExpansionAction(
          ctx.getRuleContext().getActionOwner(),
          (Artifact) params.get("template"),
          (Artifact) params.get("output"),
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
  @SkylarkBuiltin(name = "provider",
      doc = "Returns the transitive info provider provided by the target.",
      mandatoryParams = {
      @Param(name = "target", type = TransitiveInfoCollection.class,
          doc = "the configured target which provides the provider"),
      @Param(name = "type", type = String.class, doc = "the class type of the provider")})
  private static final SkylarkFunction provider = new SimpleSkylarkFunction("provider") {
    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException {
      TransitiveInfoCollection target = (TransitiveInfoCollection) params.get("target");
      String type = (String) params.get("type");
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
  @SkylarkBuiltin(name = "runfiles",
      doc = "Creates a runfiles object.",
      objectType = SkylarkRuleContext.class,
      returnType = Runfiles.class,
          optionalParams = {
      @Param(name = "files", type = SkylarkList.class,
          doc = "The list of files to be added to the runfiles."),
      // TODO(bazel-team): If we have a memory efficient support for lazy list containing NestedSets
      // we can remove this and just use files = [file] + list(set)
      @Param(name = "transitive_files", type = SkylarkNestedSet.class,
          doc = "The (transitive) set of files to be added to the runfiles."),
      @Param(name = "collect_data", type = Boolean.class, doc = "Whether to collect the data "
          + "runfiles from the dependencies in srcs, data and deps attributes."),
      @Param(name = "collect_default", type = Boolean.class, doc = "Whether to collect the default "
          + "runfiles from the dependencies in srcs, data and deps attributes.")})
  private static final SkylarkFunction runfiles = new SimpleSkylarkFunction("runfiles") {
    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = (SkylarkRuleContext) params.get("self");
      Runfiles.Builder builder = new Runfiles.Builder();
      if (params.containsKey("collect_data") && (Boolean) params.get("collect_data")) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DATA_RUNFILES);
      }
      if (params.containsKey("collect_default") && (Boolean) params.get("collect_default")) {
        builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
      }
      if (params.containsKey("files")) {
        builder.addArtifacts(castList(params.get("files"), Artifact.class, "files"));
      }
      if (params.containsKey("transitive_files")) {
        builder.addTransitiveArtifacts(cast(params.get("transitive_files"),
            SkylarkNestedSet.class, "files", loc).getSet(Artifact.class));
      }
      return builder.build();
    }
  };

  @SkylarkBuiltin(name = "command_helper", doc = "Creates a command helper class.",
      objectType = SkylarkRuleContext.class,
      returnType = CommandHelper.class,
      mandatoryParams = {
      @Param(name = "tools", type = SkylarkList.class, doc = "list of tools"),
      @Param(name = "label_dict", type = Map.class,
             doc = "dictionary of resolved labels and the corresponding list of artifacts")})
  private static final SkylarkFunction createCommandHelper =
      new SimpleSkylarkFunction("command_helper") {
        @SuppressWarnings("unchecked")
        @Override
        protected Object call(Map<String, Object> params, Location loc)
            throws ConversionException, EvalException {
          SkylarkRuleContext ctx = (SkylarkRuleContext) params.get("self");
          return new CommandHelper(ctx.getRuleContext(),
              castList(params.get("tools"), FilesToRunProvider.class, "tools"),
              // TODO(bazel-team): this cast to Map is unchecked and is not safe.
              // The best way to fix this probably is to convert CommandHelper to Skylark.
              ImmutableMap.copyOf((Map<Label, Iterable<Artifact>>) params.get("label_dict")));
        }
      };
}
