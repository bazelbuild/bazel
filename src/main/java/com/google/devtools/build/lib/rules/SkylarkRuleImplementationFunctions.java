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

import static com.google.devtools.build.lib.syntax.SkylarkFunction.cast;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.PositionalFunction;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.CommandHelper;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.RunfilesSupport;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.FileWriteAction;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.lib.view.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.view.actions.TemplateExpansionAction.Substitution;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

// TODO(bazel-team): function argument names are often duplicated,
// figure out a nicely readable way to get rid of the duplications.
/**
 * A helper class to provide an easier API for Skylark rule implementations
 * and hide the original Java API. This is experimental code.
 */
public class SkylarkRuleImplementationFunctions {

  @SkylarkBuiltin(name = "DEFAULT", doc = "The default runfiles collection state.")
  private static final Object defaultState = RunfilesProvider.DEFAULT_RUNFILES;

  @SkylarkBuiltin(name = "DATA", doc = "The data runfiles collection state.")
  private static final Object dataState = RunfilesProvider.DATA_RUNFILES;

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
      optionalParams = {
      @Param(name = "inputs", type = List.class, doc = "list of the input files of the action"),
      @Param(name = "outputs", type = List.class, doc = "list of the output files of the action"),
      @Param(name = "executable", doc = "the executable to be called by the action"),
      @Param(name = "arguments", type = List.class, doc = "command line arguments of the action"),
      @Param(name = "mnemonic", type = String.class, doc = "mnemonic"),
      @Param(name = "command", doc = "shell command to execute"),
      @Param(name = "command_line", doc = "a command line to execute"),
      @Param(name = "progress_message", type = String.class, doc = "progress message"),
      @Param(name = "use_default_shell_env", type = Boolean.class,
          doc = "whether the action should use the built in shell environment or not"),
      @Param(name = "env", type = Map.class, doc = "sets the dictionary of environment variables")})
  private static final SkylarkFunction createSpawnAction =
      new SimpleSkylarkFunction("action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
      SpawnAction.Builder builder = new SpawnAction.Builder(ctx.getRuleContext());
      builder.addInputs(castList(params.get("inputs"), Artifact.class, "inputs"));
      builder.addOutputs(castList(params.get("outputs"), Artifact.class, "outputs"));
      builder.addArguments(
          castList(params.get("arguments"), String.class, "arguments"));
      if (params.containsKey("executable")) {
        Object exe = params.get("executable");
        if (exe instanceof Artifact) {
          builder.setExecutable(cast(exe, Artifact.class, "executable", loc));
        } else if (exe instanceof FilesToRunProvider) {
          builder.setExecutable(cast(exe, FilesToRunProvider.class, "executable", loc));
        } else if (exe instanceof PathFragment) {
          builder.setExecutable(cast(exe, PathFragment.class, "executable", loc));
        } else {
          throw new EvalException(loc, "expected Artifact, FilesToRunProvider or PathFragment for "
              + "executable but got " + EvalUtils.getDatatypeName(exe) + " instead");
        }
      }
      if (params.containsKey("command")) {
        Object command = params.get("command");
        if (command instanceof String) {
          builder.setShellCommand((String) command);
        } else if (command instanceof List) {
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
        builder.setMnemonic(
            cast(params.get("mnemonic"), String.class, "mnemonic", loc));
      }
      if (params.containsKey("env")) {
        builder.setEnvironment(
            toMap(castMap(params.get("env"), String.class, String.class, "env")));
      }
      if (params.containsKey("progress_message")) {
        builder.setProgressMessage(cast(
            params.get("progress_message"), String.class, "progress_message", loc));
      }
      if (params.containsKey("use_default_shell_env")
          && EvalUtils.toBoolean(params.get("use_default_shell_env"))) {
        builder.useDefaultShellEnvironment();
      }
      // Always register the action
      builder.setRegisterSpawnAction(true);
      return builder.build();
    }
  };

  // TODO(bazel-team): improve this method to be more memory friendly
  @SkylarkBuiltin(name = "file_action",
      doc = "Creates a file write action.",
      objectType = SkylarkRuleContext.class,
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
      SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
      boolean executable = params.containsKey("executable")
          ? cast(params.get("executable"), Boolean.class, "executable", loc) : false;
      FileWriteAction action = new FileWriteAction(
          ctx.getRuleContext().getActionOwner(),
          cast(params.get("output"), Artifact.class, "output", loc),
          cast(params.get("content"), String.class, "content", loc),
          executable);
      ctx.getRuleContext().registerAction(action);
      return action;
    }
  };

  @SkylarkBuiltin(name = "template_action",
      doc = "Creates a template expansion action.",
      objectType = SkylarkRuleContext.class,
      mandatoryParams = {
      @Param(name = "template", type = Artifact.class, doc = "the template file"),
      @Param(name = "output", type = Artifact.class, doc = "the output file"),
      @Param(name = "substitutions", type = Map.class,
             doc = "substitutions to make when expanding the template"),
      @Param(name = "executable", type = Boolean.class,
             doc = "whether to change the output file to an executable or not")})
  private static final SkylarkFunction createTemplateAction =
    new SimpleSkylarkFunction("template_action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
      ImmutableList.Builder<Substitution> substitutions = ImmutableList.builder();
      for (Map.Entry<String, String> substitution
          : castMap(params.get("substitutions"), String.class, String.class, "substitutions")) {
        substitutions.add(Substitution.of(substitution.getKey(), substitution.getValue()));
      }

      TemplateExpansionAction action = new TemplateExpansionAction(
          ctx.getRuleContext().getActionOwner(),
          cast(params.get("template"), Artifact.class, "template", loc),
          cast(params.get("output"), Artifact.class, "output", loc),
          substitutions.build(),
          cast(params.get("executable"), Boolean.class, "executable", loc));
      ctx.getRuleContext().registerAction(action);
      return action;
    }
  };

  @SkylarkBuiltin(name = "runfiles_support", doc = "Creates a runfiles support",
      objectType = SkylarkRuleContext.class,
      mandatoryParams = {
      @Param(name = "runfiles", type = Runfiles.class,
          doc = "files the output of the rule needs at runtime"),
      @Param(name = "executable", type = Artifact.class,
          doc = "the executable output of the target")})
  private static final SkylarkFunction runfilesSupport =
      new SimpleSkylarkFunction("runfiles_support") {

    @Override
    public Object call(Map<String, Object> params, Location loc)
        throws EvalException, ExecutionException {
      SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
      Runfiles runfiles = cast(
          params.get("runfiles"), RunfilesProvider.class, "runfiles", loc).getDefaultRunfiles();
      if (runfiles.isEmpty()) {
        throw new IllegalArgumentException("Cannot use runfiles support with empty runfiles");
      }
      return RunfilesSupport.withExecutable(ctx.getRuleContext(), runfiles,
          cast(params.get("executable"), Artifact.class, "executable", loc));
    }
  };

  /**
   * Throws an error message if the first argument evaluates false. The error message can be
   * customized via the (optional) second argument.
   */
  @SkylarkBuiltin(name = "check_state",
      doc = "Checks if the first argument is True, if not, stops the execution of the Skylark "
          + "program signalling an error using the second argument as an error message.")
  private static final Function preconditionCheckState = new AbstractFunction("check_state") {

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      if (args.size() != 1 && args.size() != 2) {
        throw new EvalException(ast.getLocation(), getName()
              + " has to be called with 1 or 2 arguments.");
      }
      Object condition = args.get(0);
      if (args.size() == 1) {
        if (!EvalUtils.toBoolean(condition)) {
          throw new EvalException(ast.getLocation(), getName() + " failed.");
        }
      } else if (args.size() == 2) {
        Object message = cast(args.get(1), String.class, "second argument", ast.getLocation());
        if (!EvalUtils.toBoolean(condition)) {
          throw new EvalException(ast.getLocation(), (String) message);
        }
      }
      return 0;
    }
  };

  /**
   * A built in Skylark helper function to access the
   * Transitive info providers of Transitive info collections.
   */
  @SkylarkBuiltin(name = "provider",
      doc = "Returns the transitive info provider "
          + "(second argument) of the transitive info collection (first argument).")
  private static final Function getProvider = new PositionalFunction("provider", 2, 2) {

    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      Location loc = ast.getLocation();
      TransitiveInfoCollection collection =
          cast(args.get(0), TransitiveInfoCollection.class, "first argument", loc);
      String type = cast(args.get(1), String.class, "second argument", ast.getLocation());
      try {
        Class<?> classType = SkylarkRuleContext.classCache.get(type);
        Class<? extends TransitiveInfoProvider> convertedClass =
            classType.asSubclass(TransitiveInfoProvider.class);
        return collection.getProvider(convertedClass);
      } catch (ExecutionException e) {
        throw new EvalException(loc, "Unknown class type " + type);
      } catch (ClassCastException e) {
        throw new EvalException(loc, "Not a TransitiveInfoProvider " + type);
      }
    }
  };

  // TODO(bazel-team): Remove runfile states from Skylark.
  @SkylarkBuiltin(name = "runfiles",
      doc = "Creates a runfiles provider creating runfiles for every specified runfile state.",
      objectType = SkylarkRuleContext.class,
      optionalParams = {
      @Param(name = "stateless",
          doc = "list of the runfile items; cannot be specified together with other attributes"),
      @Param(name = "default", doc = "default runfile items"),
      @Param(name = "data", doc = "data runfile items")})
  private static final SkylarkFunction runfiles = new SimpleSkylarkFunction("runfiles") {
    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
      if (params.size() == 1) {
        return RunfilesProvider.EMPTY;
      } else if (params.containsKey("stateless")) {
        if (params.size() == 2) {
          return RunfilesProvider.simple(handleRunfiles(ctx, "stateless", params, loc));
        } else {
          throw new EvalException(loc,
              "runfiles('stateless') does not take any extra args");
        }
      } else {
        Runfiles defaultRunfiles = Runfiles.EMPTY;
        Runfiles dataRunfiles = Runfiles.EMPTY;
        if (params.containsKey("default")) {
          defaultRunfiles = handleRunfiles(ctx, "default", params, loc);
        }
        if (params.containsKey("data")) {
          dataRunfiles = handleRunfiles(ctx, "data", params, loc);
        }
        return RunfilesProvider.withData(defaultRunfiles, dataRunfiles);
      }
    }

    @SuppressWarnings("unchecked")
    private Runfiles handleRunfiles(
        SkylarkRuleContext ctx, String attr, Map<String, Object> params, Location loc)
        throws ConversionException, EvalException {
      Runfiles.Builder builder = new Runfiles.Builder();
      for (Object obj : castList(params.get(attr), Object.class, "runfiles artifacts")) {
        if (obj == RunfilesProvider.DEFAULT_RUNFILES) {
          builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
        } else if (obj == RunfilesProvider.DATA_RUNFILES) {
          builder.addRunfiles(ctx.getRuleContext(), RunfilesProvider.DATA_RUNFILES);
        } else if (obj instanceof Artifact) {
          builder.addArtifact((Artifact) obj);
        } else if (obj instanceof SkylarkFileset) {
          builder.addTransitiveArtifacts(((SkylarkNestedSet) obj).getSet(Artifact.class));
        } else if (obj instanceof NestedSet) {
          // TODO(bazel-team): This is probably not very safe in general. However it's only possible
          // to create NestedSets of Artifacts in Skylark. Remove this when we have only
          // SkylarkFileset.
          builder.addTransitiveArtifacts((NestedSet<Artifact>) obj);
        } else if (obj instanceof Iterable) {
          // This will throw a ClassCastException if the elements of the Iterable are
          // not Artifacts which will be converted to an EvalException
          builder.addArtifacts((Iterable<Artifact>) obj);
        } else {
          throw new EvalException(loc, String.format("expected an artifact, a collection of "
              + "artifacts or a runfiles state for runfiles artifacts but got '%s'",
              EvalUtils.getDatatypeName(obj)));
        }
      }
      return builder.build();
    }
  };

  @SkylarkBuiltin(name = "command_helper", doc = "Creates a command helper class.",
      objectType = SkylarkRuleContext.class,
      mandatoryParams = {
      @Param(name = "tools", type = List.class, doc = "list of tools"),
      @Param(name = "label_dict", type = Map.class,
             doc = "dictionary of resolved labels and the corresponding list of artifacts")})
  private static final SkylarkFunction createCommandHelper =
      new SimpleSkylarkFunction("command_helper") {
        @SuppressWarnings("unchecked")
        @Override
        protected Object call(Map<String, Object> params, Location loc)
            throws ConversionException, EvalException {
          SkylarkRuleContext ctx = cast(params.get("self"), SkylarkRuleContext.class, "ctx", loc);
          return new CommandHelper(ctx.getRuleContext(),
              castList(params.get("tools"), FilesToRunProvider.class, "tools"),
              // TODO(bazel-team): this cast to Map is unchecked and is not safe.
              // The best way to fix this probably is to convert CommandHelper to Skylark.
              ImmutableMap.copyOf((Map<Label, Iterable<Artifact>>) params.get("label_dict")));
        }
      };
}
