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

import static com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.cast;
import static com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.castList;
import static com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.castMap;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.MethodLibrary;
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
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;
import com.google.devtools.build.lib.view.CommandHelper;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder.StatelessRunfilesProvider;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesCollector;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.lib.view.actions.SpawnAction;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * A helper class to provide an easier API for Skylark rule implementations
 * and hide the original Java API. This is experimental code.
 */
public class SkylarkRuleImplementationFunctions {

  @SkylarkBuiltin(name = "DEFAULT", doc = "The default runfiles collection state.")
  private static final RunfilesCollector.State defaultState = RunfilesCollector.State.DEFAULT;

  @SkylarkBuiltin(name = "DATA", doc = "The data runfiles collection state.")
  private static final RunfilesCollector.State dataState = RunfilesCollector.State.DATA;

  public static final Map<String, Object> JAVA_OBJECTS_TO_EXPOSE =
      new ImmutableMap.Builder<String, Object>()
          .put("Files", SkylarkFileset.class)
          .put("ImplicitOutputsFunction", ImplicitOutputsFunction.class)
          // TODO(bazel-team): Check the Skylark environment for duplicate symbols,
          // also these symbols should be protected from being overwritten.
          .put("DEFAULT", defaultState)
          .put("DATA", dataState)
          .build();

  private final SkylarkRuleContext ruleContext;
  private final ImmutableList<Function> builtInFunctions;

  private SkylarkRuleImplementationFunctions(SkylarkRuleContext ruleContext) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    ImmutableList.Builder<Function> builtInFunctionsBuilder = ImmutableList.<Function>builder()
        .add(preconditionCheckState)
        .add(createObject)
        .add(getProvider);
    SkylarkFunction.collectSkylarkFunctionsFromFields(this, builtInFunctionsBuilder);
    builtInFunctions = builtInFunctionsBuilder.build();
  }

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
  @SkylarkBuiltin(name = "create_action",
      doc = "Creates an action and registers it with the environment (unless <i>register</i> "
          + "is set to False).",
      optionalParams = {
      @Param(name = "inputs", type = List.class, doc = "list of the input files of the action"),
      @Param(name = "outputs", type = List.class, doc = "list of the output files of the action"),
      @Param(name = "executable", doc = "the executable to be called by the action"),
      @Param(name = "arguments", type = List.class, doc = "command line arguments of the action"),
      @Param(name = "mnemonic", type = String.class, doc = "mnemonic"),
      @Param(name = "command", doc = "shell command to execute"),
      @Param(name = "register", type = Boolean.class,
          doc = "whether to register the action with the analysis environment"),
      @Param(name = "progress_message", type = String.class, doc = "progress message"),
      @Param(name = "use_default_shell_env", type = Boolean.class,
          doc = "whether the action should use the built in shell environment or not"),
      @Param(name = "env", type = Map.class, doc = "sets the dictionary of environment variables")})
  private final SkylarkFunction createSpawnAction = new SimpleSkylarkFunction("create_action") {

    @Override
    public Object call(Map<String, Object> params, Location loc) throws EvalException,
        ConversionException {
      SpawnAction.Builder builder = new SpawnAction.Builder(ruleContext.getRuleContext());
      builder.addInputs(castList(params.get("inputs"), Artifact.class, "inputs"));
      builder.addOutputs(castList(params.get("outputs"), Artifact.class, "outputs"));
      builder.addArguments(
          castList(params.get("arguments"), String.class, "arguments"));
      if (params.containsKey("executable")) {
        Object exe = params.get("executable");
        if (exe instanceof Artifact) {
          builder.setExecutable(
              cast(exe, Artifact.class, "executable", loc));
        } else if (exe instanceof FilesToRunProvider) {
          builder.setExecutable(
              cast(exe, FilesToRunProvider.class, "executable", loc));
        } else {
          throw new EvalException(loc, "expected Artifact or FilesToRunProvider for executable "
              + "but got " + EvalUtils.getDatatypeName(exe) + " instead");
        }
      }
      if (params.containsKey("command")) {
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
      }
      if (params.containsKey("mnemonic")) {
        builder.setMnemonic(
            cast(params.get("mnemonic"), String.class, "mnemonic", loc));
      }
      if (params.containsKey("register")) {
        builder.setRegisterSpawnAction(EvalUtils.toBoolean(params.get("register")));
      }
      if (params.containsKey("env")) {
        builder.setEnvironment(castMap(params.get("env"), String.class, String.class, "env"));
      }
      if (params.containsKey("progress_message")) {
        builder.setProgressMessage(cast(
            params.get("progress_message"), String.class, "progress_message", loc));
      }
      if (params.containsKey("use_default_shell_env")
          && EvalUtils.toBoolean(params.get("use_default_shell_env"))) {
        builder.useDefaultShellEnvironment();
      }
      return builder.build();
    }
  };

  @SkylarkBuiltin(name = "create_target", doc = "Creates a Configured Target from the rule.",
      optionalParams = {
      @Param(name = "files_to_build", type = SkylarkFileset.class,
          doc = "the files the rule outputs"),
      @Param(name = "runfiles", type = RunfilesProvider.class,
          doc = "files the output of the rule needs at runtime"),
      @Param(name = "executable", type = Artifact.class,
          doc = "the executable the rule creates"),
      @Param(name = "providers", type = Map.class,
          doc = "the dictionary of the transitive info providers of the rule")})
  private final SkylarkFunction createTarget = new SimpleSkylarkFunction("create_target") {

    // TODO(bazel-team): this is NOT safe. Create a wrapper class for NestedSet<Artifact>
    // to fix this. This is for files_to_build.
    @SuppressWarnings("unchecked")
    @Override
    public Object call(Map<String, Object> arguments, Location loc)
        throws EvalException, ExecutionException {
      GenericRuleConfiguredTargetBuilder builder =
          new GenericRuleConfiguredTargetBuilder(ruleContext.getRuleContext());
      if (arguments.containsKey("files_to_build")) {
        builder.setFilesToBuild(cast(arguments.get("files_to_build"),
            SkylarkFileset.class, "files_to_build", loc).getFileset());
      }
      if (arguments.containsKey("runfiles")) {
        builder.add(RunfilesProvider.class, cast(arguments.get("runfiles"),
            RunfilesProvider.class, "runfiles", loc));
      }
      if (arguments.containsKey("executable")) {
        builder.setExecutable(cast(arguments.get("executable"), Artifact.class, "executable", loc));
      }
      for (Map.Entry<String, TransitiveInfoProvider> provider : castMap(arguments.get("providers"),
          String.class, TransitiveInfoProvider.class, "transitive info providers")) {
        builder.addProvider(SkylarkRuleContext.classCache.get(provider.getKey())
            .asSubclass(TransitiveInfoProvider.class), provider.getValue());
      }
      return builder.build();
    }
  };

  /**
   * Throws an error message if the first argument evaluates false. The error message can be
   * customized via the (optional) second argument.
   */
  @SkylarkBuiltin(name = "check_state",
      doc = "Checks if the first argument is True, if not, stops the execution of the Skylark "
          + "program signalling an error using the second argument as an error message.")
  private final Function preconditionCheckState = new AbstractFunction("check_state") {

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
   * A built in Skylark helper function to invoke Java constructors.
   */
  @SkylarkBuiltin(name = "create_object",
      doc = "Creates an arbitrary Java object. The first argument sepcifies the Java class, "
          + "the rest are the parameters of the constructor.")
  private final Function createObject =
      new PositionalFunction("create_object", 1, Integer.MAX_VALUE) {

    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      String classTypeString = cast(args.get(0), String.class, "first argument", ast.getLocation());
      try {
        Class<?> classType = SkylarkRuleContext.classCache.get(classTypeString);
        Constructor<?> matchingConstructor = null;
        for (Constructor<?> constructor : classType.getConstructors()) {
          Class<?>[] params = constructor.getParameterTypes();
          if (params.length == args.size() - 1 && !constructor.isSynthetic()) {
            int i = 1;
            boolean matching = true;
            for (Class<?> param : params) {
              if (!param.isAssignableFrom(args.get(i).getClass())) {
                matching = false;
                break;
              }
              i++;
            }
            if (matching) {
              if (matchingConstructor == null) {
                matchingConstructor = constructor;
              } else {
                throw new EvalException(ast.getLocation(),
                    "Multiple matching constructors for " + classTypeString);
              }
            }
          }
        }
        try {
          if (matchingConstructor != null) {
            return matchingConstructor.newInstance(args.subList(1, args.size()).toArray());
          } else {
            throw new EvalException(ast.getLocation(),
                "No matching constructor for " + classTypeString);
          }
        } catch (InstantiationException | IllegalAccessException | IllegalArgumentException
            | InvocationTargetException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
      } catch (ExecutionException e) {
        throw new EvalException(ast.getLocation(), "Unknown class type " + classTypeString);
      }
    }
  };

  /**
   * A built in Skylark helper function to access the
   * Transitive info providers of Transitive info collections.
   */
  @SkylarkBuiltin(name = "get_provider",
      doc = "Returns the transitive info provider "
          + "(second argument) of the transitive info collection (first argument).")
  private final Function getProvider = new PositionalFunction("get_provider", 2, 2) {

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
      optionalParams = {
      @Param(name = "stateless",
          doc = "list of the runfile items; cannot be specified together with other attributes"),
      @Param(name = "default", doc = "default runfile items"),
      @Param(name = "data", doc = "data runfile items")})
  private final SkylarkFunction runfiles = new SimpleSkylarkFunction("runfiles") {
    @Override
    public Object call(Map<String, Object> arguments, Location loc) throws EvalException,
        ConversionException {
      if (arguments.size() == 0) {
        return RunfilesProvider.EMPTY;
      } else if (arguments.containsKey("stateless")) {
        if (arguments.size() == 1) {
          return new StatelessRunfilesProvider(handleRunfiles("stateless", arguments, loc));
        } else {
          throw new EvalException(loc,
              "runfiles('stateless') does not take any extra args");
        }
      } else {
        RunfilesProvider.Builder builder = new RunfilesProvider.Builder();
        for (String param : arguments.keySet()) {
          RunfilesCollector.State state = RunfilesCollector.State.valueOf(param.toUpperCase());
          builder.add(handleRunfiles(param, arguments, loc), state);
        }
        return builder.build();
      }
    }

    @SuppressWarnings("unchecked")
    private Runfiles handleRunfiles(String attr, Map<String, Object> arguments, Location loc)
        throws ConversionException, EvalException {
      Runfiles.Builder builder = new Runfiles.Builder();
      for (Object obj : castList(arguments.get(attr), Object.class,
          "runfiles artifacts")) {
        if (obj instanceof RunfilesCollector.State) {
          builder.addRunfiles((RunfilesCollector.State) obj, ruleContext.getRuleContext());
        } else if (obj instanceof Artifact) {
          builder.addArtifact((Artifact) obj);
        } else if (obj instanceof SkylarkFileset) {
          builder.addTransitiveArtifacts(((SkylarkFileset) obj).getFileset());
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

  @SkylarkBuiltin(name = "fileset",
      doc = "Creates a nested fileset from the <i>items</i>. "
          + "The nesting is applied to other nested filesets among <i>items</i>.",
      mandatoryParams = {
      @Param(name = "order", type = String.class, doc = "ordering of the nested file set")},
      optionalParams = {
      @Param(name = "items", doc = "items of the nested file set")})
  private final SkylarkFunction createFileset = new SimpleSkylarkFunction("fileset") {

    // TODO(bazel-team): Not safe, see comments at runfiles and createTarget.
    @SuppressWarnings("unchecked")
    @Override
    protected Object call(Map<String, Object> arguments, Location loc)
        throws EvalException, ConversionException {
      // TODO(bazel-team): Investigate if enums or constants can be used here in the argument.
      String orderString = cast(arguments.get("order"), String.class, "fileset order", loc);
      Order order;
      try {
        order = Order.valueOf(orderString);
      } catch (IllegalArgumentException e) {
        throw new EvalException(loc, "Invalid fileset order " + orderString);
      }
      return new SkylarkFileset(
          order, castList(arguments.get("items"), Object.class, "fileset items"));
    }
  };

  @SkylarkBuiltin(name = "create_command_helper", doc = "Creates a command helper class.",
      mandatoryParams = {
      @Param(name = "tools", type = List.class, doc = "list of tools"),
      @Param(name = "label_dict", type = Map.class,
             doc = "dictionary of resolved labels and the corresponding list of artifacts")})
  private final SkylarkFunction createCommandHelper =
      new SimpleSkylarkFunction("create_command_helper") {
        @SuppressWarnings("unchecked")
        @Override
        protected Object call(Map<String, Object> arguments, Location loc)
            throws ConversionException {
          return new CommandHelper(ruleContext.getRuleContext(),
              castList(arguments.get("tools"), FilesToRunProvider.class, "tools"),
              // TODO(bazel-team): this cast to Map is unchecked and is not safe.
              // The best way to fix this probably is to convert CommandHelper to Skylark.
              ImmutableMap.copyOf((Map<Label, Iterable<Artifact>>) arguments.get("label_dict")));
        }
      };

  public static SkylarkEnvironment getNewEnvironment(SkylarkRuleContext context,
      ImmutableMap<String, Class<?>> builtInClasses) {
    // The rule definition environment is needed because of the helper functions and constants
    // defined there. However the rule definition environment is shared between rules
    // of the same type, so it needs to be cloned to avoid conflicts. Note that Skylark
    // Environments don't have parents by design.
    // TODO(bazel-team): Alternatively (or additionally), we should forbid to write global variables
    // in rule implementations. We could also introduce Immutable Environments.
    // Another problem: this provides access to SkylarkRuleClass functions (e.g. rule, attr) for
    // the rule implementations.
    SkylarkEnvironment ruleDefEnv =
        context.getRuleContext().getRule().getRuleClassObject().getRuleDefinitionEnvironment();
    final SkylarkEnvironment env;
    // TODO(bazel-team): This is needed because of the tests. If we separated Skylark from legacy
    // Blaze completely (including testing) we can remove this.
    if (ruleDefEnv != null) {
      Preconditions.checkState(ruleDefEnv.isGlobalEnvironment());
      env = ruleDefEnv.cloneEnv();
    } else {
      env = new SkylarkEnvironment();
      MethodLibrary.setupMethodEnvironment(env);
    }

    SkylarkRuleImplementationFunctions functions = new SkylarkRuleImplementationFunctions(context);
    for (Function builtInFunction : functions.builtInFunctions) {
      env.update(builtInFunction.getName(), builtInFunction);
    }
    for (Map.Entry<String, Class<?>> entry : builtInClasses.entrySet()) {
      env.update(entry.getKey(), entry.getValue());
    }
    for (Map.Entry<String, Object> entry : JAVA_OBJECTS_TO_EXPOSE.entrySet()) {
      env.update(entry.getKey(), entry.getValue());
    }

    return env;
  }

  public static ValidationEnvironment getValidationEnvironment(
      ImmutableMap<String, Class<?>> extraObjects) {
    ImmutableMap.Builder<String, Class<?>> builder = ImmutableMap.builder();
    // TODO(bazel-team): kill check_state, create_object and get_provider.
    builder
        .put("check_state", Object.class)
        .put("check_state.return", Object.class)
        .put("create_object", Object.class)
        .put("create_object.return", Object.class)
        .put("get_provider", Object.class)
        .put("get_provider.return", Object.class);
    MethodLibrary.setupValidationEnvironment(builder);
    SkylarkFunction.collectSkylarkFunctionReturnTypesFromFields(
        SkylarkRuleClassFunctions.class, builder);
    SkylarkFunction.collectSkylarkFunctionReturnTypesFromFields(
        SkylarkRuleImplementationFunctions.class, builder);
    for (Map.Entry<String, Object> entry : JAVA_OBJECTS_TO_EXPOSE.entrySet()) {
      builder.put(entry.getKey(), entry.getValue().getClass());
    }
    builder.putAll(extraObjects);
    return new ValidationEnvironment(builder.build());
  }

  @VisibleForTesting
  public static ValidationEnvironment getValidationEnvironment() {
    return getValidationEnvironment(ImmutableMap.<String, Class<?>>of());
  }
}
