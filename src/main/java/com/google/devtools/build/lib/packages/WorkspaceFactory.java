// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;

/** Parser for WORKSPACE files. Fills in an ExternalPackage.Builder */
// TODO(adonovan): make a simpler API around a single static function of this form:
//  nextState = Workspace.executeChunk(environment, previousState).
public class WorkspaceFactory {

  private final Package.Builder builder;

  private final Path installDir;
  private final Path workspaceDir;
  private final Path defaultSystemJavabaseDir;
  private final Mutability mutability;

  private final RuleFactory ruleFactory;

  private final WorkspaceGlobals workspaceGlobals;
  private final StarlarkSemantics starlarkSemantics;
  private final ImmutableMap<String, Object> workspaceFunctions;
  private final ImmutableList<EnvironmentExtension> environmentExtensions;

  // Values accumulated from all previous WORKSPACE file parts.
  private final Map<String, Module> loadedModules = new HashMap<>();
  private final Map<String, Object> bindings = new HashMap<>();

  // TODO(bazel-team): document installDir
  /**
   * @param builder a builder for the Workspace
   * @param ruleClassProvider a provider for known rule classes
   * @param environmentExtensions the Starlark environment extensions
   * @param mutability the Mutability for the current evaluation context
   * @param installDir the install directory
   * @param workspaceDir the workspace directory
   * @param defaultSystemJavabaseDir the local JDK directory
   */
  public WorkspaceFactory(
      Package.Builder builder,
      RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions,
      Mutability mutability,
      boolean allowOverride,
      @Nullable Path installDir,
      @Nullable Path workspaceDir,
      @Nullable Path defaultSystemJavabaseDir,
      StarlarkSemantics starlarkSemantics) {
    this.builder = builder;
    this.mutability = mutability;
    this.installDir = installDir;
    this.workspaceDir = workspaceDir;
    this.defaultSystemJavabaseDir = defaultSystemJavabaseDir;
    this.environmentExtensions = environmentExtensions;
    this.ruleFactory = new RuleFactory(ruleClassProvider);
    this.workspaceGlobals = new WorkspaceGlobals(allowOverride, ruleFactory);
    this.starlarkSemantics = starlarkSemantics;
    this.workspaceFunctions =
        createWorkspaceFunctions(
            allowOverride, ruleFactory, this.workspaceGlobals, starlarkSemantics);
  }

  /**
   * Actually runs through the AST, calling the functions in the WORKSPACE file and adding rules to
   * the //external package.
   */
  public void execute(
      StarlarkFile file, // becomes resolved as a side effect
      Map<String, Module> additionalLoadedModules,
      WorkspaceFileValue.WorkspaceFileKey workspaceFileKey)
      throws InterruptedException {
    loadedModules.putAll(additionalLoadedModules);

    // set up predeclared environment
    HashMap<String, Object> predeclared = new HashMap<>();
    predeclared.putAll(getDefaultEnvironment());
    predeclared.putAll(bindings); // (may shadow bindings in default environment)
    Module module = Module.withPredeclared(starlarkSemantics, predeclared);

    StoredEventHandler localReporter = new StoredEventHandler();
    try {
      // compile
      Program prog = Program.compileFile(file, module);

      // create thread
      StarlarkThread thread = new StarlarkThread(mutability, starlarkSemantics);
      thread.setLoader(loadedModules::get);
      thread.setPrintHandler(Event.makeDebugPrintHandler(localReporter));
      thread.setThreadLocal(
          PackageFactory.PackageContext.class,
          new PackageFactory.PackageContext(builder, null, localReporter));

      // The workspace environment doesn't need the tools repository or the fragment map
      // because executing workspace rules happens before analysis and it doesn't need a
      // repository mapping because calls to the Label constructor in the WORKSPACE file
      // are, by definition, not in an external repository and so they don't need the mapping
      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.WORKSPACE,
              /*toolsRepository=*/ null,
              /*fragmentNameToClass=*/ null,
              /*repoMapping=*/ ImmutableMap.of(),
              new SymbolGenerator<>(workspaceFileKey),
              /*analysisRuleLabel=*/ null)
          .storeInThread(thread);

      List<String> globs = new ArrayList<>(); // unused
      if (PackageFactory.checkBuildSyntax(file, globs, globs, new HashMap<>(), localReporter)) {
        try {
          Starlark.execFileProgram(prog, module, thread);
        } catch (EvalException ex) {
          localReporter.handle(
              Package.error(null, ex.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
        }
      }

      // Accumulate the global bindings created by this chunk of the WORKSPACE file,
      // for use in the next chunk. This set does not include the bindings
      // added by getDefaultEnvironment; but it does include bindings created by load,
      // so we will need to set the legacy load-binds-globally flag for this file in due course.
      this.bindings.putAll(module.getGlobals());

    } catch (SyntaxError.Exception ex) {
      // compilation failed
      Event.replayEventsOn(localReporter, ex.errors());
    }

    // cleanup (success or failure)
    builder.addPosts(localReporter.getPosts());
    builder.addEvents(localReporter.getEvents());
    if (localReporter.hasErrors()) {
      builder.setContainsErrors();
    }
  }

  /**
   * Adds the various values returned by the parsing of the previous workspace file parts. {@code
   * aPackage} is the package returned by the parent WorkspaceFileFunction, {@code loadedModules} is
   * the set of modules loaded by load statements in the parent WorkspaceFileFunction and {@code
   * variableBindings} the list of top level variable bindings of that same call.
   */
  public void setParent(
      Package aPackage, Map<String, Module> loadedModules, Map<String, Object> bindings)
      throws NameConflictException, InterruptedException {
    this.bindings.putAll(bindings);
    this.loadedModules.putAll(loadedModules);
    builder.setWorkspaceName(aPackage.getWorkspaceName());
    // Transmit the content of the parent package to the new package builder.
    if (aPackage.containsErrors()) {
      builder.setContainsErrors();
    }
    if (aPackage.getFailureDetail() != null) {
      builder.setFailureDetailOverride(aPackage.getFailureDetail());
    }
    builder.addRegisteredExecutionPlatforms(aPackage.getRegisteredExecutionPlatforms());
    builder.addRegisteredToolchains(aPackage.getRegisteredToolchains());
    builder.addRepositoryMappings(aPackage);
    for (Rule rule : aPackage.getTargets(Rule.class)) {
      try {
        // The old rule references another Package instance and we wan't to keep the invariant that
        // every Rule references the Package it is contained within
        Rule newRule =
            builder.createRule(
                rule.getLabel(),
                rule.getRuleClassObject(),
                rule.getLocation(),
                rule.getCallStack().toList(),
                rule.getAttributeContainer());
        newRule.populateOutputFiles(NullEventHandler.INSTANCE, builder);
        if (rule.containsErrors()) {
          newRule.setContainsErrors();
        }
        builder.addRule(newRule);
      } catch (LabelSyntaxException e) {
        // This rule has already been created once, so it should have worked the second time, too
        throw new IllegalStateException(e);
      }
    }
  }

  /**
   * Returns a callable Starlark value that implements the build or workspace rule "ruleClass" (e.g.
   * cc_library) in the specified package context.
   */
  private static StarlarkCallable newRuleFunction(
      final RuleFactory ruleFactory, final String ruleClassName, final boolean allowOverride) {
    return new StarlarkCallable() {
      @Override
      public String getName() {
        return ruleClassName;
      }

      @Override
      public String toString() {
        return getName() + "(...)";
      }

      @Override
      public boolean isImmutable() {
        return true;
      }

      @Override
      public void repr(Printer printer) {
        printer.append("<built-in function " + getName() + ">");
      }

      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
          throws EvalException, InterruptedException {
        if (!args.isEmpty()) {
          throw new EvalException("unexpected positional arguments");
        }
        try {
          Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
          // TODO(adonovan): this doesn't look safe!
          String externalRepoName = (String) kwargs.get("name");
          if (!allowOverride
              && externalRepoName != null
              && builder.getTarget(externalRepoName) != null) {
            throw Starlark.errorf(
                "Cannot redefine repository after any load statement in the WORKSPACE file (for"
                    + " repository '%s')",
                externalRepoName);
          }
          // Add an entry in every repository from @<mainRepoName> to "@" to avoid treating
          // @<mainRepoName> as a separate repository. This will be overridden if the main
          // repository has a repo_mapping entry from <mainRepoName> to something.
          WorkspaceFactoryHelper.addMainRepoEntry(builder, externalRepoName, thread.getSemantics());
          WorkspaceFactoryHelper.addRepoMappings(builder, kwargs, externalRepoName);
          RuleClass ruleClass = ruleFactory.getRuleClass(ruleClassName);
          RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
          Rule rule =
              WorkspaceFactoryHelper.createAndAddRepositoryRule(
                  builder,
                  ruleClass,
                  bindRuleClass,
                  WorkspaceFactoryHelper.getFinalKwargs(kwargs),
                  thread.getSemantics(),
                  thread.getCallStack());
          if (!WorkspaceGlobals.isLegalWorkspaceName(rule.getName())) {
            throw Starlark.errorf(
                "%s's name field must be a legal workspace name; workspace names may contain only"
                    + " A-Z, a-z, 0-9, '-', '_' and '.'",
                rule);
          }
        } catch (RuleFactory.InvalidRuleException
            | Package.NameConflictException
            | LabelSyntaxException e) {
          throw new EvalException(e);
        }
        return Starlark.NONE;
      }
    };
  }

  private static ImmutableMap<String, Object> createWorkspaceFunctions(
      boolean allowOverride,
      RuleFactory ruleFactory,
      WorkspaceGlobals workspaceGlobals,
      StarlarkSemantics starlarkSemantics) {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, workspaceGlobals, starlarkSemantics);

    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      // There is both a "bind" WORKSPACE function and a "bind" rule. In workspace files,
      // the non-rule function takes precedence.
      // TODO(cparsons): Rule functions should not be added to WORKSPACE files.
      if (!ruleClass.equals("bind")) {
        StarlarkCallable ruleFunction = newRuleFunction(ruleFactory, ruleClass, allowOverride);
        env.put(ruleClass, ruleFunction);
      }
    }

    return env.build();
  }

  private ImmutableMap<String, Object> getDefaultEnvironment() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.putAll(StarlarkLibrary.COMMON); // e.g. select, depset
    env.putAll(workspaceFunctions);
    if (installDir != null) {
      env.put("__embedded_dir__", installDir.getPathString());
    }
    if (workspaceDir != null) {
      env.put("__workspace_dir__", workspaceDir.getPathString());
    }
    env.put("DEFAULT_SYSTEM_JAVABASE", getDefaultSystemJavabase());
    for (EnvironmentExtension ext : environmentExtensions) {
      ext.updateWorkspace(env);
    }
    return env.build();
  }

  private String getDefaultSystemJavabase() {
    // --javabase is empty if there's no locally installed JDK
    return defaultSystemJavabaseDir != null
        ? defaultSystemJavabaseDir.toString()
        : installDir.getRelative("embedded_tools/tools/jdk/nosystemjdk").getPathString();
  }

  /** Returns the entries to populate the "native" module with, for WORKSPACE-loaded .bzl files. */
  static ImmutableMap<String, Object> createNativeModuleBindings(
      RuleClassProvider ruleClassProvider, String version) {
    // Machinery to build the collection of workspace functions.
    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider);
    WorkspaceGlobals workspaceGlobals = new WorkspaceGlobals(/*allowOverride=*/ false, ruleFactory);
    // TODO(bazel-team): StarlarkSemantics should be a parameter here, as native module can be
    // configured by flags. [brandjon: This should be possible now that we create the native module
    // in StarlarkBuiltinsFunction. We could defer creation until the StarlarkSemantics are known.
    // But mind that some code may depend on being able to enumerate all possible entries regardless
    // of the particular semantics.]
    ImmutableMap<String, Object> workspaceFunctions =
        createWorkspaceFunctions(
            /*allowOverride=*/ false, ruleFactory, workspaceGlobals, StarlarkSemantics.DEFAULT);

    // Determine the contents for native.
    ImmutableMap.Builder<String, Object> bindings = new ImmutableMap.Builder<>();
    Starlark.addMethods(bindings, new StarlarkNativeModule());
    for (Map.Entry<String, Object> entry : workspaceFunctions.entrySet()) {
      String name = entry.getKey();
      if (name.startsWith("$")) {
        // Skip "abstract" rules like "$go_rule".
        continue;
      }
      // "workspace" is explicitly omitted from the native module,
      // as it must only occur at the top of a WORKSPACE file.
      // TODO(cparsons): Clean up separation between environments.
      if (name.equals("workspace")) {
        continue;
      }
      bindings.put(entry);
    }
    bindings.put("bazel_version", version);

    return bindings.build();
  }

  public Map<String, Module> getLoadedModules() {
    return loadedModules;
  }

  public Map<String, Object> getVariableBindings() {
    return ImmutableMap.copyOf(bindings);
  }

  public Map<PathFragment, RepositoryName> getManagedDirectories() {
    return workspaceGlobals.getManagedDirectories();
  }

  public ImmutableSortedSet<String> getDoNotSymlinkInExecrootPaths() {
    return workspaceGlobals.getDoNotSymlinkInExecrootPaths();
  }
}
