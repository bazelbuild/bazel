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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.syntax.Tuple;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

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
  private final Map<String, Extension> importMap = new HashMap<>();
  private final Map<String, Object> bindings = new HashMap<>();

  // TODO(bazel-team): document installDir
  /**
   * @param builder a builder for the Workspace
   * @param ruleClassProvider a provider for known rule classes
   * @param environmentExtensions the Skylark environment extensions
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
        WorkspaceFactory.createWorkspaceFunctions(
            allowOverride, ruleFactory, this.workspaceGlobals, starlarkSemantics);
  }

  @VisibleForTesting
  void parseForTesting(
      ParserInput source,
      @Nullable StoredEventHandler localReporter)
      throws BuildFileContainsErrorsException, InterruptedException {
    if (localReporter == null) {
      localReporter = new StoredEventHandler();
    }
    StarlarkFile file = StarlarkFile.parse(source);
    if (!file.ok()) {
      Event.replayEventsOn(localReporter, file.errors());
      throw new BuildFileContainsErrorsException(
          LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, "Failed to parse " + source.getFile());
    }
    execute(
        file,
        /*importedExtensions=*/ ImmutableMap.of(),
        localReporter,
        WorkspaceFileValue.key(
            RootedPath.toRootedPath(
                Root.fromPath(workspaceDir), PathFragment.create(source.getFile()))));
  }

  /**
   * Actually runs through the AST, calling the functions in the WORKSPACE file and adding rules to
   * the //external package.
   */
  public void execute(
      StarlarkFile file,
      Map<String, Extension> importedExtensions,
      WorkspaceFileValue.WorkspaceFileKey workspaceFileKey)
      throws InterruptedException {
    Preconditions.checkNotNull(file);
    Preconditions.checkNotNull(importedExtensions);
    execute(file, importedExtensions, new StoredEventHandler(), workspaceFileKey);
  }

  private void execute(
      StarlarkFile file,
      Map<String, Extension> importedExtensions,
      StoredEventHandler localReporter,
      WorkspaceFileValue.WorkspaceFileKey workspaceFileKey)
      throws InterruptedException {
    importMap.putAll(importedExtensions);

    // environment
    HashMap<String, Object> env = new HashMap<>();
    env.putAll(getDefaultEnvironment());
    env.putAll(bindings); // (may shadow bindings in default environment)

    StarlarkThread thread =
        StarlarkThread.builder(this.mutability)
            .setSemantics(this.starlarkSemantics)
            .setGlobals(Module.createForBuiltins(env))
            .setImportedExtensions(importMap)
            .build();
    thread.setPrintHandler(StarlarkThread.makeDebugPrintHandler(localReporter));
    thread.setThreadLocal(
        PackageFactory.PackageContext.class,
        new PackageFactory.PackageContext(builder, null, localReporter));
    Module module = thread.getGlobals();

    // The workspace environment doesn't need the tools repository or the fragment map
    // because executing workspace rules happens before analysis and it doesn't need a
    // repository mapping because calls to the Label constructor in the WORKSPACE file
    // are, by definition, not in an external repository and so they don't need the mapping
    new BazelStarlarkContext(
            BazelStarlarkContext.Phase.WORKSPACE,
            /* toolsRepository= */ null,
            /* fragmentNameToClass= */ null,
            /* repoMapping= */ ImmutableMap.of(),
            new SymbolGenerator<>(workspaceFileKey),
            /* analysisRuleLabel= */ null)
        .storeInThread(thread);

    // Validate the file, apply BUILD dialect checks, then execute.
    ValidationEnvironment.validateFile(
        file, thread.getGlobals(), this.starlarkSemantics, /*isBuildFile=*/ true);
    List<String> globs = new ArrayList<>(); // unused
    if (!file.ok()) {
      Event.replayEventsOn(localReporter, file.errors());
    } else if (PackageFactory.checkBuildSyntax(
        file, globs, globs, new HashMap<>(), localReporter)) {
      try {
        EvalUtils.exec(file, module, thread);
      } catch (EvalException ex) {
        localReporter.handle(Event.error(ex.getLocation(), ex.getMessage()));
      }
    }

    // Accumulate the global bindings created by this chunk of the WORKSPACE file,
    // for use in the next chunk. This set does not include the bindings
    // added by getDefaultEnvironment; but it does include bindings created by load,
    // so we will need to set the legacy load-binds-globally flag for this file in due course.
    this.bindings.putAll(thread.getGlobals().getBindings());

    builder.addPosts(localReporter.getPosts());
    builder.addEvents(localReporter.getEvents());
    if (localReporter.hasErrors()) {
      builder.setContainsErrors();
    }
    localReporter.clear();
  }

  /**
   * Adds the various values returned by the parsing of the previous workspace file parts. {@code
   * aPackage} is the package returned by the parent WorkspaceFileFunction, {@code importMap} is the
   * list of load statements imports computed by the parent WorkspaceFileFunction and {@code
   * variableBindings} the list of top level variable bindings of that same call.
   */
  public void setParent(
      Package aPackage, Map<String, Extension> importMap, Map<String, Object> bindings)
      throws NameConflictException, InterruptedException {
    this.bindings.putAll(bindings);
    this.importMap.putAll(importMap);
    builder.setWorkspaceName(aPackage.getWorkspaceName());
    // Transmit the content of the parent package to the new package builder.
    builder.addPosts(aPackage.getPosts());
    builder.addEvents(aPackage.getEvents());
    if (aPackage.containsErrors()) {
      builder.setContainsErrors();
    }
    builder.addRegisteredExecutionPlatforms(aPackage.getRegisteredExecutionPlatforms());
    builder.addRegisteredToolchains(aPackage.getRegisteredToolchains());
    builder.addRepositoryMappings(aPackage);
    for (Rule rule : aPackage.getTargets(Rule.class)) {
      try {
        // The old rule references another Package instance and we wan't to keep the invariant that
        // every Rule references the Package it is contained within
        Rule newRule = builder.createRule(
            rule.getLabel(),
            rule.getRuleClassObject(),
            rule.getLocation(),
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
   * Returns a function-value implementing the build or workspace rule "ruleClass" (e.g. cc_library)
   * in the specified package context.
   */
  private static BaseFunction newRuleFunction(
      final RuleFactory ruleFactory, final String ruleClassName, final boolean allowOverride) {
    return new BaseFunction() {
      @Override
      public String getName() {
        return ruleClassName;
      }

      @Override
      public FunctionSignature getSignature() {
        return FunctionSignature.KWARGS; // just for documentation
      }

      @Override
      public Object call(StarlarkThread thread, Tuple<Object> args, Dict<String, Object> kwargs)
          throws EvalException, InterruptedException {
        if (!args.isEmpty()) {
          throw new EvalException(null, "unexpected positional arguments");
        }
        try {
          Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
          // TODO(adonovan): this doesn't look safe!
          String externalRepoName = (String) kwargs.get("name");
          if (!allowOverride
              && externalRepoName != null
              && builder.getTarget(externalRepoName) != null) {
            throw new EvalException(
                null,
                "Cannot redefine repository after any load statement in the WORKSPACE file"
                    + " (for repository '"
                    + kwargs.get("name")
                    + "')");
          }
          // Add an entry in every repository from @<mainRepoName> to "@" to avoid treating
          // @<mainRepoName> as a separate repository. This will be overridden if the main
          // repository has a repo_mapping entry from <mainRepoName> to something.
          WorkspaceFactoryHelper.addMainRepoEntry(builder, externalRepoName, thread.getSemantics());
          Location loc = thread.getCallerLocation();
          WorkspaceFactoryHelper.addRepoMappings(builder, kwargs, externalRepoName, loc);
          RuleClass ruleClass = ruleFactory.getRuleClass(ruleClassName);
          RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
          Rule rule =
              WorkspaceFactoryHelper.createAndAddRepositoryRule(
                  builder,
                  ruleClass,
                  bindRuleClass,
                  WorkspaceFactoryHelper.getFinalKwargs(kwargs),
                  loc);
          if (!WorkspaceGlobals.isLegalWorkspaceName(rule.getName())) {
            throw new EvalException(
                null,
                rule
                    + "'s name field must be a legal workspace name;"
                    + " workspace names may contain only A-Z, a-z, 0-9, '-', '_' and '.'");
          }
        } catch (RuleFactory.InvalidRuleException
            | Package.NameConflictException
            | LabelSyntaxException e) {
          throw new EvalException(null, e.getMessage());
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
        BaseFunction ruleFunction = newRuleFunction(ruleFactory, ruleClass, allowOverride);
        env.put(ruleClass, ruleFunction);
      }
    }

    return env.build();
  }

  private ImmutableMap<String, Object> getDefaultEnvironment() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.putAll(Starlark.UNIVERSE);
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
    return defaultSystemJavabaseDir != null ? defaultSystemJavabaseDir.toString() : "";
  }

  private static ClassObject newNativeModule(
      ImmutableMap<String, Object> workspaceFunctions, String version) {
    ImmutableMap.Builder<String, Object> env = new ImmutableMap.Builder<>();
    Starlark.addMethods(env, new SkylarkNativeModule());
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
      env.put(entry);
    }

    env.put("bazel_version", version);
    return StructProvider.STRUCT.create(env.build(), "no native function or rule '%s'");
  }

  static ClassObject newNativeModule(RuleClassProvider ruleClassProvider, String version) {
    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider);
    WorkspaceGlobals workspaceGlobals = new WorkspaceGlobals(false, ruleFactory);
    // TODO(ichern): StarlarkSemantics should be a parameter here, as native module can be
    //  configured by flags.
    return WorkspaceFactory.newNativeModule(
        WorkspaceFactory.createWorkspaceFunctions(
            false, ruleFactory, workspaceGlobals, StarlarkSemantics.DEFAULT_SEMANTICS),
        version);
  }

  public Map<String, Extension> getImportMap() {
    return importMap;
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
