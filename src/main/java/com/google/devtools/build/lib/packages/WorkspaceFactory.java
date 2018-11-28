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

import static com.google.devtools.build.lib.syntax.Runtime.NONE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SymbolGenerator;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.BuiltinFunction.Factory;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.SkylarkUtils.Phase;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Parser for WORKSPACE files.  Fills in an ExternalPackage.Builder
 */
public class WorkspaceFactory {
  private static final Pattern LEGAL_WORKSPACE_NAME = Pattern.compile("^\\p{Alpha}\\w*$");

  // List of static function added by #addWorkspaceFunctions. Used to trim them out from the
  // serialized list of variables bindings.
  private static final ImmutableList<String> STATIC_WORKSPACE_FUNCTIONS =
      ImmutableList.of(
          "workspace",
          "__embedded_dir__", // serializable so optional
          "__workspace_dir__", // serializable so optional
          "DEFAULT_SYSTEM_JAVABASE", // serializable so optional
          PackageFactory.PKG_CONTEXT);

  private final Package.Builder builder;

  private final Path installDir;
  private final Path workspaceDir;
  private final Path defaultSystemJavabaseDir;
  private final Mutability mutability;

  private final boolean allowOverride;
  private final RuleFactory ruleFactory;

  private final ImmutableMap<String, BaseFunction> workspaceFunctions;
  private final ImmutableList<EnvironmentExtension> environmentExtensions;

  // Values from the previous workspace file parts.
  // List of load statements
  private ImmutableMap<String, Extension> parentImportMap = ImmutableMap.of();
  // List of top level variable bindings
  private ImmutableMap<String, Object> parentVariableBindings = ImmutableMap.of();

  // Values accumulated up to the currently parsed workspace file part.
  // List of load statements
  private ImmutableMap<String, Extension> importMap = ImmutableMap.of();
  // List of top level variable bindings
  private ImmutableMap<String, Object> variableBindings = ImmutableMap.of();

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
      @Nullable Path defaultSystemJavabaseDir) {
    this.builder = builder;
    this.mutability = mutability;
    this.installDir = installDir;
    this.workspaceDir = workspaceDir;
    this.defaultSystemJavabaseDir = defaultSystemJavabaseDir;
    this.allowOverride = allowOverride;
    this.environmentExtensions = environmentExtensions;
    this.ruleFactory = new RuleFactory(ruleClassProvider, AttributeContainer::new);
    this.workspaceFunctions = WorkspaceFactory.createWorkspaceFunctions(
        allowOverride, ruleFactory);
  }

  @VisibleForTesting
  void parseForTesting(
      ParserInputSource source,
      StarlarkSemantics starlarkSemantics,
      @Nullable StoredEventHandler localReporter)
      throws BuildFileContainsErrorsException, InterruptedException {
    if (localReporter == null) {
      localReporter = new StoredEventHandler();
    }
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(source, localReporter);
    if (buildFileAST.containsErrors()) {
      throw new BuildFileContainsErrorsException(
          LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, "Failed to parse " + source.getPath());
    }
    execute(
        buildFileAST,
        null,
        starlarkSemantics,
        localReporter,
        WorkspaceFileValue.key(
            RootedPath.toRootedPath(Root.fromPath(workspaceDir), source.getPath())));
  }

  /**
   * Actually runs through the AST, calling the functions in the WORKSPACE file and adding rules to
   * the //external package.
   */
  public void execute(
      BuildFileAST ast,
      Map<String, Extension> importedExtensions,
      StarlarkSemantics starlarkSemantics,
      WorkspaceFileValue.WorkspaceFileKey workspaceFileKey)
      throws InterruptedException {
    Preconditions.checkNotNull(ast);
    Preconditions.checkNotNull(importedExtensions);
    execute(ast, importedExtensions, starlarkSemantics, new StoredEventHandler(), workspaceFileKey);
  }

  private void execute(
      BuildFileAST ast,
      @Nullable Map<String, Extension> importedExtensions,
      StarlarkSemantics starlarkSemantics,
      StoredEventHandler localReporter,
      WorkspaceFileValue.WorkspaceFileKey workspaceFileKey)
      throws InterruptedException {
    if (importedExtensions != null) {
      importMap = ImmutableMap.copyOf(importedExtensions);
    } else {
      importMap = parentImportMap;
    }
    Environment workspaceEnv =
        Environment.builder(mutability)
            .setSemantics(starlarkSemantics)
            .setGlobals(BazelLibrary.GLOBALS)
            .setEventHandler(localReporter)
            .setImportedExtensions(importMap)
            // The workspace environment doesn't need the tools repository or the fragment map
            // because executing workspace rules happens before analysis and it doesn't need a
            // repository mapping because calls to the Label constructor in the WORKSPACE file
            // are, by definition, not in an external repository and so they don't need the mapping
            .setStarlarkContext(
                new BazelStarlarkContext(
                    /* toolsRepository= */ null,
                    /* fragmentNameToClass= */ null,
                    ImmutableMap.of(),
                    new SymbolGenerator<>(workspaceFileKey)))
            .build();
    SkylarkUtils.setPhase(workspaceEnv, Phase.WORKSPACE);
    addWorkspaceFunctions(workspaceEnv, localReporter);
    for (Map.Entry<String, Object> binding : parentVariableBindings.entrySet()) {
      try {
        workspaceEnv.update(binding.getKey(), binding.getValue());
      } catch (EvalException e) {
        // This should never happen because everything was already evaluated.
        throw new IllegalStateException(e);
      }
    }
    if (!ast.exec(workspaceEnv, localReporter)) {
      localReporter.handle(Event.error("Error evaluating WORKSPACE file"));
    }

    // Save the list of variable bindings for the next part of the workspace file. The list of
    // variable bindings of interest are the global variable bindings that are defined by the user,
    // so not the workspace functions.
    // Workspace functions are not serializable and should not be passed over sky values. They
    // also have a package builder specific to the current part and should be reinitialized for
    // each workspace file.
    ImmutableMap.Builder<String, Object> bindingsBuilder = ImmutableMap.builder();
    GlobalFrame globals = workspaceEnv.getGlobals();
    for (String s : globals.getBindings().keySet()) {
      Object o = globals.get(s);
      if (!isAWorkspaceFunction(s, o)) {
        bindingsBuilder.put(s, o);
      }
    }
    variableBindings = bindingsBuilder.build();

    builder.addPosts(localReporter.getPosts());
    builder.addEvents(localReporter.getEvents());
    if (localReporter.hasErrors()) {
      builder.setContainsErrors();
    }
    localReporter.clear();
  }

  private boolean isAWorkspaceFunction(String name, Object o) {
    return STATIC_WORKSPACE_FUNCTIONS.contains(name) || (workspaceFunctions.get(name) == o);
  }

  private static boolean isLegalWorkspaceName(String name) {
    Matcher matcher = LEGAL_WORKSPACE_NAME.matcher(name);
    return matcher.matches();
  }

  /**
   * Adds the various values returned by the parsing of the previous workspace file parts. {@code
   * aPackage} is the package returned by the parent WorkspaceFileFunction, {@code importMap} is the
   * list of load statements imports computed by the parent WorkspaceFileFunction and {@code
   * variableBindings} the list of top level variable bindings of that same call.
   */
  public void setParent(
      Package aPackage,
      ImmutableMap<String, Extension> importMap,
      ImmutableMap<String, Object> bindings)
      throws NameConflictException, InterruptedException {
    this.parentVariableBindings = bindings;
    this.parentImportMap = importMap;
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

  @SkylarkSignature(
      name = "refresh",
      objectType = Object.class,
      returnType = NoneType.class,
      doc = "TODO",
      parameters = {
          @Param(
              name = "roots",
              doc = "TODO",
              type = SkylarkList.class,
              generic1 = String.class,
              named = true,
              positional = false,
              noneable = false
          ),
          @Param(
              name = "repository",
              doc = "TODO",
              type = String.class,
              named = true,
              positional = false,
              noneable = false
          ),
      },
      useAst = true,
      useEnvironment = true)
  private static final BuiltinFunction.Factory refresh = new BuiltinFunction.Factory("refresh") {
    public BuiltinFunction create() {
      return new BuiltinFunction("refresh", FunctionSignature.namedOnly("roots", "repository"), USE_AST_ENV) {
        public Object invoke(SkylarkList<String> roots, String repository,
            FuncallExpression ast, Environment env) throws EvalException {
          Package.Builder builder = PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
          // todo (ichern, prototype) parameters validation
          RepositoryName repositoryName;
          try {
            repositoryName = repository.startsWith("@") ?
                RepositoryName.create(repository) : RepositoryName.createFromValidStrippedName(repository);
          } catch (LabelSyntaxException e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }
          roots.forEach(root -> builder.addRefreshRootMapping(root, repositoryName));
          return NONE;
        }
      };
    }
  };

  @SkylarkSignature(
      name = "workspace",
      objectType = Object.class,
      returnType = SkylarkList.class,
      doc =
          "Sets the name for this workspace. Workspace names should be a Java-package-style "
              + "description of the project, using underscores as separators, e.g., "
              + "github.com/bazelbuild/bazel should use com_github_bazelbuild_bazel. Names must "
              + "start with a letter and can only contain letters, numbers, and underscores.",
      parameters = {@Param(name = "name", type = String.class, doc = "the name of the workspace.")},
      useAst = true,
      useEnvironment = true)
  private static final BuiltinFunction.Factory newWorkspaceFunction =
      new BuiltinFunction.Factory("workspace") {
        public BuiltinFunction create(boolean allowOverride, final RuleFactory ruleFactory) {
          if (allowOverride) {
            return new BuiltinFunction(
                "workspace", FunctionSignature.namedOnly("name"), BuiltinFunction.USE_AST_ENV) {
              public Object invoke(String name, FuncallExpression ast, Environment env)
                  throws EvalException, InterruptedException {
                if (!isLegalWorkspaceName(name)) {
                  throw new EvalException(
                      ast.getLocation(), name + " is not a legal workspace name");
                }
                String errorMessage = LabelValidator.validateTargetName(name);
                if (errorMessage != null) {
                  throw new EvalException(ast.getLocation(), errorMessage);
                }
                PackageFactory.getContext(env, ast.getLocation()).pkgBuilder.setWorkspaceName(name);
                Package.Builder builder =
                    PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
                RuleClass localRepositoryRuleClass = ruleFactory.getRuleClass("local_repository");
                RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
                Map<String, Object> kwargs =
                    ImmutableMap.<String, Object>of("name", name, "path", ".");
                try {
                  // This effectively adds a "local_repository(name = "<ws>", path = ".")"
                  // definition to the WORKSPACE file.
                  WorkspaceFactoryHelper.createAndAddRepositoryRule(
                      builder, localRepositoryRuleClass, bindRuleClass, kwargs, ast);
                } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
                  throw new EvalException(ast.getLocation(), e.getMessage());
                }
                // Add entry in repository map from "@name" --> "@" to avoid issue where bazel
                // treats references to @name as a separate external repo
                if (env.getSemantics().incompatibleRemapMainRepo()) {
                  builder.addRepositoryMappingEntry(
                      RepositoryName.MAIN,
                      RepositoryName.createFromValidStrippedName(name),
                      RepositoryName.MAIN);
                }
                return NONE;
              }
            };
          } else {
            return new BuiltinFunction(
                "workspace", FunctionSignature.namedOnly("name"), BuiltinFunction.USE_AST) {
              public Object invoke(String name, FuncallExpression ast) throws EvalException {
                throw new EvalException(
                    ast.getLocation(),
                    "workspace() function should be used only at the top of the WORKSPACE file");
              }
            };
          }
        }
      };

  private static BuiltinFunction newBindFunction(final RuleFactory ruleFactory) {
    return new BuiltinFunction(
        "bind", FunctionSignature.namedOnly(1, "name", "actual"), BuiltinFunction.USE_AST_ENV) {
      public Object invoke(String name, String actual, FuncallExpression ast, Environment env)
          throws EvalException, InterruptedException {
        Label nameLabel;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name, ImmutableMap.of());
          try {
            Package.Builder builder = PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
            RuleClass ruleClass = ruleFactory.getRuleClass("bind");
            WorkspaceFactoryHelper.addBindRule(
                builder,
                ruleClass,
                nameLabel,
                actual == null ? null : Label.parseAbsolute(actual, ImmutableMap.of()),
                ast.getLocation(),
                ruleFactory.getAttributeContainer(ruleClass));
          } catch (RuleFactory.InvalidRuleException
              | Package.NameConflictException
              | LabelSyntaxException e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }

        } catch (LabelSyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return NONE;
      }
    };
  }

  @SkylarkSignature(
    name = "register_execution_platforms",
    objectType = Object.class,
    returnType = NoneType.class,
    doc = "Registers a platform so that it is available to execute actions.",
    extraPositionals =
        @Param(
          name = "platform_labels",
          type = SkylarkList.class,
          generic1 = String.class,
          doc = "The labels of the platforms to register."
        ),
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction.Factory newRegisterExecutionPlatformsFunction =
      new BuiltinFunction.Factory("register_execution_platforms") {
        public BuiltinFunction create(final RuleFactory ruleFactory) {
          return new BuiltinFunction(
              "register_execution_platforms",
              FunctionSignature.POSITIONALS,
              BuiltinFunction.USE_LOC_ENV) {
            public Object invoke(
                SkylarkList<String> platformLabels, Location location, Environment env)
                throws EvalException, InterruptedException {

              // Add to the package definition for later.
              Package.Builder builder = PackageFactory.getContext(env, location).pkgBuilder;
              builder.addRegisteredExecutionPlatforms(
                  platformLabels.getContents(String.class, "platform_labels"));

              return NONE;
            }
          };
        }
      };

  @SkylarkSignature(
    name = "register_toolchains",
    objectType = Object.class,
    returnType = NoneType.class,
    doc =
        "Registers a toolchain created with the toolchain() rule so that it is available for "
            + "toolchain resolution.",
    extraPositionals =
        @Param(
          name = "toolchain_labels",
          type = SkylarkList.class,
          generic1 = String.class,
          doc = "The labels of the toolchains to register."
        ),
    useLocation = true,
    useEnvironment = true
  )
  private static final BuiltinFunction.Factory newRegisterToolchainsFunction =
      new BuiltinFunction.Factory("register_toolchains") {
        public BuiltinFunction create(final RuleFactory ruleFactory) {
          return new BuiltinFunction(
              "register_toolchains", FunctionSignature.POSITIONALS, BuiltinFunction.USE_LOC_ENV) {
            public Object invoke(
                SkylarkList<String> toolchainLabels, Location location, Environment env)
                throws EvalException, InterruptedException {

              // Add to the package definition for later.
              Package.Builder builder = PackageFactory.getContext(env, location).pkgBuilder;
              builder.addRegisteredToolchains(
                  toolchainLabels.getContents(String.class, "toolchain_labels"));

              return NONE;
            }
          };
        }
      };

  /**
   * Returns a function-value implementing the build or workspace rule "ruleClass" (e.g. cc_library)
   * in the specified package context.
   */
  private static BuiltinFunction newRuleFunction(
      final RuleFactory ruleFactory, final String ruleClassName, final boolean allowOverride) {
    return new BuiltinFunction(
        ruleClassName, FunctionSignature.KWARGS, BuiltinFunction.USE_AST_ENV) {
      public Object invoke(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException, InterruptedException {
        try {
          Package.Builder builder = PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
          String externalRepoName = (String) kwargs.get("name");
          if (!allowOverride
              && externalRepoName != null
              && builder.getTarget(externalRepoName) != null) {
            throw new EvalException(
                ast.getLocation(),
                "Cannot redefine repository after any load statement in the WORKSPACE file"
                    + " (for repository '"
                    + kwargs.get("name")
                    + "')");
          }
          // Add an entry in every repository from @<mainRepoName> to "@" to avoid treating
          // @<mainRepoName> as a separate repository. This will be overridden if the main
          // repository has a repo_mapping entry from <mainRepoName> to something.
          WorkspaceFactoryHelper.addMainRepoEntry(builder, externalRepoName, env.getSemantics());
          WorkspaceFactoryHelper.addRepoMappings(
              builder, kwargs, externalRepoName, ast.getLocation());
          RuleClass ruleClass = ruleFactory.getRuleClass(ruleClassName);
          RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
          Rule rule =
              WorkspaceFactoryHelper.createAndAddRepositoryRule(
                  builder,
                  ruleClass,
                  bindRuleClass,
                  WorkspaceFactoryHelper.getFinalKwargs(kwargs),
                  ast);
          if (!isLegalWorkspaceName(rule.getName())) {
            throw new EvalException(
                ast.getLocation(), rule + "'s name field must be a legal workspace name");
          }
        } catch (RuleFactory.InvalidRuleException
            | Package.NameConflictException
            | LabelSyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return NONE;
      }
    };
  }

  private static ImmutableMap<String, BaseFunction> createWorkspaceFunctions(
      boolean allowOverride, RuleFactory ruleFactory) {
    Map<String, BaseFunction> map = new HashMap<>();
    map.put("bind", newBindFunction(ruleFactory));
    map.put(
        "register_execution_platforms", newRegisterExecutionPlatformsFunction.apply(ruleFactory));
    map.put("register_toolchains", newRegisterToolchainsFunction.apply(ruleFactory));
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      if (!map.containsKey(ruleClass)) {
        BaseFunction ruleFunction = newRuleFunction(ruleFactory, ruleClass, allowOverride);
        map.put(ruleClass, ruleFunction);
      }
    }
    return ImmutableMap.copyOf(map);
  }

  private void addWorkspaceFunctions(Environment workspaceEnv, StoredEventHandler localReporter) {
    try {
      workspaceEnv.setup("workspace", newWorkspaceFunction.apply(allowOverride, ruleFactory));
      workspaceEnv.setup("refresh", refresh.apply());
      for (Map.Entry<String, BaseFunction> function : workspaceFunctions.entrySet()) {
        workspaceEnv.update(function.getKey(), function.getValue());
      }
      if (installDir != null) {
        workspaceEnv.update("__embedded_dir__", installDir.getPathString());
      }
      if (workspaceDir != null) {
        workspaceEnv.update("__workspace_dir__", workspaceDir.getPathString());
      }
      File javaHome = new File(System.getProperty("java.home"));
      if (javaHome.getName().equalsIgnoreCase("jre")) {
        javaHome = javaHome.getParentFile();
      }
      workspaceEnv.update("DEFAULT_SYSTEM_JAVABASE", getDefaultSystemJavabase());

      for (EnvironmentExtension extension : environmentExtensions) {
        extension.updateWorkspace(workspaceEnv);
      }
      workspaceEnv.setupDynamic(
          PackageFactory.PKG_CONTEXT,
          new PackageFactory.PackageContext(builder, null, localReporter, AttributeContainer::new));
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }

  private String getDefaultSystemJavabase() {
    // --javabase is empty if there's no locally installed JDK
    return defaultSystemJavabaseDir != null ? defaultSystemJavabaseDir.toString() : "";
  }

  private static ClassObject newNativeModule(
      ImmutableMap<String, BaseFunction> workspaceFunctions, String version) {
    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    SkylarkNativeModule nativeModuleInstance = new SkylarkNativeModule();
    for (String nativeFunction : FuncallExpression.getMethodNames(SkylarkNativeModule.class)) {
      builder.put(nativeFunction,
          FuncallExpression.getBuiltinCallable(nativeModuleInstance, nativeFunction));
    }
    for (Map.Entry<String, BaseFunction> function : workspaceFunctions.entrySet()) {
      builder.put(function.getKey(), function.getValue());
    }

    builder.put("bazel_version", version);
    return StructProvider.STRUCT.create(builder.build(), "no native function or rule '%s'");
  }

  static ClassObject newNativeModule(RuleClassProvider ruleClassProvider, String version) {
    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider, AttributeContainer::new);
    return WorkspaceFactory.newNativeModule(
        WorkspaceFactory.createWorkspaceFunctions(false, ruleFactory), version);
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(WorkspaceFactory.class);
  }

  public Map<String, Extension> getImportMap() {
    return importMap;
  }

  public Map<String, Object> getVariableBindings() {
    return variableBindings;
  }
}
