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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.vfs.Path;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Parser for WORKSPACE files.  Fills in an ExternalPackage.Builder
 */
public class WorkspaceFactory {
  public static final String BIND = "bind";

  private final LegacyBuilder builder;
  
  private final Path installDir;
  private final Path workspaceDir;
  private final Mutability mutability;

  private final ImmutableMap<String, BaseFunction> workspaceFunctions;
  private final ImmutableList<EnvironmentExtension> environmentExtensions;

  /**
   * @param builder a builder for the Workspace
   * @param ruleClassProvider a provider for known rule classes
   * @param mutability the Mutability for the current evaluation context
   */
  public WorkspaceFactory(
      LegacyBuilder builder,
      RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions,
      Mutability mutability) {
    this(builder, ruleClassProvider, environmentExtensions, mutability, null, null);
  }

  // TODO(bazel-team): document installDir
  /**
   * @param builder a builder for the Workspace
   * @param ruleClassProvider a provider for known rule classes
   * @param environmentExtensions the Skylark environment extensions
   * @param mutability the Mutability for the current evaluation context
   * @param installDir the install directory
   * @param workspaceDir the workspace directory
   */
  public WorkspaceFactory(
      LegacyBuilder builder,
      RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions,
      Mutability mutability,
      @Nullable Path installDir,
      @Nullable Path workspaceDir) {
    this.builder = builder;
    this.mutability = mutability;
    this.installDir = installDir;
    this.workspaceDir = workspaceDir;
    this.environmentExtensions = environmentExtensions;
    this.workspaceFunctions = createWorkspaceFunctions(ruleClassProvider);
  }

  /**
   * Parses the given WORKSPACE file without resolving skylark imports.
   *
   * <p>Called by com.google.devtools.build.workspace.Resolver from
   * //src/tools/generate_workspace.</p>
   */
  public void parse(ParserInputSource source) throws InterruptedException, IOException {
    parse(source, null);
  }

  @VisibleForTesting
  public void parse(ParserInputSource source, @Nullable StoredEventHandler localReporter)
      throws InterruptedException, IOException {
    // This method is split in 2 so WorkspaceFileFunction can call the two parts separately and
    // do the Skylark load imports in between. We can't load skylark imports from
    // generate_workspace at the moment because it doesn't have access to skyframe, but that's okay
    // because most people are just using it to resolve Maven dependencies.
    if (localReporter == null) {
      localReporter = new StoredEventHandler();
    }
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(source, localReporter, false);
    if (buildFileAST.containsErrors()) {
      throw new IOException("Failed to parse " + source.getPath());
    }
    execute(buildFileAST, null, localReporter);
  }


  /**
   * Actually runs through the AST, calling the functions in the WORKSPACE file and adding rules
   * to the //external package.
   */
  public void execute(BuildFileAST ast, Map<String, Extension> importedExtensions)
      throws InterruptedException {
    Preconditions.checkNotNull(ast);
    Preconditions.checkNotNull(importedExtensions);
    execute(ast, importedExtensions, new StoredEventHandler());
  }
  
  private void execute(BuildFileAST ast, @Nullable Map<String, Extension> importedExtensions,
      StoredEventHandler localReporter)
      throws InterruptedException {
    Environment.Builder environmentBuilder = Environment.builder(mutability)
        .setGlobals(Environment.BUILD)
        .setEventHandler(localReporter);
    if (importedExtensions != null) {
      environmentBuilder.setImportedExtensions(importedExtensions);
    }
    Environment workspaceEnv = environmentBuilder.setLoadingPhase().build();
    addWorkspaceFunctions(workspaceEnv, localReporter);
    if (!ast.exec(workspaceEnv, localReporter)) {
      localReporter.handle(Event.error("Error evaluating WORKSPACE file"));
    }

    builder.addEvents(localReporter.getEvents());
    if (localReporter.hasErrors()) {
      builder.setContainsErrors();
    }
    localReporter.clear();
  }

  // TODO(bazel-team): use @SkylarkSignature annotations on a BuiltinFunction.Factory
  // for signature + documentation of this and other functions in this file.
  private static BuiltinFunction newWorkspaceNameFunction() {
    return new BuiltinFunction(
        "workspace", FunctionSignature.namedOnly("name"), BuiltinFunction.USE_AST_ENV) {
      public Object invoke(String name, FuncallExpression ast, Environment env)
          throws EvalException {
        String errorMessage = LabelValidator.validateTargetName(name);
        if (errorMessage != null) {
          throw new EvalException(ast.getLocation(), errorMessage);
        }

        PackageFactory.getContext(env, ast).pkgBuilder.setWorkspaceName(name);
        return NONE;
      }
    };
  }

  private static BuiltinFunction newBindFunction(final RuleFactory ruleFactory) {
    return new BuiltinFunction(
        "bind", FunctionSignature.namedOnly(1, "name", "actual"), BuiltinFunction.USE_AST_ENV) {
      public Object invoke(String name, String actual, FuncallExpression ast, Environment env)
          throws EvalException, InterruptedException {
        Label nameLabel = null;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name);
          try {
            LegacyBuilder builder = PackageFactory.getContext(env, ast).pkgBuilder;
            RuleClass ruleClass = ruleFactory.getRuleClass("bind");
            builder
                .externalPackageData()
                .addBindRule(
                    builder,
                    ruleClass,
                    nameLabel,
                    actual == null ? null : Label.parseAbsolute(actual),
                    ast.getLocation());
          } catch (
              RuleFactory.InvalidRuleException | Package.NameConflictException
                      | LabelSyntaxException
                  e) {
            throw new EvalException(ast.getLocation(), e.getMessage());
          }

        } catch (LabelSyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return NONE;
      }
    };
  }

  /**
   * Returns a function-value implementing the build rule "ruleClass" (e.g. cc_library) in the
   * specified package context.
   */
  private static BuiltinFunction newRuleFunction(
      final RuleFactory ruleFactory, final String ruleClassName) {
    return new BuiltinFunction(
        ruleClassName, FunctionSignature.KWARGS, BuiltinFunction.USE_AST_ENV) {
      public Object invoke(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException, InterruptedException {
        try {
          Builder builder = PackageFactory.getContext(env, ast).pkgBuilder;
          RuleClass ruleClass = ruleFactory.getRuleClass(ruleClassName);
          RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
          builder
              .externalPackageData()
              .createAndAddRepositoryRule(builder, ruleClass, bindRuleClass, kwargs, ast);
        } catch (
            RuleFactory.InvalidRuleException | Package.NameConflictException | LabelSyntaxException
                e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return NONE;
      }
    };
  }

  private static ImmutableMap<String, BaseFunction> createWorkspaceFunctions(
      RuleClassProvider ruleClassProvider) {
    ImmutableMap.Builder<String, BaseFunction> mapBuilder = ImmutableMap.builder();
    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider);
    mapBuilder.put(BIND, newBindFunction(ruleFactory));
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      if (!ruleClass.equals(BIND)) {
        BaseFunction ruleFunction = newRuleFunction(ruleFactory, ruleClass);
        mapBuilder.put(ruleClass, ruleFunction);
      }
    }
    return mapBuilder.build();
  }

  private void addWorkspaceFunctions(Environment workspaceEnv, StoredEventHandler localReporter) {
    try {
      workspaceEnv.update("workspace", newWorkspaceNameFunction());
      for (Map.Entry<String, BaseFunction> function : workspaceFunctions.entrySet()) {
        workspaceEnv.update(function.getKey(), function.getValue());
      }
      if (installDir != null) {
        workspaceEnv.update("__embedded_dir__", installDir.getPathString());
      }
      if (workspaceDir != null) {
        workspaceEnv.update("__workspace_dir__", workspaceDir.getPathString());
      }
      File jreDirectory = new File(System.getProperty("java.home"));
      workspaceEnv.update("DEFAULT_SERVER_JAVABASE", jreDirectory.getParentFile().toString());

      for (EnvironmentExtension extension : environmentExtensions) {
        extension.updateWorkspace(workspaceEnv);
      }
      workspaceEnv.setupDynamic(
          PackageFactory.PKG_CONTEXT,
          new PackageFactory.PackageContext(builder, null, localReporter));
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }

  private static ClassObject newNativeModule(
      ImmutableMap<String, BaseFunction> workspaceFunctions) {
    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (String nativeFunction : Runtime.getFunctionNames(SkylarkNativeModule.class)) {
      builder.put(nativeFunction, Runtime.getFunction(SkylarkNativeModule.class, nativeFunction));
    }
    for (Map.Entry<String, BaseFunction> function : workspaceFunctions.entrySet()) {
      builder.put(function.getKey(), function.getValue());
    }

    return new ClassObject.SkylarkClassObject(builder.build(), "no native function or rule '%s'");
  }

  public static ClassObject newNativeModule(RuleClassProvider ruleClassProvider) {
    return newNativeModule(createWorkspaceFunctions(ruleClassProvider));
  }
}
