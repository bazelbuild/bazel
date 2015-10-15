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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.Path;

import java.io.File;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Parser for WORKSPACE files.  Fills in an ExternalPackage.Builder
 */
public class WorkspaceFactory {
  private final Builder builder;
  private final Environment environment;

  /**
   * @param builder a builder for the Workspace
   * @param ruleClassProvider a provider for known rule classes
   * @param mutability the Mutability for the current evaluation context
   */
  public WorkspaceFactory(
      Builder builder, RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions, Mutability mutability) {
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
      Builder builder,
      RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions,
      Mutability mutability,
      @Nullable Path installDir,
      @Nullable Path workspaceDir) {
    this.builder = builder;
    this.environment = createWorkspaceEnv(builder, ruleClassProvider, environmentExtensions,
        mutability, installDir, workspaceDir);
  }

  public void parse(ParserInputSource source)
      throws InterruptedException {
    StoredEventHandler localReporter = new StoredEventHandler();
    BuildFileAST buildFileAST;
    buildFileAST = BuildFileAST.parseBuildFile(source, localReporter, false);
    if (buildFileAST.containsErrors()) {
      localReporter.handle(Event.error("WORKSPACE file could not be parsed"));
    } else {
      if (!buildFileAST.exec(environment, localReporter)) {
        localReporter.handle(Event.error("Error evaluating WORKSPACE file " + source.getPath()));
      }
    }

    builder.addEvents(localReporter.getEvents());
    if (localReporter.hasErrors()) {
      builder.setContainsErrors();
    }
  }

  // TODO(bazel-team): use @SkylarkSignature annotations on a BuiltinFunction.Factory
  // for signature + documentation of this and other functions in this file.
  private static BuiltinFunction newWorkspaceNameFunction(final Builder builder) {
    return new BuiltinFunction("workspace",
        FunctionSignature.namedOnly("name"), BuiltinFunction.USE_LOC) {
      public Object invoke(String name, Location loc) throws EvalException {
        String errorMessage = LabelValidator.validateTargetName(name);
        if (errorMessage != null) {
          throw new EvalException(loc, errorMessage);
        }
        builder.setWorkspaceName(name);
        return NONE;
      }
    };
  }

  private static BuiltinFunction newBindFunction(
      final RuleFactory ruleFactory, final Builder builder) {
    return new BuiltinFunction(
        "bind", FunctionSignature.namedOnly(1, "name", "actual"), BuiltinFunction.USE_LOC) {
      public Object invoke(String name, String actual, Location loc)
          throws EvalException, InterruptedException {
        Label nameLabel = null;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name);
          try {
            RuleClass ruleClass = ruleFactory.getRuleClass("bind");
            builder
                .externalPackageData()
                .addBindRule(
                    builder,
                    ruleClass,
                    nameLabel,
                    actual == null ? null : Label.parseAbsolute(actual),
                    loc);
          } catch (
              RuleFactory.InvalidRuleException | Package.NameConflictException
                      | LabelSyntaxException
                  e) {
            throw new EvalException(loc, e.getMessage());
          }

        } catch (LabelSyntaxException e) {
          throw new EvalException(loc, e.getMessage());
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
      final RuleFactory ruleFactory, final Builder builder, final String ruleClassName) {
    return new BuiltinFunction(
        ruleClassName, FunctionSignature.KWARGS, BuiltinFunction.USE_AST_ENV) {
      public Object invoke(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException, InterruptedException {
        try {
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

  private Environment createWorkspaceEnv(
      Builder builder,
      RuleClassProvider ruleClassProvider,
      ImmutableList<EnvironmentExtension> environmentExtensions,
      Mutability mutability,
      Path installDir,
      Path workspaceDir) {
    Environment workspaceEnv = Environment.builder(mutability)
        .setGlobals(Environment.BUILD)
        .setLoadingPhase()
        .build();
    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider);
    try {
      for (String ruleClass : ruleFactory.getRuleClassNames()) {
        BaseFunction ruleFunction = newRuleFunction(ruleFactory, builder, ruleClass);
        workspaceEnv.update(ruleClass, ruleFunction);
      }
      if (installDir != null) {
        workspaceEnv.update("__embedded_dir__", installDir.getPathString());
      }
      if (workspaceDir != null) {
        workspaceEnv.update("__workspace_dir__", workspaceDir.getPathString());
      }
      File jreDirectory = new File(System.getProperty("java.home"));
      workspaceEnv.update("DEFAULT_SERVER_JAVABASE", jreDirectory.getParentFile().toString());
      workspaceEnv.update("bind", newBindFunction(ruleFactory, builder));
      workspaceEnv.update("workspace", newWorkspaceNameFunction(builder));

      for (EnvironmentExtension extension : environmentExtensions) {
        extension.updateWorkspace(workspaceEnv);
      }

      return workspaceEnv;
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }
}
