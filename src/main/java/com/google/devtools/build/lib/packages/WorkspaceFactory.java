// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ExternalPackage.Builder;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.ParserInputSource;

import java.io.File;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Parser for WORKSPACE files.  Fills in an ExternalPackage.Builder
 */
public class WorkspaceFactory {
  private final Builder builder;
  private final Environment environment;

  public WorkspaceFactory(Builder builder, RuleClassProvider ruleClassProvider) {
    this(builder, ruleClassProvider, null);
  }

  public WorkspaceFactory(
      Builder builder, RuleClassProvider ruleClassProvider, @Nullable String installDir) {
    this.builder = builder;
    this.environment = createWorkspaceEnv(builder, ruleClassProvider, installDir);
  }

  public void parse(ParserInputSource source)
      throws InterruptedException {
    StoredEventHandler localReporter = new StoredEventHandler();
    BuildFileAST buildFileAST;
    buildFileAST = BuildFileAST.parseBuildFile(source, localReporter, null, false);
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
    return new BuiltinFunction("bind",
        FunctionSignature.namedOnly(1, "name", "actual"), BuiltinFunction.USE_LOC) {
      public Object invoke(String name, String actual, Location loc)
          throws EvalException, InterruptedException {
        Label nameLabel = null;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name);
          try {
            RuleClass ruleClass = ruleFactory.getRuleClass("bind");
            builder.addBindRule(ruleClass, nameLabel,
                actual == null ? null : Label.parseAbsolute(actual), loc);
          } catch (RuleFactory.InvalidRuleException | Package.NameConflictException |
            Label.SyntaxException e) {
            throw new EvalException(loc, e.getMessage());
          }

        } catch (Label.SyntaxException e) {
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
    return new BuiltinFunction(ruleClassName,
        FunctionSignature.KWARGS, BuiltinFunction.USE_AST) {
      public Object invoke(Map<String, Object> kwargs, FuncallExpression ast)
          throws EvalException {
        try {
          RuleClass ruleClass = ruleFactory.getRuleClass(ruleClassName);
          RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
          builder.createAndAddRepositoryRule(ruleClass, bindRuleClass, kwargs, ast);
        } catch (RuleFactory.InvalidRuleException | Package.NameConflictException |
            Label.SyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
        return NONE;
      }
    };
  }

  private Environment createWorkspaceEnv(
      Builder builder, RuleClassProvider ruleClassProvider, String installDir) {
    Environment workspaceEnv = new Environment();
    MethodLibrary.setupMethodEnvironment(workspaceEnv);
    workspaceEnv.setLoadingPhase();

    RuleFactory ruleFactory = new RuleFactory(ruleClassProvider);
    for (String ruleClass : ruleFactory.getRuleClassNames()) {
      BaseFunction ruleFunction = newRuleFunction(ruleFactory, builder, ruleClass);
      workspaceEnv.update(ruleClass, ruleFunction);
    }

    if (installDir != null) {
      workspaceEnv.update("__embedded_dir__", installDir);
    }
    File jreDirectory = new File(System.getProperty("java.home"));
    workspaceEnv.update("DEFAULT_SERVER_JAVABASE", jreDirectory.getParentFile().toString());

    workspaceEnv.update("bind", newBindFunction(ruleFactory, builder));
    workspaceEnv.update("workspace", newWorkspaceNameFunction(builder));
    return workspaceEnv;
  }
}
