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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ExternalPackage.Binding;
import com.google.devtools.build.lib.packages.ExternalPackage.ExternalPackageBuilder;
import com.google.devtools.build.lib.packages.ExternalPackage.ExternalPackageBuilder.NoSuchBindingException;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * A SkyFunction to parse WORKSPACE files.
 */
public class WorkspaceFileFunction implements SkyFunction {

  private static final String BIND = "bind";

  private PackageFactory packageFactory;

  public WorkspaceFileFunction(PackageFactory packageFactory) {
    this.packageFactory = packageFactory;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws WorkspaceFileFunctionException,
      InterruptedException {
    // TODO(bazel-team): correctness in the presence of changes to the WORKSPACE file.
    Path workspaceFilePath = (Path) skyKey.argument();
    WorkspaceNameHolder holder = new WorkspaceNameHolder();
    ExternalPackageBuilder builder = new ExternalPackageBuilder(workspaceFilePath);
    StoredEventHandler localReporter = new StoredEventHandler();
    BuildFileAST buildFileAST;
    ParserInputSource inputSource = null;
    try {
      inputSource = ParserInputSource.create(workspaceFilePath);
    } catch (IOException e) {
      throw new WorkspaceFileFunctionException(skyKey, e);
    }
    buildFileAST = BuildFileAST.parseBuildFile(inputSource, localReporter, null, false);
    if (buildFileAST.containsErrors()) {
      localReporter.handle(Event.error("WORKSPACE file could not be parsed"));
    } else {
      try {
        if (!evaluateWorkspaceFile(buildFileAST, holder, builder)) {
          localReporter.handle(
              Event.error("Error evaluating WORKSPACE file " + workspaceFilePath));
        }
      } catch (NoSuchBindingException | InvalidRuleException | NameConflictException e) {
        localReporter.handle(Event.error(e.getMessage()));
      }
    }

    builder.addEvents(localReporter.getEvents());
    return new WorkspaceFileValue(holder.workspaceName, builder.build());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static Function newWorkspaceNameFunction(final WorkspaceNameHolder holder) {
    List<String> params = ImmutableList.of("name");
    return new MixedModeFunction("workspace", params, 1, true) {
      @Override
      public Object call(Object[] namedArgs, List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments, FuncallExpression ast) throws EvalException,
          ConversionException, InterruptedException {
        String name = Type.STRING.convert(namedArgs[0], "'name' argument");
        String errorMessage = LabelValidator.validateTargetName(name);
        if (errorMessage != null) {
          throw new EvalException(ast.getLocation(), errorMessage);
        }
        holder.workspaceName = name;
        return com.google.devtools.build.lib.syntax.Environment.NONE;
      }
    };
  }

  private static Function newBindFunction(final ExternalPackageBuilder builder) {
    List<String> params = ImmutableList.of("name", "actual");
    return new MixedModeFunction(BIND, params, 2, true) {
      @Override
      public Object call(Object[] namedArgs, List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments, FuncallExpression ast)
              throws EvalException, ConversionException {
        String name = Type.STRING.convert(namedArgs[0], "'name' argument");
        String actual = Type.STRING.convert(namedArgs[1], "'actual' argument");

        Label nameLabel = null;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name);
          builder.addBinding(
              nameLabel, new Binding(Label.parseWorkspaceLabel(actual), ast.getLocation()));
        } catch (SyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }

        return com.google.devtools.build.lib.syntax.Environment.NONE;
      }
    };
  }

  public boolean evaluateWorkspaceFile(BuildFileAST buildFileAST, WorkspaceNameHolder holder,
      ExternalPackageBuilder builder)
          throws InterruptedException, NoSuchBindingException, InvalidRuleException,
          NameConflictException {
    // Environment is defined in SkyFunction and the syntax package.
    com.google.devtools.build.lib.syntax.Environment workspaceEnv =
        new com.google.devtools.build.lib.syntax.Environment();
    workspaceEnv.update(BIND, newBindFunction(builder));
    workspaceEnv.update("workspace", newWorkspaceNameFunction(holder));

    StoredEventHandler eventHandler = new StoredEventHandler();
    if (!buildFileAST.exec(workspaceEnv, eventHandler)) {
      return false;
    }

    builder.resolveBindTargets(packageFactory.getRuleClass(BIND));
    return true;
  }

  private static final class WorkspaceNameHolder {
    String workspaceName;
  }

  private static final class WorkspaceFileFunctionException extends SkyFunctionException {
    public WorkspaceFileFunctionException(SkyKey key, IOException e) {
      super(key, e);
    }
  }
}
