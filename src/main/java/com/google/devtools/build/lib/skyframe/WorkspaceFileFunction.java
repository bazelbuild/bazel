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
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue.NoSuchBindingException;
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

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    Path workspaceFilePath = (Path) skyKey.argument();
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
      throw WorkspaceFileFunctionException.syntaxError(skyKey,
          "WORKSPACE file could not be parsed into an AST");
    }

    WorkspaceFileValueBuilder builder = new WorkspaceFileValueBuilder();
    if (!evaluateWorkspaceFile(buildFileAST, builder)) {
      throw WorkspaceFileFunctionException.syntaxError(skyKey,
          "Error evaluating WORKSPACE file " + workspaceFilePath);
    }
    try {
      return builder.build();
    } catch (NoSuchBindingException e) {
      throw new WorkspaceFileFunctionException(skyKey, e);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static Function newRepositoryFunction(final WorkspaceFileValueBuilder builder) {
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
        builder.setWorkspaceName(name);
        return null;
      }
    };
  }

  private static Function newBindFunction(final WorkspaceFileValueBuilder builder) {
    List<String> params = ImmutableList.of("name", "actual");
    return new MixedModeFunction("bind", params, 2, true) {
      @Override
      public Object call(Object[] namedArgs, List<Object> surplusPositionalArguments,
          Map<String, Object> surplusKeywordArguments, FuncallExpression ast)
              throws EvalException, ConversionException {
        String name = Type.STRING.convert(namedArgs[0], "'name' argument");
        String actual = Type.STRING.convert(namedArgs[1], "'actual' argument");

        Label nameLabel = null;
        try {
          nameLabel = Label.parseAbsolute("//external:" + name);
          builder.addBinding(nameLabel, new ExternalPackage.Binding(
              Label.parseWorkspaceLabel(actual), ast.getLocation()));
        } catch (SyntaxException e) {
          throw new EvalException(ast.getLocation(), e.getMessage());
        }

        return null;
      }
    };
  }

  public boolean evaluateWorkspaceFile(BuildFileAST buildFileAST,
      WorkspaceFileValueBuilder builder) throws InterruptedException {
    // Environment is defined in SkyFunction and the syntax package.
    com.google.devtools.build.lib.syntax.Environment workspaceEnv =
        new com.google.devtools.build.lib.syntax.Environment();
    workspaceEnv.update("bind", newBindFunction(builder));
    workspaceEnv.update("workspace", newRepositoryFunction(builder));

    StoredEventHandler eventHandler = new StoredEventHandler();
    return buildFileAST.exec(workspaceEnv, eventHandler);
  }

  private static final class WorkspaceFileFunctionException extends SkyFunctionException {
    public WorkspaceFileFunctionException(SkyKey key, Throwable cause) {
      super(key, cause);
    }

    public static WorkspaceFileFunctionException syntaxError(SkyKey key, String message) {
      return new WorkspaceFileFunctionException(key, new SyntaxException(message));
    }
  }
}
