// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.ATTRIBUTES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.NATIVE;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.REPOSITORIES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.RULE_CLASS;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedFileValue;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/** A SkyFunction to parse WORKSPACE files into a StarlarkFile. */
public class WorkspaceASTFunction implements SkyFunction {
  private final RuleClassProvider ruleClassProvider;

  public WorkspaceASTFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, WorkspaceASTFunctionException {
    RootedPath workspaceRoot = (RootedPath) skyKey.argument();

    Optional<RootedPath> resolvedFile =
        RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE.get(env);
    if (resolvedFile == null) {
      return null;
    }
    String newWorkspaceFileContents = null;
    FileValue workspaceFileValue = null;
    if (resolvedFile.isPresent()) {
      newWorkspaceFileContents = workspaceFromResolvedValue(resolvedFile.get(), env);
      if (newWorkspaceFileContents == null) {
        return null;
      }
    } else {
      workspaceFileValue = (FileValue) env.getValue(FileValue.key(workspaceRoot));
      if (workspaceFileValue == null) {
        return null;
      }
    }
    @Nullable StarlarkSemantics semantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    FileOptions options =
        FileOptions.builder()
            // These three options follow BUILD norms, but should probably be flipped.
            .allowToplevelRebinding(true)
            .requireLoadStatementsFirst(false)
            .recordScope(false)
            .restrictStringEscapes(
                semantics != null && semantics.incompatibleRestrictStringEscapes())
            .build();

    Path repoWorkspace = workspaceRoot.getRoot().getRelative(workspaceRoot.getRootRelativePath());
    try {
      StarlarkFile file =
          StarlarkFile.parse(
              ParserInput.fromString(
                  ruleClassProvider.getDefaultWorkspacePrefix(), "/DEFAULT.WORKSPACE"),
              options);
      if (!file.ok()) {
        Event.replayEventsOn(env.getListener(), file.errors());
        throw resolvedValueError("Failed to parse default WORKSPACE file");
      }
      if (newWorkspaceFileContents != null) {
        file =
            StarlarkFile.parseWithPrelude(
                ParserInput.fromString(
                    newWorkspaceFileContents, resolvedFile.get().asPath().toString()),
                file.getStatements(),
                // The WORKSPACE.resolved file breaks through the usual privacy mechanism.
                options.toBuilder().allowLoadPrivateSymbols(true).build());
      } else if (workspaceFileValue.exists()) {
        byte[] bytes =
            FileSystemUtils.readWithKnownFileSize(repoWorkspace, repoWorkspace.getFileSize());
        file =
            StarlarkFile.parseWithPrelude(
                ParserInput.fromLatin1(bytes, repoWorkspace.toString()),
                file.getStatements(),
                options);
        if (!file.ok()) {
          Event.replayEventsOn(env.getListener(), file.errors());
          throw resolvedValueError("Failed to parse WORKSPACE file");
        }
      }

      String suffix;
      if (resolvedFile.isPresent()) {
        suffix = "";
      } else if (semantics == null) {
        // Starlark semantics was not found, but Skyframe is happy. That means we're in the test
        // that didn't provide complete Skyframe environment. Just move along.
        suffix = ruleClassProvider.getDefaultWorkspaceSuffix();
        // TODO(hlopko): Uncomment once Bazel tests pass with --all_incompatible_changes
        // } else if (semantics.incompatibleUseCcConfigureFromRulesCc()) {
        //   suffix = ruleClassProvider.getDefaultWorkspaceSuffix();
      } else if (!ruleClassProvider.getDefaultWorkspaceSuffix().contains("sh_configure")) {
        // It might look fragile to check for sh_configure in the WORKSPACE file, but it turns
        // out its the best approximation. The problem is that some tests want the ruleClassProvider
        // together with logic from BazelRulesModule, some tests only want the
        // BazelRuleClassProvider and some only a subset of that.
        suffix = ruleClassProvider.getDefaultWorkspaceSuffix();
      } else {
        suffix =
            ruleClassProvider.getDefaultWorkspaceSuffix()
                + "\nload('@bazel_tools//tools/cpp:cc_configure.bzl', 'cc_configure')\n\n"
                + "cc_configure()";
      }

      file =
          StarlarkFile.parseWithPrelude(
              ParserInput.fromString(suffix, "/DEFAULT.WORKSPACE.SUFFIX"),
              file.getStatements(),
              // The DEFAULT.WORKSPACE.SUFFIX file breaks through the usual privacy mechanism.
              options.toBuilder().allowLoadPrivateSymbols(true).build());
      if (!file.ok()) {
        Event.replayEventsOn(env.getListener(), file.errors());
        throw resolvedValueError("Failed to parse default WORKSPACE file suffix");
      }
      return new WorkspaceASTValue(splitAST(file));
    } catch (IOException ex) {
      throw new WorkspaceASTFunctionException(ex, Transience.TRANSIENT);
    }
  }

  private static WorkspaceASTFunctionException resolvedValueError(String message) {
    return new WorkspaceASTFunctionException(
        new BuildFileContainsErrorsException(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER, message),
        Transience.PERSISTENT);
  }

  /**
   * Return the contents of the WORKSPACE file that is implicitly represented by the resolved value
   * found in the given file.
   *
   * <p>TODO(aehlig): at the moment we serialize the value as a string just to re-parse it
   * immediately again; we probably should construct the statements directly out of the value to
   * improve performance.
   */
  private static String workspaceFromResolvedValue(RootedPath resolvedPath, Environment env)
      throws WorkspaceASTFunctionException, InterruptedException {
    ResolvedFileValue resolvedValue =
        (ResolvedFileValue) env.getValue(ResolvedFileValue.key(resolvedPath));
    if (resolvedValue == null) {
      return null;
    }
    List<Map<String, Object>> resolved = resolvedValue.getResolvedValue();
    StringBuilder builder = new StringBuilder();
    for (Map<String, Object> entry : resolved) {
      Object repositories = entry.get(REPOSITORIES);
      if (repositories != null) {
        if (!(repositories instanceof List)) {
          throw resolvedValueError(
              "In 'resolved' the " + REPOSITORIES + " entry is or not a list for item " + entry);
        }
        for (Object repo : (List) repositories) {
          if (!(repo instanceof Map)) {
            throw resolvedValueError("A description of an individual repository is not a map");
          }
          Object rule = ((Map) repo).get(RULE_CLASS);
          if (!(rule instanceof String)) {
            throw resolvedValueError("Expected " + RULE_CLASS + " to be a string.");
          }
          int separatorPosition = ((String) rule).lastIndexOf('%');
          if (separatorPosition < 0) {
            throw resolvedValueError("Malformed rule class: " + ((String) rule));
          }
          String fileName = ((String) rule).substring(0, separatorPosition);
          String symbol = ((String) rule).substring(separatorPosition + 1);

          Object args = ((Map) repo).get(ATTRIBUTES);
          if (!(args instanceof Map)) {
            throw resolvedValueError("Arguments for " + ((String) rule) + " not a dict.");
          }

          builder
              .append("load(\"")
              .append(fileName)
              .append("\", \"")
              .append(symbol)
              .append("\")\n");
          builder.append(symbol).append("(\n");
          for (Map.Entry<?, ?> arg : ((Map<?, ?>) args).entrySet()) {
            Object key = arg.getKey();
            if (!(key instanceof String)) {
              throw resolvedValueError(
                  "In arguments to " + ((String) rule) + " found a non-string key.");
            }
            builder.append("    ").append((String) key).append(" = ");
            builder.append(Printer.getPrinter().repr(arg.getValue()).toString());
            builder.append(",\n");
          }
          builder.append(")\n\n");
        }
      }
      Object nativeEntry = entry.get(NATIVE);
      if (nativeEntry != null) {
        if (!(nativeEntry instanceof String)) {
          throw resolvedValueError(
              "In 'resolved' the " + NATIVE + " entry is not a string for item " + entry);
        }
        builder.append(nativeEntry).append("\n");
      }
    }
    return builder.toString();
  }

  /**
   * Cut {@code ast} into a list of AST separated by load statements. We cut right before each load
   * statement series.
   */
  private static ImmutableList<StarlarkFile> splitAST(StarlarkFile ast) {
    ImmutableList.Builder<StarlarkFile> asts = ImmutableList.builder();
    int prevIdx = 0;
    boolean lastIsLoad = true; // don't cut if the first statement is a load.
    List<Statement> statements = ast.getStatements();
    for (int idx = 0; idx < statements.size(); idx++) {
      Statement st = statements.get(idx);
      if (st instanceof LoadStatement) {
        if (!lastIsLoad) {
          asts.add(ast.subTree(prevIdx, idx));
          prevIdx = idx;
        }
        lastIsLoad = true;
      } else {
        lastIsLoad = false;
      }
    }
    if (!statements.isEmpty()) {
      asts.add(ast.subTree(prevIdx, statements.size()));
    }
    return asts.build();
  }

  private static final class WorkspaceASTFunctionException extends SkyFunctionException {
    WorkspaceASTFunctionException(BuildFileContainsErrorsException e, Transience transience) {
      super(e, transience);
    }
    WorkspaceASTFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
