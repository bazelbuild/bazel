// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.ProjectFileSupport;
import com.google.devtools.build.lib.runtime.events.InputFileEvent;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import net.starlark.java.eval.StarlarkSemantics;

/** Provides support for reading target patterns from a file or the command-line. */
public final class TargetPatternsHelper {

  private static final Splitter TARGET_PATTERN_SPLITTER = Splitter.on('#');

  private TargetPatternsHelper() {}

  /**
   * Reads a list of target patterns, either from the command-line residue, by reading newline
   * delimited target patterns from the --target_pattern_file flag, or from --query/--query_file.
   * If multiple options are specified, throws {@link TargetPatternsHelperException}.
   *
   * @return A list of target patterns.
   */
  public static List<String> readFrom(CommandEnvironment env, OptionsParsingResult options)
      throws TargetPatternsHelperException {
    List<String> targets = options.getResidue();
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);

    int optionCount = 0;
    if (!targets.isEmpty()) optionCount++;
    if (!buildRequestOptions.targetPatternFile.isEmpty()) optionCount++;
    if (!buildRequestOptions.query.isEmpty()) optionCount++;
    if (!buildRequestOptions.queryFile.isEmpty()) optionCount++;
    if (optionCount > 1) {
      throw new TargetPatternsHelperException(
          "Only one of command-line target patterns, --target_pattern_file, --query, "
              + "or --query_file may be specified",
          TargetPatterns.Code.TARGET_PATTERN_FILE_WITH_COMMAND_LINE_PATTERN);
    }

    if (!buildRequestOptions.query.isEmpty()) {
      try {
        return executeQuery(env, buildRequestOptions.query, options);
      } catch (QueryException | InterruptedException | IOException e) {
        throw new TargetPatternsHelperException(
            "Error executing query: " + e.getMessage(),
            TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
      }
    } else if (!buildRequestOptions.queryFile.isEmpty()) {
      Path queryFilePath = env.getWorkingDirectory().getRelative(buildRequestOptions.queryFile);
      try {
        env.getEventBus()
            .post(
                InputFileEvent.create(
                    /* type= */ "query_file", queryFilePath.getFileSize()));
        String queryExpression = FileSystemUtils.readContent(queryFilePath, ISO_8859_1).trim();
        return executeQuery(env, queryExpression, options);
      } catch (IOException e) {
        throw new TargetPatternsHelperException(
            "I/O error reading from " + queryFilePath.getPathString() + ": " + e.getMessage(),
            TargetPatterns.Code.TARGET_PATTERN_FILE_READ_FAILURE);
      } catch (QueryException | InterruptedException e) {
        throw new TargetPatternsHelperException(
            "Error executing query from file: " + e.getMessage(),
            TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
      }
    }

    if (!buildRequestOptions.targetPatternFile.isEmpty()) {
      // Works for absolute or relative file.
      Path residuePath =
          env.getWorkingDirectory().getRelative(buildRequestOptions.targetPatternFile);
      try {
        env.getEventBus()
            .post(
                InputFileEvent.create(
                    /* type= */ "target_pattern_file", residuePath.getFileSize()));
        targets =
            FileSystemUtils.readLines(residuePath, ISO_8859_1).stream()
                .map(s -> TARGET_PATTERN_SPLITTER.splitToList(s).get(0))
                .map(String::trim)
                .filter(Predicate.not(String::isEmpty))
                .collect(toImmutableList());
      } catch (IOException e) {
        throw new TargetPatternsHelperException(
            "I/O error reading from " + residuePath.getPathString() + ": " + e.getMessage(),
            TargetPatterns.Code.TARGET_PATTERN_FILE_READ_FAILURE);
      }
    } else {
      try (SilentCloseable closeable =
          Profiler.instance().profile("ProjectFileSupport.getTargets")) {
        targets = ProjectFileSupport.getTargets(env.getRuntime().getProjectFileProvider(), options);
      }
    }
    return targets;
  }

  /** Thrown when target patterns couldn't be read. */
  public static class TargetPatternsHelperException extends Exception {
    private final TargetPatterns.Code detailedCode;

    private TargetPatternsHelperException(String message, TargetPatterns.Code detailedCode) {
      super(Preconditions.checkNotNull(message));
      this.detailedCode = detailedCode;
    }

    public FailureDetail getFailureDetail() {
      return FailureDetail.newBuilder()
          .setMessage(getMessage())
          .setTargetPatterns(TargetPatterns.newBuilder().setCode(detailedCode))
          .build();
    }
  }

  /** Executes a query and returns the resulting target patterns. */
  private static List<String> executeQuery(
      CommandEnvironment env, String queryExpression, OptionsParsingResult options)
      throws QueryException, InterruptedException, IOException, TargetPatternsHelperException {
    try {
      LoadingPhaseThreadsOption threadsOption = options.getOptions(LoadingPhaseThreadsOption.class);
      RepositoryMapping repoMapping =
          env.getSkyframeExecutor()
              .getMainRepoMapping(false, threadsOption.threads, env.getReporter());
      TargetPattern.Parser mainRepoTargetParser =
          new TargetPattern.Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, repoMapping);

      StarlarkSemantics starlarkSemantics =
          options.getOptions(BuildLanguageOptions.class).toStarlarkSemantics();
      LabelPrinter labelPrinter =
          new QueryOptions().getLabelPrinter(starlarkSemantics, mainRepoTargetParser.getRepoMapping());

      AbstractBlazeQueryEnvironment<Target> queryEnv =
          QueryEnvironmentBasedCommand.newQueryEnvironment(
              env,
              /* keepGoing=*/ false,
              /* orderedResults= */ false,
              UniverseScope.EMPTY,
              threadsOption.threads,
              Set.of(),
              /* useGraphlessQuery= */ true,
              mainRepoTargetParser,
              labelPrinter);

      QueryExpression expr = QueryExpression.parse(queryExpression, queryEnv);
      Set<String> targetPatterns = new LinkedHashSet<>();
      ThreadSafeOutputFormatterCallback<Target> callback =
          new ThreadSafeOutputFormatterCallback<Target>() {
            @Override
            public void processOutput(Iterable<Target> partialResult) {
              for (Target target : partialResult) {
                targetPatterns.add(target.getLabel().toString());
              }
            }
          };

      QueryEvalResult result = queryEnv.evaluateQuery(expr, callback);
      if (!result.getSuccess()) {
        throw new TargetPatternsHelperException("Query evaluation failed",
            TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
      }

      return new ArrayList<>(targetPatterns);
    } catch (InterruptedException e) {
      throw new TargetPatternsHelperException("Query interrupted",
          TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
    } catch (RepositoryMappingResolutionException e) {
      throw new TargetPatternsHelperException(e.getMessage(),
          TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
    } catch (QuerySyntaxException e) {
      throw new TargetPatternsHelperException("Query syntax error: " + e.getMessage(),
          TargetPatterns.Code.TARGET_PATTERNS_UNKNOWN);
    }
  }
}
