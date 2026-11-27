// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.runtime.Command.BuildPhase.LOADS;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashFunction;
import com.google.common.io.CountingOutputStream;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.hash.DirectTargetHashFunction;
import com.google.devtools.build.lib.query2.query.hash.TransitiveHashes;
import com.google.devtools.build.lib.query2.query.output.HashOutputFormatter;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.query2.query.output.QueryOutputUtils;
import com.google.devtools.build.lib.query2.query.output.StreamedFormatter;
import com.google.devtools.build.lib.query2.query.output.StreamedProtoOutputFormatter;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Either;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.skyframe.EvaluationContext;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.ClosedByInterruptException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** Command line wrapper for executing a query with blaze. */
@Command(
    name = "query",
    buildPhase = LOADS,
    options = {
      CoreOptions.class, // for --action_env, which affects the repo env
      PackageOptions.class,
      QueryOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    help = "resource:query.txt",
    shortDescription = "Executes a dependency graph query.",
    allowResidue = true,
    binaryStdOut = true,
    completion = "label")
public final class QueryCommand extends QueryEnvironmentBasedCommand {
  private static final HexFormat HEX_FORMAT = HexFormat.of();
  private static final int MIN_ORDERED_HASH_KEYS = 4_096;

  @Override
  protected Either<BlazeCommandResult, QueryEvalResult> doQuery(
      String query,
      CommandEnvironment env,
      QueryOptions queryOptions,
      boolean streamResults,
      OutputFormatter formatter,
      AbstractBlazeQueryEnvironment<Target> queryEnv,
      QueryRuntimeHelper queryRuntimeHelper) {
    QueryExpression expr;
    try (SilentCloseable closeable = Profiler.instance().profile("QueryExpression.parse")) {
      expr = QueryExpression.parse(query, queryEnv, queryOptions.toParserOptions());
    } catch (QuerySyntaxException e) {
      String message =
          String.format(
              "Error while parsing '%s': %s", QueryExpression.truncate(query), e.getMessage());
      env.getReporter().handle(Event.error(null, message));
      return Either.ofLeft(
          BlazeCommandResult.detailedExitCode(
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(e.getMessage())
                      .setQuery(
                          FailureDetails.Query.newBuilder()
                              .setCode(FailureDetails.Query.Code.SYNTAX_ERROR))
                      .build())));
    }

    try {
      if (!queryOptions.getGenerateHashes()) {
        formatter.verifyCompatible(queryEnv, expr);
      }
    } catch (QueryException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return Either.ofLeft(BlazeCommandResult.failureDetail(e.getFailureDetail()));
    }

    expr = queryEnv.transformParsedQuery(expr);

    // This only applies to --order_output=auto. Instead of being written directly to the stream
    // by the callback, this option aggregates the results in the lexicographically sorted
    // aggregator first before using the StreamedFormatter to write it to stream later.
    // An exception to this is when somepath is used at the top level of the query expression.
    boolean lexicographicallySortOutput =
        QueryOutputUtils.lexicographicallySortOutput(queryOptions, formatter)
            && !expr.isSomePathFunction();

    OutputStream out;
    if (formatter.canBeBuffered()) {
      // There is no particular reason for the 16384 constant here, except its a multiple of the
      // gRPC buffer size. We mainly don't want to send each label individually because the output
      // stream is connected to gRPC, and every write gets converted to one gRPC call.
      out = new BufferedOutputStream(queryRuntimeHelper.getOutputStreamForQueryOutput(), 16384);
    } else {
      out = queryRuntimeHelper.getOutputStreamForQueryOutput();
    }

    ThreadSafeOutputFormatterCallback<Target> callback;
    HashFunction hashFunction =
        env.getRuntime().getFileSystem().getDigestFunction().getHashFunction();
    boolean generateHashes = queryOptions.getGenerateHashes();
    if (generateHashes) {
      // Hashes are computed after evaluation, so aggregate the results instead of streaming them.
      callback = QueryUtil.newLexicographicallySortedTargetAggregator(/* parallelSort= */ true);
    } else if (streamResults) {
      disableAnsiCharactersFiltering(env);
      StreamedFormatter streamedFormatter = ((StreamedFormatter) formatter);
      streamedFormatter.setOptions(
          queryOptions,
          queryOptions.getAspectDeps().createResolver(env.getPackageManager(), env.getReporter()),
          hashFunction);
      streamedFormatter.setEventHandler(env.getReporter());
      if (lexicographicallySortOutput) {
        callback = QueryUtil.newLexicographicallySortedTargetAggregator();
      } else {
        callback = streamedFormatter.createStreamCallback(out, queryOptions, queryEnv);
      }
    } else {
      callback = QueryUtil.newOrderedAggregateAllOutputFormatterCallback(queryEnv);
    }

    QueryEvalResult result;
    long evaluationStart = System.nanoTime();
    long evaluationNanos;
    boolean catastrophe = true;
    try {
      try (SilentCloseable closeable = Profiler.instance().profile("queryEnv.evaluateQuery")) {
        result = queryEnv.evaluateQuery(expr, callback);
        evaluationNanos = System.nanoTime() - evaluationStart;
        catastrophe = false;
      } catch (QueryException e) {
        catastrophe = false;
        // Keep consistent with reportBuildFileError()
        env.getReporter()
            // TODO(bazel-team): this is a kludge to fix a bug observed in the wild. We should make
            // sure no null error messages ever get in.
            .handle(Event.error(e.getMessage() == null ? e.toString() : e.getMessage()));
        return Either.ofLeft(finalizeBlazeCommandResult(ExitCode.ANALYSIS_FAILURE, e));
      } catch (InterruptedException e) {
        catastrophe = false;
        IOException ioException = callback.getIoException();
        if (ioException == null || ioException instanceof ClosedByInterruptException) {
          return reportAndCreateInterruptedResult(env);
        }
        return reportAndCreateIOExceptionResult(env, e.getMessage());
      } catch (IOException e) {
        catastrophe = false;
        return reportAndCreateIOExceptionResult(env, e.getMessage());
      } finally {
        if (!catastrophe) {
          out.flush();
        }
      }
      if (generateHashes) {
        disableAnsiCharactersFiltering(env);
        try (SilentCloseable closeable = Profiler.instance().profile("QueryCommand.outputHashes")) {
          long sortStart = System.nanoTime();
          Set<Target> targets =
              ((AggregateAllOutputFormatterCallback<Target, ?>) callback).getResult();
          long sortNanos = System.nanoTime() - sortStart;
          HashOutputFormatter catalog = null;
          if (formatter instanceof HashOutputFormatter hashed) {
            // Require complete enumeration before writing any snapshot rows.
            if (!result.getSuccess()) {
              return Either.ofRight(result);
            }
            hashed.setOptions(
                queryOptions,
                queryOptions
                    .getAspectDeps()
                    .createResolver(env.getPackageManager(), env.getReporter()),
                hashFunction);
            hashed.setEventHandler(env.getReporter());
            catalog = hashed;
          }
          BlazeCommandResult failure =
              outputHashes(
                  env,
                  queryOptions,
                  targets,
                  out,
                  queryEnv.getLabelPrinter(),
                  catalog,
                  evaluationNanos,
                  sortNanos);
          if (failure != null) {
            return Either.ofLeft(failure);
          }
        } catch (ClosedByInterruptException | InterruptedException e) {
          return reportAndCreateInterruptedResult(env);
        } catch (IOException e) {
          return reportAndCreateIOExceptionResult(env, e.getMessage());
        } finally {
          out.flush();
        }
      } else if (!streamResults || lexicographicallySortOutput) {
        disableAnsiCharactersFiltering(env);
        try (SilentCloseable closeable = Profiler.instance().profile("QueryOutputUtils.output")) {
          Set<Target> targets =
              ((AggregateAllOutputFormatterCallback<Target, ?>) callback).getResult();
          QueryOutputUtils.output(
              queryOptions,
              result,
              targets,
              formatter,
              out,
              queryOptions
                  .getAspectDeps()
                  .createResolver(env.getPackageManager(), env.getReporter()),
              env.getReporter(),
              hashFunction,
              queryEnv.getLabelPrinter());
        } catch (ClosedByInterruptException | InterruptedException e) {
          return reportAndCreateInterruptedResult(env);
        } catch (IOException e) {
          return reportAndCreateIOExceptionResult(env, e.getMessage());
        } finally {
          out.flush();
        }
      }
    } catch (IOException e) {
      return reportAndCreateFlushFailureResult(env, e);
    }

    return Either.ofRight(result);
  }

  /**
   * Writes transitive hashes, including metadata for every target when a catalog is requested.
   * Returns a failure result if a hash could not be computed or may be wrong, or null.
   */
  @Nullable
  private static BlazeCommandResult outputHashes(
      CommandEnvironment env,
      QueryOptions queryOptions,
      Set<Target> targets,
      OutputStream out,
      LabelPrinter labelPrinter,
      @Nullable HashOutputFormatter catalog,
      long evaluationNanos,
      long sortNanos)
      throws IOException, InterruptedException {
    ImmutableSet<Label> labels = targets.stream().map(Target::getLabel).collect(toImmutableSet());
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setParallelism(
                env.getOptions().getOptions(LoadingPhaseThreadsOption.class).getThreads())
            .setEventHandler(env.getReporter())
            .build();
    EvaluationContext orderedEvaluationContext =
        evaluationContext.builder().setOrderedRootResults(true).build();
    Stopwatch skyframeTime = Stopwatch.createUnstarted();
    TransitiveHashes.Result result =
        TransitiveHashes.compute(
            labels,
            keys -> {
              skyframeTime.start();
              try {
                boolean useOrderedValues =
                    keys.size() >= MIN_ORDERED_HASH_KEYS
                        && keys.iterator().next() instanceof DirectTargetHashFunction.Key;
                return env.getSkyframeExecutor()
                    .prepareAndGet(
                        keys, useOrderedValues ? orderedEvaluationContext : evaluationContext);
              } finally {
                skyframeTime.stop();
              }
            },
            message -> env.getReporter().handle(Event.progress(message)),
            queryOptions.getQueryHashFailureImpact(),
            queryOptions.getQueryBatchDirectRuleHashes(),
            getHashProvenancePath(env, queryOptions),
            "1".equals(env.getClientEnv().get("BAZEL_QUERY_HASH_PROVENANCE_EDGES")));
    CountingOutputStream binary = null;
    if (catalog instanceof StreamedProtoOutputFormatter) {
      binary = new CountingOutputStream(out);
      out = binary;
    }
    String lineTerminator = queryOptions.getLineTerminator();
    long writeStart = System.nanoTime();
    if (catalog != null) {
      catalog.writeHashedTargets(targets, labelPrinter, result.hashes(), out);
    } else {
      for (Target target : targets) {
        Label label = target.getLabel();
        var hash = result.hashes().get(label);
        if (hash != null) {
          out.write(
              (labelPrinter.toString(label)
                      + " "
                      + HEX_FORMAT.formatHex(hash.toByteArray())
                      + lineTerminator)
                  .getBytes(UTF_8));
        }
      }
    }
    long writeEnd = System.nanoTime();
    long writeNanos = writeEnd - writeStart;
    if (catalog != null) {
      Profiler.instance().logSimpleTask(
          writeStart, writeEnd, ProfilerTask.INFO, "NativeQuery.writeCatalog");
    }
    result
        .errors()
        .forEach(
            (label, error) ->
                env.getReporter()
                    .handle(
                        Event.error(
                            String.format(
                                "failed to compute hash for %s: %s",
                                labelPrinter.toString(label),
                                error.getException() != null
                                    ? error.getException().getMessage()
                                    : error.toString()))));
    int omitted = labels.size() - result.hashes().size();
    if (binary != null) {
      long flushStart = System.nanoTime();
      binary.flush();
      long outputNanos = writeNanos + (System.nanoTime() - flushStart);
      var timing = result.timing();
      env.getReporter()
          .handle(
              Event.info(
                  String.format(
                      Locale.ROOT,
                      "Native query phases (ms): file_diff=%d evaluate=%d sort=%d discovery=%d"
                          + " skyframe=%d combine=%d assemble=%d output=%d",
                      env.getSkyframeExecutor().getSourceDiffCheckingDuration().toMillis(),
                      TimeUnit.NANOSECONDS.toMillis(evaluationNanos),
                      TimeUnit.NANOSECONDS.toMillis(sortNanos),
                      timing.discoveryMs(),
                      skyframeTime.elapsed(TimeUnit.MILLISECONDS),
                      timing.combineMs(),
                      timing.assembleMs(),
                      TimeUnit.NANOSECONDS.toMillis(outputNanos))));
      env.getReporter()
          .handle(
              Event.info(
                  String.format(
                      "Streamed %d query hash records (%d unknown, %d bytes)",
                      targets.size(), omitted, binary.getCount())));
    }
    if (omitted == 0 && result.errors().isEmpty()) {
      return null;
    }
    result
        .failureImpact()
        .forEach(
            (label, impact) ->
                env.getReporter()
                    .handle(
                        Event.info(
                            String.format(
                                "Unknown hash impact of %s: %d affected, %d exclusive",
                                label, impact.affected(), impact.exclusive()))));
    if (result.unlistedFailureImpact() != 0) {
      env.getReporter().handle(
          Event.info(
              String.format(
                  "Unknown hash impact of remaining failures: %d affected",
                  result.unlistedFailureImpact())));
    }
    String message =
        String.format(
            "omitted %d of %d requested hashes because dependency information was incomplete"
                + " (%d reported errors)",
            omitted, labels.size(), result.errors().size());
    env.getReporter().handle(Event.error(message));
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setQuery(Query.newBuilder().setCode(Code.BUILD_FILE_ERROR))
                .build()));
  }

  @Nullable
  private static Path getHashProvenancePath(CommandEnvironment env, QueryOptions options) {
    String value = env.getClientEnv().get("BAZEL_QUERY_HASH_PROVENANCE");
    if (value == null || value.isEmpty()) {
      return null;
    }
    try {
      var cwd = env.getWorkingDirectory();
      Path path = cwd.getRelative(value).getPathFile().toPath();
      String outputFile = options.getOutputFile();
      if (outputFile != null && !outputFile.isEmpty()) {
        Path primary = cwd.getRelative(outputFile).getPathFile().toPath();
        if (path.equals(primary) || (Files.exists(path) && Files.isSameFile(path, primary))) {
          env.getReporter()
              .handle(Event.warn("Skipping query hash provenance at --output_file: " + path));
          return null;
        }
      }
      return path;
    } catch (IOException | InvalidPathException e) {
      env.getReporter()
          .handle(Event.warn("Could not resolve query hash provenance: " + e.getMessage()));
      return null;
    }
  }

  /**
   * When Blaze is used with --color=no or not in a tty a ansi characters filter is set so that
   * we don't print fancy colors in non-supporting terminal outputs. But query output, specifically
   * the binary formatters, can print actual data that contain ansi bytes/chars. Because of that
   * we need to remove the filtering before printing any query result.
   */
  private static void disableAnsiCharactersFiltering(CommandEnvironment env) {
    env.getReporter().switchToAnsiAllowingHandler();
  }

  private static Either<BlazeCommandResult, QueryEvalResult> reportAndCreateFlushFailureResult(
      CommandEnvironment env, IOException e) {
    String message = "Failed to flush query results: " + e.getMessage();
    env.getReporter().handle(Event.error(message));
    return Either.ofLeft(
        BlazeCommandResult.failureDetail(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setQuery(Query.newBuilder().setCode(Code.QUERY_RESULTS_FLUSH_FAILURE))
                .build()));
  }

  private static Either<BlazeCommandResult, QueryEvalResult> reportAndCreateInterruptedResult(
      CommandEnvironment env) {
    String message = "query interrupted";
    env.getReporter().handle(Event.error(message));
    return Either.ofLeft(
        BlazeCommandResult.detailedExitCode(InterruptedFailureDetails.detailedExitCode(message)));
  }

  private static Either<BlazeCommandResult, QueryEvalResult> reportAndCreateIOExceptionResult(
      CommandEnvironment env, String message) {
    String prefixedMessage = "I/O error: " + message;
    env.getReporter().handle(Event.error(prefixedMessage));
    return Either.ofLeft(
        BlazeCommandResult.failureDetail(
            FailureDetail.newBuilder()
                .setMessage(prefixedMessage)
                .setQuery(Query.newBuilder().setCode(Code.OUTPUT_FORMATTER_IO_EXCEPTION))
                .build()));
  }

  private static BlazeCommandResult finalizeBlazeCommandResult(
      ExitCode exitCode, QueryException e) {
    return BlazeCommandResult.detailedExitCode(DetailedExitCode.of(exitCode, e.getFailureDetail()));
  }
}
