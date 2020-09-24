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

import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QuerySyntaxException;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.query2.query.output.QueryOutputUtils;
import com.google.devtools.build.lib.query2.query.output.StreamedFormatter;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Either;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.ClosedByInterruptException;
import java.util.Set;

/** Command line wrapper for executing a query with blaze. */
@Command(
    name = "query",
    options = {
      PackageOptions.class,
      QueryOptions.class,
      KeepGoingOption.class,
      LoadingPhaseThreadsOption.class
    },
    help = "resource:query.txt",
    shortDescription = "Executes a dependency graph query.",
    allowResidue = true,
    binaryStdOut = true,
    completion = "label",
    canRunInOutputDirectory = true)
public final class QueryCommand extends QueryEnvironmentBasedCommand {

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
      expr = QueryExpression.parse(query, queryEnv);
    } catch (QuerySyntaxException e) {
      String message =
          String.format(
              "Error while parsing '%s': %s", QueryExpression.truncate(query), e.getMessage());
      env.getReporter().handle(Event.error(null, message));
      return Either.ofLeft(
          BlazeCommandResult.detailedExitCode(
              DetailedExitCode.of(
                  ExitCode.COMMAND_LINE_ERROR,
                  FailureDetail.newBuilder()
                      .setMessage(e.getMessage())
                      .setQuery(
                          FailureDetails.Query.newBuilder()
                              .setCode(FailureDetails.Query.Code.SYNTAX_ERROR))
                      .build())));
    }

    try {
      formatter.verifyCompatible(queryEnv, expr);
    } catch (QueryException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return Either.ofLeft(finalizeBlazeCommandResult(ExitCode.COMMAND_LINE_ERROR, e));
    }

    expr = queryEnv.transformParsedQuery(expr);

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
    if (streamResults) {
      disableAnsiCharactersFiltering(env);
      StreamedFormatter streamedFormatter = ((StreamedFormatter) formatter);
      streamedFormatter.setOptions(
          queryOptions,
          queryOptions.aspectDeps.createResolver(env.getPackageManager(), env.getReporter()),
          hashFunction);
      streamedFormatter.setEventHandler(env.getReporter());
      callback = streamedFormatter.createStreamCallback(out, queryOptions, queryEnv);
    } else {
      callback = QueryUtil.newOrderedAggregateAllOutputFormatterCallback(queryEnv);
    }

    QueryEvalResult result;
    boolean catastrophe = true;
    try {
      try (SilentCloseable closeable = Profiler.instance().profile("queryEnv.evaluateQuery")) {
        result = queryEnv.evaluateQuery(expr, callback);
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
      if (!streamResults) {
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
              queryOptions.aspectDeps.createResolver(env.getPackageManager(), env.getReporter()),
              env.getReporter(),
              hashFunction);
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
        BlazeCommandResult.detailedExitCode(
            InterruptedFailureDetails.detailedExitCode(message, Interrupted.Code.QUERY)));
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
