// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildtool.BuildTool.ExitException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback.IoExceptionInterruptedException;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.ClientOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper;
import com.google.devtools.build.lib.runtime.QueryRuntimeHelper.QueryRuntimeHelperException;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.ParsedOptionDescription;
import java.io.IOException;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Version of {@link BuildTool} that handles all work for queries based on results from the analysis
 * phase.
 */
public abstract class PostAnalysisQueryProcessor<T> implements BuildTool.AnalysisPostProcessor {

  private static final Set<String> LAUNCHER_OPTIONS =
      Set.of("config", "deleted_packages", "package_path", "override_repository", "override_module");

  private final QueryExpression queryExpression;
  protected final TargetPattern.Parser mainRepoTargetParser;

  PostAnalysisQueryProcessor(
      QueryExpression queryExpression, TargetPattern.Parser mainRepoTargetParser) {
    this.queryExpression = queryExpression;
    this.mainRepoTargetParser = mainRepoTargetParser;
  }

  @Override
  public void process(
      BuildRequest request,
      CommandEnvironment env,
      BlazeRuntime runtime,
      AnalysisResult analysisResult)
      throws InterruptedException, ViewCreationFailedException, ExitException {
    if (queryExpression == null) {
      return;
    }

    if (!env.getSkyframeExecutor().tracksStateForIncrementality()) {
      throw new ExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      "Queries based on analysis results are not allowed if incrementality state"
                          + " is not being kept. Pass --track_incremental_state to enable this"
                          + " feature.")
                  .setQuery(Query.newBuilder().setCode(Query.Code.ANALYSIS_QUERY_PREREQ_UNMET))
                  .build()));
    }

    CommonQueryOptions options = getQueryOptions(env);
    CqueryOptions cqueryOptions = options instanceof CqueryOptions configured ? configured : null;
    boolean hashing = cqueryOptions != null && cqueryOptions.getGenerateHashes();
    boolean topLevelHashOutput = cqueryOptions != null && cqueryOptions.useTopLevelHashOutput();
    boolean hashStreamReceipt = cqueryOptions != null && cqueryOptions.getHashStreamReceipt();
    // With the label file or literal //... as the entire query, analysis already found every
    // top-level target the formatter could emit. Other expressions may select a subset.
    boolean useAnalyzedRoots =
        cqueryOptions != null
            && topLevelHashOutput
            && (request.targetsAreLabels()
                || (cqueryOptions.getUniverseScope().isEmpty()
                    && queryExpression instanceof TargetLiteral literal
                    && literal.getPattern().equals("//...")
                    && request.getTargets().equals(ImmutableList.of("//..."))))
            && !request.getLoadingOptions().getCompileOneDependency()
            && !request.getLoadingOptions().getBuildTestsOnly();
    // Forward-only hashing starts from this analysis's roots. Keep unrelated dirty
    // values available for change pruning. General queries still clean up old nodes
    // because reverse traversals can see state left behind by earlier commands.
    if (!useAnalyzedRoots) {
      env.getSkyframeExecutor()
          .deleteOldNodes(/* versionWindowForDirtyGc= */ 0, /* keepChangePrunableNodes= */ false);
    }
    env.getSkyframeExecutor().applyInvalidation(env.getReporter());
    var topLevelConfigurations =
        new TopLevelConfigurations(
            useAnalyzedRoots ? ImmutableList.of() : analysisResult.getTopLevelTargetsWithConfigs(),
            topLevelHashOutput ? analysisResult.getTargetsToBuild() : Set.of(),
            useAnalyzedRoots);
    try (QueryRuntimeHelper queryRuntimeHelper =
        env.getRuntime().getQueryRuntimeHelperFactory().create(env, options)) {
      if (hashStreamReceipt && (!useAnalyzedRoots || !options.getLineTerminator().equals("\n"))) {
        throw new OptionsParsingException(
            "A hash stream receipt requires top-level hashes for //... or a canonical label file"
                + " with LF line endings");
      }
      doPostAnalysisQuery(
          request,
          env,
          runtime,
          topLevelConfigurations,
          analysisResult.getAspectsMap(),
          useAnalyzedRoots
              ? ImmutableList.of()
              : env.getSkyframeExecutor().getTransitiveConfigurationKeys(),
          queryRuntimeHelper,
          queryExpression,
          hashing,
          hashStreamReceipt);
    } catch (QueryException e) {
      String errorMessage = "Error doing post analysis query";
      if (!request.getKeepGoing() || hashing) {
        throw new ViewCreationFailedException(errorMessage, e.getFailureDetail(), e);
      }
      env.getReporter().error(null, errorMessage + ": " + e.getFailureDetail().getMessage());
    } catch (IOException e) {
      String errorMessage = "I/O error doing post analysis query";
      FailureDetail failureDetail =
          FailureDetail.newBuilder()
              .setMessage(errorMessage + ": " + e.getMessage())
              .setQuery(Query.newBuilder().setCode(Query.Code.OUTPUT_FORMATTER_IO_EXCEPTION))
              .build();
      if (!request.getKeepGoing() || hashing) {
        throw new ViewCreationFailedException(errorMessage, failureDetail, e);
      }
      env.getReporter().error(null, failureDetail.getMessage());
    } catch (QueryRuntimeHelperException e) {
      throw new ExitException(DetailedExitCode.of(e.getFailureDetail()));
    } catch (OptionsParsingException e) {
      throw new ExitException(
          DetailedExitCode.of(
              ExitCode.COMMAND_LINE_ERROR,
              FailureDetail.newBuilder()
                  .setMessage(e.getMessage())
                  .setActionQuery(
                      ActionQuery.newBuilder().setCode(ActionQuery.Code.INCORRECT_ARGUMENTS))
                  .build()));
    }
  }

  private static String workerOptionsFingerprint(CommandEnvironment env) throws IOException {
    var options = env.getOptions();
    var client = options.getOptions(ClientOptions.class);
    var sources = client.getRcSource();
    if (sources.isEmpty()) {
      throw new IOException("Missing Bazel RC sources for configured hash stream");
    }
    var workspace = env.getWorkspace().resolveSymbolicLinks();
    var external = new HashSet<Integer>();
    for (int i = 0; i < sources.size(); i++) {
      String source = sources.get(i);
      if (!source.equals("client")
          && !workspace.getRelative(source).resolveSymbolicLinks().startsWith(workspace)) {
        external.add(i);
      }
    }
    // Workers attach per-invocation BES headers, including in inactive rc configs. A
    // header-shaped token can still be the value of another option, such as --define.
    var optionValues =
        options.asCompleteListOfParsedOptions().stream()
            .map(ParsedOptionDescription::getUnconvertedValue)
            .collect(toSet());
    var defaults = ImmutableList.<String>builder();
    for (var override : client.getOptionsOverrides()) {
      String arg = override.getOption();
      if (external.contains(override.getRcIndex())
          && (!arg.startsWith("--bes_header=") || optionValues.contains(arg))) {
        defaults.add(override.getCommand(), arg);
      }
    }
    // Workspace defaults are versioned. Compare worker RC defaults and launcher choices
    // without exposing client environment or storing the option values in the receipt.
    var explicit =
        options.asListOfExplicitOptions().stream()
            .filter(
                option -> option.getPriority().getPriorityCategory() == PriorityCategory.COMMAND_LINE)
            .filter(option -> LAUNCHER_OPTIONS.contains(option.getOptionDefinition().getOptionName()))
            .map(option -> option.getCanonicalForm())
            .toList();
    return new Fingerprint().addStrings(defaults.build()).addStrings(explicit).hexDigestAndReset();
  }

  protected abstract CommonQueryOptions getQueryOptions(CommandEnvironment env);

  protected abstract PostAnalysisQueryEnvironment<T> getQueryEnvironment(
      BuildRequest request,
      CommandEnvironment env,
      TopLevelConfigurations topLevelConfigurations,
      ImmutableMap<String, BuildConfigurationValue> transitiveConfigurations,
      ImmutableMap<AspectKeyCreator.AspectKey, ConfiguredAspect> topLevelAspects,
      WalkableGraph walkableGraph)
      throws InterruptedException;

  private static ImmutableMap<String, BuildConfigurationValue> getTransitiveConfigurations(
      Collection<SkyKey> transitiveConfigurationKeys, WalkableGraph graph)
      throws InterruptedException {
    // BuildConfigurationKey and BuildConfigurationValue should be 1:1
    // so merge function intentionally omitted
    return graph.getSuccessfulValues(transitiveConfigurationKeys).values().stream()
        .map(BuildConfigurationValue.class::cast)
        .sorted(Comparator.comparing(BuildConfigurationValue::checksum))
        .collect(toImmutableMap(BuildConfigurationValue::checksum, Function.identity()));
  }

  private void doPostAnalysisQuery(
      BuildRequest request,
      CommandEnvironment env,
      BlazeRuntime runtime,
      TopLevelConfigurations topLevelConfigurations,
      ImmutableMap<AspectKeyCreator.AspectKey, ConfiguredAspect> topLevelAspects,
      Collection<SkyKey> transitiveConfigurationKeys,
      QueryRuntimeHelper queryRuntimeHelper,
      QueryExpression queryExpression,
      boolean hashing,
      boolean hashStreamReceipt)
      throws InterruptedException,
          QueryException,
          IOException,
          QueryRuntimeHelperException,
          OptionsParsingException {
    WalkableGraph walkableGraph = env.getSkyframeExecutor().getWalkableGraph();
    ImmutableMap<String, BuildConfigurationValue> transitiveConfigurations =
        getTransitiveConfigurations(transitiveConfigurationKeys, walkableGraph);

    PostAnalysisQueryEnvironment<T> postAnalysisQueryEnvironment =
        getQueryEnvironment(
            request,
            env,
            topLevelConfigurations,
            transitiveConfigurations,
            topLevelAspects,
            walkableGraph);

    var queryOutput = queryRuntimeHelper.getOutputStreamForQueryOutput();
    Iterable<NamedThreadSafeOutputFormatterCallback<T>> callbacks =
        postAnalysisQueryEnvironment.getDefaultOutputFormatters(
            postAnalysisQueryEnvironment.getAccessor(),
            env.getReporter(),
            queryOutput,
            env.getSkyframeExecutor(),
            runtime.getRuleClassProvider(),
            env.getPackageManager(),
            env.getClientEnv(),
            env.getSkyframeExecutor()
                .getEffectiveStarlarkSemantics(
                    env.getOptions().getOptions(BuildLanguageOptions.class)));
    String outputFormat = postAnalysisQueryEnvironment.getOutputFormat();
    NamedThreadSafeOutputFormatterCallback<T> callback =
        NamedThreadSafeOutputFormatterCallback.selectCallback(outputFormat, callbacks);
    if (callback == null) {
      throw new OptionsParsingException(
          String.format(
              "Invalid output format '%s'. Valid values are: %s",
              outputFormat, NamedThreadSafeOutputFormatterCallback.callbackNames(callbacks)));
    }

    // A certain subset of output formatters support "streaming" results - the formatter is called
    // multiple times where each call has only a some of the full query results (see
    // StreamedOutputFormatter for details). cquery and aquery don't do this. But the reason is
    // subtle and hard to follow. Post-analysis output formatters inherit from Callback, which
    // declares "void process(Iterable<T> partialResult)". Its javadoc says that the subinterface
    // BatchCallback may stream partial results. But post-analysis callbacks don't inherit
    // BatchCallback!
    //
    // To protect against accidental feature regression (like implementing a callback that
    // accidentally inherits BatchCallback), we explicitly disable streaming here. The aggregating
    // callback collects the entire query's results, even if the query was evaluated in a streaming
    // manner. Note that streaming query evaluation is a distinct concept from streaming output
    // formatting. Once the complete query finishes, we replay the full results back to the original
    // callback. That way callback implementations can safely assume they're only called once and
    // the results for that call are indeed complete.
    AggregateAllOutputFormatterCallback<T, Set<T>> aggregateResultsCallback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(postAnalysisQueryEnvironment);
    QueryEvalResult result =
        postAnalysisQueryEnvironment.evaluateQuery(queryExpression, aggregateResultsCallback);
    if (result.isEmpty()) {
      env.getReporter().handle(Event.info("Empty query results"));
    }
    try {
      if (hashStreamReceipt) {
        String header =
            "# bazel-configured-root-hashes-v1\t"
                + env.getCommandId()
                + "\t"
                + workerOptionsFingerprint(env)
                + "\n";
        queryOutput.write(header.getBytes(UTF_8));
        if (request.targetsAreLabels()) {
          // Loading certified the complete scope. Preserve analysis failures and incompatible
          // targets as explicit unknown hashes, alongside any failed digests from the formatter.
          var analyzed = Objects.requireNonNull(topLevelConfigurations.getAnalyzedRoots());
          var analyzedLabels = HashSet.<String>newHashSet(analyzed.size());
          for (var target : analyzed) {
            analyzedLabels.add(target.getOriginalLabel().getCanonicalForm());
          }
          for (String label : request.getTargets()) {
            if (!analyzedLabels.contains(label)) {
              queryOutput.write((internalToUnicode(label) + "\tnone\t\t\n").getBytes(UTF_8));
            }
          }
        }
      }
      callback.start();
      callback.process(aggregateResultsCallback.getResult());
      // Under --keep_going, query evaluation completed and partial results were replayed above.
      // Therefore, close() should not fail fast, allowing output formatters to flush
      // streams and write closing proto/JSON structures.
      callback.close(/* failFast= */ !result.getSuccess() && !request.getKeepGoing());
    } catch (IoExceptionInterruptedException e) {
      throw (IOException) e.getCause();
    }

    queryRuntimeHelper.afterQueryOutputIsWritten();
    if (hashing && !result.getSuccess()) {
      throw new QueryException(
          queryExpression,
          "Errors encountered while evaluating the hash query",
          result.getDetailedExitCode().getFailureDetail());
    }
  }
}
