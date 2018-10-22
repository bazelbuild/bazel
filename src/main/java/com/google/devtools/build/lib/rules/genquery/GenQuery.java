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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ByteStringDeterministicWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.query2.BlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.QueryEnvironmentFactory;
import com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.query2.output.QueryOutputUtils;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.channels.ClosedByInterruptException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An implementation of the 'genquery' rule.
 */
public class GenQuery implements RuleConfiguredTargetFactory {
  private static final QueryEnvironmentFactory QUERY_ENVIRONMENT_FACTORY =
      new QueryEnvironmentFactory();

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    Artifact outputArtifact = ruleContext.createOutputArtifact();

    // The query string
    final String query = ruleContext.attributes().get("expression", Type.STRING);

    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(QueryOptions.class, KeepGoingOption.class);
    optionsParser.setAllowResidue(false);
    try {
      optionsParser.parse(ruleContext.attributes().get("opts", Type.STRING_LIST));
    } catch (OptionsParsingException e) {
      ruleContext.attributeError("opts", "error while parsing query options: " + e.getMessage());
      return null;
    }

    // Parsed query options
    QueryOptions queryOptions = optionsParser.getOptions(QueryOptions.class);
    // If you change the list of options here, also change the documentation of genquery.opts in
    // GenQueryRule.java .
    if (optionsParser.getOptions(KeepGoingOption.class).keepGoing) {
      ruleContext.attributeError("opts", "option --keep_going is not allowed");
      return null;
    }
    if (!queryOptions.universeScope.isEmpty()) {
      ruleContext.attributeError("opts", "option --universe_scope is not allowed");
      return null;
    }
    if (optionsParser.containsExplicitOption("order_results")) {
      ruleContext.attributeError("opts", "option --order_results is not allowed");
      return null;
    }
    if (optionsParser.containsExplicitOption("noorder_results")) {
      ruleContext.attributeError("opts", "option --noorder_results is not allowed");
      return null;
    }
    if (optionsParser.containsExplicitOption("order_output")) {
      ruleContext.attributeError("opts", "option --order_output is not allowed");
      return null;
    }
    // Force results to be deterministic.
    queryOptions.orderOutput = OrderOutput.FULL;

    // force relative_locations to true so it has a deterministic output across machines.
    queryOptions.relativeLocations = true;

    ByteString result =
        executeQuery(
            ruleContext,
            queryOptions,
            ruleContext.attributes().get("scope", BuildType.LABEL_LIST),
            query);
    if (result == null || ruleContext.hasErrors()) {
      return null;
    }

    ruleContext.registerAction(
        new QueryResultAction(ruleContext.getActionOwner(), outputArtifact, result));

    NestedSet<Artifact> filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER, outputArtifact);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, RunfilesProvider.simple(
            new Runfiles.Builder(
                ruleContext.getWorkspaceName(),
                ruleContext.getConfiguration().legacyExternalRunfiles())
                .addTransitiveArtifacts(filesToBuild).build()))
        .build();
  }


  /**
   * DO NOT USE! We should get rid of this method: errors reported directly to this object don't set
   * the error flag in {@link ConfiguredTarget}.
   */
  private ExtendedEventHandler getEventHandler(RuleContext ruleContext) {
    return ruleContext.getAnalysisEnvironment().getEventHandler();
  }

  /**
   * Precomputes the transitive closure of the scope. Returns two maps: one identifying the
   * successful packages, and the other identifying the valid targets. Breaks in the transitive
   * closure of the scope will cause the query to error out early.
   */
  @Nullable
  private static Pair<ImmutableMap<PackageIdentifier, Package>, ImmutableMap<Label, Target>>
      constructPackageMap(SkyFunction.Environment env, Collection<Label> scope)
          throws InterruptedException, BrokenQueryScopeException {
    // It is not necessary for correctness to construct intermediate NestedSets; we could iterate
    // over individual targets in scope immediately. However, creating a composite NestedSet first
    // saves us from iterating over the same sub-NestedSets multiple times.
    NestedSetBuilder<Label> validTargets = NestedSetBuilder.stableOrder();
    Set<SkyKey> successfulPackageKeys = Sets.newHashSetWithExpectedSize(scope.size());
    Map<SkyKey, SkyValue> transitiveTargetValues =
        env.getValues(Collections2.transform(scope, TransitiveTargetKey::of));
    if (env.valuesMissing()) {
      return null;
    }
    for (SkyValue value : transitiveTargetValues.values()) {
      TransitiveTargetValue transNode = (TransitiveTargetValue) value;
      if (transNode.getTransitiveRootCauses() != null) {
        // This should only happen if the unsuccessful package was loaded in a non-selected
        // path, as otherwise this configured target would have failed earlier. See b/34132681.
        throw new BrokenQueryScopeException(
            "errors were encountered while computing transitive closure of the scope.");
      }
      validTargets.addTransitive(transNode.getTransitiveTargets());
      for (Label transitiveLabel : transNode.getTransitiveTargets()) {
        successfulPackageKeys.add(PackageValue.key(transitiveLabel.getPackageIdentifier()));
      }
    }

    // Construct the package id to package map for all successful packages.
    Map<SkyKey, SkyValue> transitivePackages = env.getValues(successfulPackageKeys);
    if (env.valuesMissing()) {
      // Packages from an untaken select branch could be missing: analysis avoids these, but query
      // does not.
      return null;
    }
    ImmutableMap.Builder<PackageIdentifier, Package> packageMapBuilder = ImmutableMap.builder();
    for (Map.Entry<SkyKey, SkyValue> pkgEntry : transitivePackages.entrySet()) {
      PackageValue pkg = (PackageValue) pkgEntry.getValue();
      Preconditions.checkState(
          !pkg.getPackage().containsErrors(),
          "package %s was found to both have and not have errors.",
          pkgEntry);
      packageMapBuilder.put(pkg.getPackage().getPackageIdentifier(), pkg.getPackage());
    }
    ImmutableMap<PackageIdentifier, Package> packageMap = packageMapBuilder.build();
    ImmutableMap.Builder<Label, Target> validTargetsMapBuilder = ImmutableMap.builder();
    for (Label label : validTargets.build()) {
      try {
        Target target = packageMap.get(label.getPackageIdentifier()).getTarget(label.getName());
        validTargetsMapBuilder.put(label, target);
      } catch (NoSuchTargetException e) {
        throw new IllegalStateException(e);
      }
    }
    return Pair.of(packageMap, validTargetsMapBuilder.build());
  }

  @Nullable
  private ByteString executeQuery(
      RuleContext ruleContext, QueryOptions queryOptions, Collection<Label> scope, String query)
      throws InterruptedException {
    SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    Pair<ImmutableMap<PackageIdentifier, Package>, ImmutableMap<Label, Target>> closureInfo;
    try {
      closureInfo = constructPackageMap(env, scope);
      if (closureInfo == null) {
        return null;
      }
    } catch (BrokenQueryScopeException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }

    ImmutableMap<PackageIdentifier, Package> packageMap = closureInfo.first;
    ImmutableMap<Label, Target> validTargetsMap = closureInfo.second;
    PreloadedMapPackageProvider packageProvider =
        new PreloadedMapPackageProvider(packageMap, validTargetsMap);
    TargetPatternPreloader preloader = new SkyframeEnvTargetPatternEvaluator(env);
    Predicate<Label> labelFilter = Predicates.in(validTargetsMap.keySet());

    return doQuery(queryOptions, packageProvider, labelFilter, preloader, query, ruleContext);
  }

  @SuppressWarnings("unchecked")
  @Nullable
  private ByteString doQuery(
      QueryOptions queryOptions,
      PreloadedMapPackageProvider packageProvider,
      Predicate<Label> labelFilter,
      TargetPatternPreloader preloader,
      String query,
      RuleContext ruleContext)
      throws InterruptedException {

    DigraphQueryEvalResult<Target> queryResult;
    OutputFormatter formatter;
    AggregateAllOutputFormatterCallback<Target, ?> targets;
    try {
      Set<Setting> settings = queryOptions.toSettings();

      // Turns out, if we have two targets with a cycle of length 2 were one of
      // the edges is of type NODEP_LABEL type, the targets both show up in
      // each other's result for deps(X) when the query is executed using
      // 'blaze query'. This obviously does not fly when doing the query as a
      // part of the build, thus, there is a slight discrepancy between the
      // behavior of the query engine in these two use cases.
      settings.add(Setting.NO_NODEP_DEPS);

      formatter =
          OutputFormatter.getFormatter(
              OutputFormatter.getDefaultFormatters(), queryOptions.outputFormat);
      if (formatter == null) {
        ruleContext.ruleError(String.format(
            "Invalid output format '%s'. Valid values are: %s",
            queryOptions.outputFormat,
            OutputFormatter.formatterNames(OutputFormatter.getDefaultFormatters())));
        return null;
      }
      // All the packages are already loaded at this point, so there is no need
      // to start up many threads. 4 are started up to make good use of multiple
      // cores.
      BlazeQueryEnvironment queryEnvironment =
          (BlazeQueryEnvironment)
              QUERY_ENVIRONMENT_FACTORY.create(
                  /*transitivePackageLoader=*/ null,
                  /* graphFactory= */ null,
                  packageProvider,
                  packageProvider,
                  preloader,
                  PathFragment.EMPTY_FRAGMENT,
                  /*keepGoing=*/ false,
                  ruleContext.attributes().get("strict", Type.BOOLEAN),
                  /*orderedResults=*/ !QueryOutputUtils.shouldStreamResults(
                      queryOptions, formatter),
                  /*universeScope=*/ ImmutableList.<String>of(),
                  /*loadingPhaseThreads=*/ 4,
                  labelFilter,
                  getEventHandler(ruleContext),
                  settings,
                  ImmutableList.<QueryFunction>of(),
                  /*packagePath=*/ null,
                  /*blockUniverseEvaluationErrors=*/ false);
      QueryExpression expr = QueryExpression.parse(query, queryEnvironment);
      formatter.verifyCompatible(queryEnvironment, expr);
      targets = QueryUtil.newOrderedAggregateAllOutputFormatterCallback(queryEnvironment);
      queryResult = queryEnvironment.evaluateQuery(expr, targets);
    } catch (SkyframeRestartQueryException e) {
      // Do not emit errors for skyframe restarts. They make output of the ConfiguredTargetFunction
      // inconsistent from run to run, and make detecting legitimate errors more difficult.
      return null;
    } catch (QueryException e) {
      ruleContext.ruleError("query failed: " + e.getMessage());
      return null;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    ByteString.Output outputStream = ByteString.newOutput();
    try {
      QueryOutputUtils
          .output(queryOptions, queryResult, targets.getResult(), formatter, outputStream,
          queryOptions.aspectDeps.createResolver(packageProvider, getEventHandler(ruleContext)));
    } catch (ClosedByInterruptException e) {
      throw new InterruptedException(e.getMessage());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return outputStream.toByteString();
  }

  @Immutable // assuming no other reference to result
  private static final class QueryResultAction extends AbstractFileWriteAction {
    private final ByteString result;

    private QueryResultAction(ActionOwner owner, Artifact output, ByteString result) {
      super(owner, ImmutableList.<Artifact>of(), output, /*makeExecutable=*/false);
      this.result = result;
    }

    @Override
    public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
      return new ByteStringDeterministicWriter(result);
    }

    @Override
    protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
      fp.addBytes(result.toByteArray());
    }
  }

  /**
   * Provide target pattern evaluation to the query operations using Skyframe dep lookup. For thread
   * safety, we must synchronize access to the SkyFunction.Environment.
   */
  private static final class SkyframeEnvTargetPatternEvaluator implements TargetPatternPreloader {
    private final SkyFunction.Environment env;

    public SkyframeEnvTargetPatternEvaluator(SkyFunction.Environment env) {
      this.env = env;
    }

    private static Target getExistingTarget(Label label,
        Map<PackageIdentifier, Package> packages) {
      try {
        return packages.get(label.getPackageIdentifier()).getTarget(label.getName());
      } catch (NoSuchTargetException e) {
        // Unexpected since the label was part of the TargetPatternValue.
        throw new IllegalStateException(e);
      }
    }

    @Override
    public Map<String, ResolvedTargets<Target>> preloadTargetPatterns(
        ExtendedEventHandler eventHandler,
        PathFragment relativeWorkingDirectory,
        Collection<String> patterns,
        boolean keepGoing)
            throws TargetParsingException, InterruptedException {
      Preconditions.checkArgument(!keepGoing);
      Preconditions.checkArgument(relativeWorkingDirectory.isEmpty());
      boolean ok = true;
      Map<String, ResolvedTargets<Target>> preloadedPatterns =
          Maps.newHashMapWithExpectedSize(patterns.size());
      Map<TargetPatternKey, String> patternKeys = Maps.newHashMapWithExpectedSize(patterns.size());
      for (String pattern : patterns) {
        checkValidPatternType(pattern);
        patternKeys.put(TargetPatternValue.key(pattern, FilteringPolicies.NO_FILTER, ""), pattern);
      }
      Set<SkyKey> packageKeys = new HashSet<>();
      Map<String, ResolvedTargets<Label>> resolvedLabelsMap =
          Maps.newHashMapWithExpectedSize(patterns.size());
      synchronized (this) {
        for (Map.Entry<SkyKey, ValueOrException<TargetParsingException>> entry :
          env.getValuesOrThrow(patternKeys.keySet(), TargetParsingException.class).entrySet()) {
          TargetPatternValue patternValue = (TargetPatternValue) entry.getValue().get();
          if (patternValue == null) {
            ok = false;
          } else {
            ResolvedTargets<Label> resolvedLabels = patternValue.getTargets();
            resolvedLabelsMap.put(patternKeys.get(entry.getKey()), resolvedLabels);
            for (Label label
                : Iterables.concat(resolvedLabels.getTargets(),
                    resolvedLabels.getFilteredTargets())) {
              packageKeys.add(PackageValue.key(label.getPackageIdentifier()));
            }
          }
        }
      }
      if (!ok) {
        throw new SkyframeRestartQueryException();
      }
      Map<PackageIdentifier, Package> packages =
          Maps.newHashMapWithExpectedSize(packageKeys.size());
      synchronized (this) {
        for (Map.Entry<SkyKey, ValueOrException<NoSuchPackageException>> entry :
          env.getValuesOrThrow(packageKeys, NoSuchPackageException.class).entrySet()) {
          PackageIdentifier pkgName = (PackageIdentifier) entry.getKey().argument();
          Package pkg;
          try {
            PackageValue packageValue = (PackageValue) entry.getValue().get();
            if (packageValue == null) {
              ok = false;
              continue;
            }
            pkg = packageValue.getPackage();
          } catch (NoSuchPackageException nspe) {
            continue;
          }
          Preconditions.checkNotNull(pkg, pkgName);
          packages.put(pkgName, pkg);
        }
      }
      if (!ok) {
        throw new SkyframeRestartQueryException();
      }
      for (Map.Entry<String, ResolvedTargets<Label>> entry : resolvedLabelsMap.entrySet()) {
        String pattern = entry.getKey();
        ResolvedTargets<Label> resolvedLabels = resolvedLabelsMap.get(pattern);
        ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
        for (Label label : resolvedLabels.getTargets()) {
          builder.add(getExistingTarget(label, packages));
        }
        for (Label label : resolvedLabels.getFilteredTargets()) {
          builder.remove(getExistingTarget(label, packages));
        }
        preloadedPatterns.put(pattern, builder.build());
      }
      return preloadedPatterns;
    }

    private void checkValidPatternType(String pattern) throws TargetParsingException {
      TargetPattern.Type type = new TargetPattern.Parser("").parse(pattern).getType();
      if (type == TargetPattern.Type.PATH_AS_TARGET) {
        throw new TargetParsingException(
            String.format("couldn't determine target from filename '%s'", pattern));
      } else if (type == TargetPattern.Type.TARGETS_BELOW_DIRECTORY) {
        throw new TargetParsingException(
            String.format("recursive target patterns are not permitted: '%s'", pattern));
      }
    }
  }

  /**
   * Provide packages and targets to the query operations using precomputed transitive closure.
   */
  private static final class PreloadedMapPackageProvider
      implements PackageProvider, CachingPackageLocator {

    private final ImmutableMap<PackageIdentifier, Package> pkgMap;
    private final ImmutableMap<Label, Target> labelToTarget;

    public PreloadedMapPackageProvider(ImmutableMap<PackageIdentifier, Package> pkgMap,
        ImmutableMap<Label, Target> labelToTarget) {
      this.pkgMap = pkgMap;
      this.labelToTarget = labelToTarget;
    }

    @Override
    public Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageId)
        throws NoSuchPackageException {
      Package pkg = pkgMap.get(packageId);
      if (pkg != null) {
        return pkg;
      }
      // Prefer to throw a checked exception on error; malformed genquery should not crash.
      throw new NoSuchPackageException(packageId, "is not within the scope of the query");
    }

    @Override
    public Target getTarget(ExtendedEventHandler eventHandler, Label label)
        throws NoSuchPackageException, NoSuchTargetException {
      // Try to perform only one map lookup in the common case.
      Target target = labelToTarget.get(label);
      if (target != null) {
        return target;
      }
      // Prefer to throw a checked exception on error; malformed genquery should not crash.
      getPackage(eventHandler, label.getPackageIdentifier());  // maybe throw NoSuchPackageException
      throw new NoSuchTargetException(label, "is not within the scope of the query");
    }

    @Override
    public boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Path getBuildFileForPackage(PackageIdentifier packageId) {
      Package pkg = pkgMap.get(packageId);
      if (pkg == null) {
        return null;
      }
      return pkg.getBuildFile().getPath();
    }
  }

  private static class BrokenQueryScopeException extends Exception {
    public BrokenQueryScopeException(String message) {
      super(message);
    }
  }
}
