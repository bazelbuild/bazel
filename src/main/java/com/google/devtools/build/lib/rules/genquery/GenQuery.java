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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.DigraphQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllCallback;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.query2.output.QueryOutputUtils;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.channels.ClosedByInterruptException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * An implementation of the 'genquery' rule.
 */
public class GenQuery implements RuleConfiguredTargetFactory {
  public static final Precomputed<ImmutableList<OutputFormatter>> QUERY_OUTPUT_FORMATTERS =
      new Precomputed<>(SkyKey.create(SkyFunctions.PRECOMPUTED, "query_output_formatters"));

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    Artifact outputArtifact = ruleContext.createOutputArtifact();

    // The query string
    final String query = ruleContext.attributes().get("expression", Type.STRING);

    OptionsParser optionsParser = OptionsParser.newOptionsParser(QueryOptions.class);
    optionsParser.setAllowResidue(false);
    try {
      optionsParser.parse(ruleContext.attributes().get("opts", Type.STRING_LIST));
    } catch (OptionsParsingException e) {
      ruleContext.attributeError("opts", "error while parsing query options: " + e.getMessage());
      return null;
    }

    // Parsed query options
    QueryOptions queryOptions = optionsParser.getOptions(QueryOptions.class);
    if (queryOptions.keepGoing) {
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

    final byte[] result = executeQuery(ruleContext, queryOptions, getScope(ruleContext), query);
    if (result == null || ruleContext.hasErrors()) {
      return null;
    }

    ruleContext.registerAction(
        new AbstractFileWriteAction(
            ruleContext.getActionOwner(), Collections.<Artifact>emptySet(), outputArtifact, false) {
          @Override
          public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
            return new DeterministicWriter() {
              @Override
              public void writeOutputFile(OutputStream out) throws IOException {
                out.write(result);
              }
            };
          }

          @Override
          protected String computeKey() {
            Fingerprint f = new Fingerprint();
            f.addBytes(result);
            return f.hexDigestAndReset();
          }
        });

    NestedSet<Artifact> filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER, outputArtifact);
    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, RunfilesProvider.simple(
            new Runfiles.Builder(ruleContext.getWorkspaceName())
                .addTransitiveArtifacts(filesToBuild).build()))
        .build();
  }

  // The transitive closure of these targets is an upper estimate on the labels
  // the query will touch
  private Set<Target> getScope(RuleContext context) {
    List<Label> scopeLabels = context.attributes().get("scope", BuildType.LABEL_LIST);
    Set<Target> scope = Sets.newHashSetWithExpectedSize(scopeLabels.size());
    for (Label scopePart : scopeLabels) {
      SkyFunction.Environment env = context.getAnalysisEnvironment().getSkyframeEnv();
      PackageValue packageNode =
          (PackageValue) env.getValue(PackageValue.key(scopePart.getPackageIdentifier()));
      Preconditions.checkNotNull(
          packageNode,
          "Packages in transitive closure of scope '%s'"
              + "were already loaded during the loading phase",
          scopePart);
      try {
        scope.add(packageNode.getPackage().getTarget(scopePart.getName()));
      } catch (NoSuchTargetException e) {
        throw new IllegalStateException(e);
      }
    }
    return scope;
  }


  /**
   * DO NOT USE! We should get rid of this method: errors reported directly to this object don't set
   * the error flag in {@link ConfiguredTarget}.
   */
  private EventHandler getEventHandler(RuleContext ruleContext) {
    return ruleContext.getAnalysisEnvironment().getEventHandler();
  }

  @Nullable
  private Pair<ImmutableMap<PackageIdentifier, Package>, Set<Label>> constructPackageMap(
      SkyFunction.Environment env, Collection<Target> scope) {
    // It is not necessary for correctness to construct intermediate NestedSets; we could iterate
    // over individual targets in scope immediately. However, creating a composite NestedSet first
    // saves us from iterating over the same sub-NestedSets multiple times.
    NestedSetBuilder<Label> validTargets = NestedSetBuilder.stableOrder();
    NestedSetBuilder<PackageIdentifier> packageNames = NestedSetBuilder.stableOrder();
    for (Target target : scope) {
      SkyKey key = TransitiveTargetValue.key(target.getLabel());
      TransitiveTargetValue transNode = (TransitiveTargetValue) env.getValue(key);
      if (transNode == null) {
        return null;
      }
      validTargets.addTransitive(transNode.getTransitiveTargets());
      packageNames.addTransitive(transNode.getTransitiveSuccessfulPackages());
    }

    ImmutableMap.Builder<PackageIdentifier, Package> packageMapBuilder = ImmutableMap.builder();
    for (PackageIdentifier pkgId : packageNames.build()) {
      PackageValue pkg = (PackageValue) env.getValue(PackageValue.key(pkgId));
      Preconditions.checkNotNull(pkg, "package %s not preloaded", pkgId);
      Preconditions.checkState(!pkg.getPackage().containsErrors(), pkgId);
      packageMapBuilder.put(pkg.getPackage().getPackageIdentifier(), pkg.getPackage());
    }
    return Pair.of(packageMapBuilder.build(), validTargets.build().toSet());
  }

  @Nullable
  private byte[] executeQuery(RuleContext ruleContext, QueryOptions queryOptions,
      Set<Target> scope, String query) throws InterruptedException {
    SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    Pair<ImmutableMap<PackageIdentifier, Package>, Set<Label>> closureInfo =
        constructPackageMap(env, scope);
    if (closureInfo == null) {
      return null;
    }
    ImmutableMap<PackageIdentifier, Package> packageMap = closureInfo.first;
    Set<Label> validTargets = closureInfo.second;
    PackageProvider packageProvider = new PreloadedMapPackageProvider(packageMap, validTargets);
    TargetPatternEvaluator evaluator = new SkyframeEnvTargetPatternEvaluator(env);
    Predicate<Label> labelFilter = Predicates.in(validTargets);

    return doQuery(queryOptions, packageProvider, labelFilter, evaluator, query, ruleContext);
  }

  @Nullable
  private byte[] doQuery(QueryOptions queryOptions, PackageProvider packageProvider,
                         Predicate<Label> labelFilter, TargetPatternEvaluator evaluator,
                         String query, RuleContext ruleContext)
      throws InterruptedException {

    DigraphQueryEvalResult<Target> queryResult;
    OutputFormatter formatter;
    AggregateAllCallback<Target> targets = new AggregateAllCallback<>();
    try {
      Set<Setting> settings = queryOptions.toSettings();

      // Turns out, if we have two targets with a cycle of length 2 were one of
      // the edges is of type NODEP_LABEL type, the targets both show up in
      // each other's result for deps(X) when the query is executed using
      // 'blaze query'. This obviously does not fly when doing the query as a
      // part of the build, thus, there is a slight discrepancy between the
      // behavior of the query engine in these two use cases.
      settings.add(Setting.NO_NODEP_DEPS);

      ImmutableList<OutputFormatter> outputFormatters = QUERY_OUTPUT_FORMATTERS.get(
          ruleContext.getAnalysisEnvironment().getSkyframeEnv());
      // This is a precomputed value so it should have been injected by the rules module by the
      // time we get there.
      formatter =  OutputFormatter.getFormatter(
          Preconditions.checkNotNull(outputFormatters), queryOptions.outputFormat);

      // All the packages are already loaded at this point, so there is no need
      // to start up many threads. 4 are started up to make good use of multiple
      // cores.
      queryResult = (DigraphQueryEvalResult<Target>) AbstractBlazeQueryEnvironment
          .newQueryEnvironment(
              /*transitivePackageLoader=*/null, /*graph=*/null, packageProvider,
              evaluator,
              /*keepGoing=*/false,
              ruleContext.attributes().get("strict", Type.BOOLEAN),
              /*orderedResults=*/!QueryOutputUtils.shouldStreamResults(queryOptions, formatter),
              /*universeScope=*/ImmutableList.<String>of(),
              /*loadingPhaseThreads=*/4,
              labelFilter,
              getEventHandler(ruleContext),
              settings,
              ImmutableList.<QueryFunction>of(),
              /*packagePath=*/null).evaluateQuery(query, targets);
    } catch (SkyframeRestartQueryException e) {
      // Do not emit errors for skyframe restarts. They make output of the ConfiguredTargetFunction
      // inconsistent from run to run, and make detecting legitimate errors more difficult.
      return null;
    } catch (QueryException e) {
      ruleContext.ruleError("query failed: " + e.getMessage());
      return null;
    }

    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    PrintStream printStream = new PrintStream(outputStream);

    try {
      QueryOutputUtils
          .output(queryOptions, queryResult, targets.getResult(), formatter, printStream,
          queryOptions.aspectDeps.createResolver(packageProvider, getEventHandler(ruleContext)));
    } catch (ClosedByInterruptException e) {
      throw new InterruptedException(e.getMessage());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    printStream.flush();

    return outputStream.toByteArray();
  }

  /**
   * Provide target pattern evaluation to the query operations using Skyframe dep lookup. For thread
   * safety, we must synchronize access to the SkyFunction.Environment.
   */
  private static final class SkyframeEnvTargetPatternEvaluator implements TargetPatternEvaluator {
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
    public Map<String, ResolvedTargets<Target>> preloadTargetPatterns(EventHandler eventHandler,
                                                               Collection<String> patterns,
                                                               boolean keepGoing)
        throws TargetParsingException {
      Preconditions.checkArgument(!keepGoing);
      boolean ok = true;
      Map<String, ResolvedTargets<Target>> preloadedPatterns =
          Maps.newHashMapWithExpectedSize(patterns.size());
      Map<SkyKey, String> patternKeys = Maps.newHashMapWithExpectedSize(patterns.size());
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
            String.format("recursive target patterns are not permitted: '%s''", pattern));
      }
    }

    @Override
    public ResolvedTargets<Target> parseTargetPatternList(EventHandler eventHandler,
                                                          List<String> targetPatterns,
                                                          FilteringPolicy policy, boolean keepGoing)
        throws TargetParsingException {
      throw new UnsupportedOperationException();
    }

    @Override
    public ResolvedTargets<Target> parseTargetPattern(EventHandler eventHandler, String pattern,
                                                      boolean keepGoing)
        throws TargetParsingException {
      throw new UnsupportedOperationException();
    }

    @Override
    public void updateOffset(PathFragment relativeWorkingDirectory) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String getOffset() {
      throw new UnsupportedOperationException();
    }
  }

  /**
   * Provide packages and targets to the query operations using precomputed transitive closure.
   */
  private static final class PreloadedMapPackageProvider implements PackageProvider {

    private final ImmutableMap<PackageIdentifier, Package> pkgMap;
    private final Set<Label> targets;

    public PreloadedMapPackageProvider(ImmutableMap<PackageIdentifier, Package> pkgMap,
        Set<Label> targets) {
      this.pkgMap = pkgMap;
      this.targets = targets;
    }

    @Override
    public Package getPackage(EventHandler eventHandler, PackageIdentifier packageId)
        throws NoSuchPackageException, InterruptedException {
      return Preconditions.checkNotNull(pkgMap.get(packageId));
    }

    @Override
    public Target getTarget(EventHandler eventHandler, Label label)
        throws NoSuchPackageException, NoSuchTargetException {
      Preconditions.checkState(targets.contains(label), label);
      Package pkg = Preconditions.checkNotNull(pkgMap.get(label.getPackageIdentifier()), label);
      Target target = Preconditions.checkNotNull(pkg.getTarget(label.getName()), label);
      return target;
    }

    @Override
    public boolean isPackage(EventHandler eventHandler, PackageIdentifier packageName) {
      throw new UnsupportedOperationException();
    }
  }
}
