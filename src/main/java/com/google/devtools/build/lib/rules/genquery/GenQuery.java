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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.BlazeQueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.SkyframeRestartQueryException;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOutputUtils;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;

import javax.annotation.Nullable;

/**
 * An implementation of the 'genquery' rule.
 */
public class GenQuery implements RuleConfiguredTargetFactory {
  public static final Precomputed<ImmutableList<OutputFormatter>> QUERY_OUTPUT_FORMATTERS =
      new Precomputed<>(new SkyKey(SkyFunctions.PRECOMPUTED, "query_output_formatters"));

  @Override
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
          public DeterministicWriter newDeterministicWriter(EventHandler eventHandler,
                                                            Executor executor) {
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
            new Runfiles.Builder().addTransitiveArtifacts(filesToBuild).build()))
        .build();
  }

  // The transitive closure of these targets is an upper estimate on the labels
  // the query will touch
  private Set<Target> getScope(RuleContext context) {
    List<Label> scopeLabels = context.attributes().get("scope", Type.LABEL_LIST);
    Set<Target> scope = Sets.newHashSetWithExpectedSize(scopeLabels.size());
    for (Label scopePart : scopeLabels) {
      try {
        SkyFunction.Environment env = context.getAnalysisEnvironment().getSkyframeEnv();
        PackageValue packageNode = Preconditions.checkNotNull(
            (PackageValue) env.getValue(PackageValue.key(scopePart.getPackageFragment())));

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
      Preconditions.checkState(transNode != null, "%s not preloaded", key);
      validTargets.addTransitive(transNode.getTransitiveTargets());
      packageNames.addTransitive(transNode.getTransitiveSuccessfulPackages());
    }

    ImmutableMap.Builder<PackageIdentifier, Package> packageMapBuilder = ImmutableMap.builder();
    for (PackageIdentifier pkgId : packageNames.build()) {
      PackageValue pkg = (PackageValue) env.getValue(PackageValue.key(pkgId));
      Preconditions.checkState(pkg != null, "package %s not preloaded", pkgId);
      packageMapBuilder.put(pkg.getPackage().getPackageIdentifier(), pkg.getPackage());
    }
    return Pair.of(packageMapBuilder.build(), validTargets.build().toSet());
  }

  @Nullable
  private byte[] executeQuery(RuleContext ruleContext, QueryOptions queryOptions,
      Set<Target> scope, String query) throws InterruptedException {
    RecursivePackageProvider packageProvider;
    Predicate<Label> labelFilter;
    TargetPatternEvaluator evaluator;

    SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    Pair<ImmutableMap<PackageIdentifier, Package>, Set<Label>> closureInfo =
        constructPackageMap(env, scope);
    ImmutableMap<PackageIdentifier, Package> packageMap = closureInfo.first;
    Set<Label> validTargets = closureInfo.second;
    packageProvider = new PreloadedMapPackageProvider(packageMap, validTargets);
    evaluator = new SkyframeEnvTargetPatternEvaluator(env);
    labelFilter = Predicates.in(validTargets);

    return doQuery(queryOptions, packageProvider, labelFilter, evaluator, query, ruleContext);
  }

  private byte[] doQuery(QueryOptions queryOptions, RecursivePackageProvider packageProvider,
                         Predicate<Label> labelFilter, TargetPatternEvaluator evaluator,
                         String query, RuleContext ruleContext)
      throws InterruptedException {

    BlazeQueryEvalResult<Target> queryResult;
    OutputFormatter formatter;
    try {
      Set<Setting> settings = queryOptions.toSettings();

      // Turns out, if we have two targets with a cycle of length 2 were one of
      // the edges is of type NODEP_LABEL type, the targets both show up in
      // each other's result for deps(X) when the query is executed using
      // 'blaze query'. This obviously does not fly when doing the query as a
      // part of the build, thus, there is a slight discrepancy between the
      // behavior of the query engine in these two use cases.
      settings.add(Setting.NO_NODEP_DEPS);

      // All the packages are already loaded at this point, so there is no need
      // to start up many threads. 4 is started up to make good use of multiple
      // cores.
      ImmutableList<OutputFormatter> outputFormatters = QUERY_OUTPUT_FORMATTERS.get(
          ruleContext.getAnalysisEnvironment().getSkyframeEnv());
      // This is a precomputed value so it should have been injected by the rules module by the
      // time we get there.
      formatter =  OutputFormatter.getFormatter(
          Preconditions.checkNotNull(outputFormatters), queryOptions.outputFormat);
      queryResult = (BlazeQueryEvalResult<Target>) AbstractBlazeQueryEnvironment
          .newQueryEnvironment(
          /*transitivePackageLoader=*/null, /*graph=*/null, packageProvider,
              evaluator,
          /* keepGoing = */ false,
              ruleContext.attributes().get("strict", Type.BOOLEAN),
          /*orderedResults=*/QueryOutputUtils.orderResults(queryOptions, formatter),
              /*universeScope=*/ImmutableList.<String>of(), 4,
              labelFilter,
              getEventHandler(ruleContext),
              settings,
              ImmutableList.<QueryFunction>of()).evaluateQuery(query);
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
      QueryOutputUtils.output(queryOptions, queryResult, formatter, printStream);
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

    @Override
    public Map<String, ResolvedTargets<Target>> preloadTargetPatterns(EventHandler eventHandler,
                                                               Collection<String> patterns,
                                                               boolean keepGoing)
        throws TargetParsingException {
      Preconditions.checkArgument(!keepGoing);
      boolean ok = true;
      Map<String, ResolvedTargets<Target>> preloadedPatterns =
          Maps.newHashMapWithExpectedSize(patterns.size());
      Map<SkyKey, String> keys = Maps.newHashMapWithExpectedSize(patterns.size());
      for (String pattern : patterns) {
        checkValidPatternType(pattern);
        keys.put(TargetPatternValue.key(pattern, FilteringPolicies.NO_FILTER, ""), pattern);
      }
      synchronized (this) {
        for (Map.Entry<SkyKey, ValueOrException<TargetParsingException>> entry :
          env.getValuesOrThrow(keys.keySet(), TargetParsingException.class).entrySet()) {
          TargetPatternValue patternValue = (TargetPatternValue) entry.getValue().get();
          if (patternValue == null) {
            ok = false;
          } else {
            preloadedPatterns.put(keys.get(entry.getKey()), patternValue.getTargets());
          }
        }
      }
      if (!ok) {
        throw new SkyframeRestartQueryException();
      }
      return preloadedPatterns;
    }

    private void checkValidPatternType(String pattern) throws TargetParsingException {
      TargetPattern.Type type = new TargetPattern.Parser("").parse(pattern).getType();
      if (type == TargetPattern.Type.PATH_AS_TARGET) {
        throw new TargetParsingException(
            String.format("couldn't determine target from filename '%s'", pattern));
      } else if (type == TargetPattern.Type.TARGETS_BELOW_PACKAGE) {
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
  private static final class PreloadedMapPackageProvider implements RecursivePackageProvider {

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
        throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
      Preconditions.checkState(targets.contains(label), label);
      Package pkg = Preconditions.checkNotNull(pkgMap.get(label.getPackageIdentifier()), label);
      Target target = Preconditions.checkNotNull(pkg.getTarget(label.getName()), label);
      return target;
    }

    @Override
    public boolean isPackage(String packageName) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void visitPackageNamesRecursively(EventHandler eventHandler,
                                             PathFragment directory,
                                             boolean useTopLevelExcludes,
                                             ThreadPoolExecutor visitorPool,
                                             PathPackageLocator.AcceptsPathFragment observer)
        throws InterruptedException {
      throw new UnsupportedOperationException();
    }
  }
}
