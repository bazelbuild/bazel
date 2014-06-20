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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.concurrent.ExecutorShutdownUtil;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * A target pattern parser that supports wildcards and negation.
 *
 * WARNING: This parser is only to be used to parse command line arguments for
 *          build tools. Don't even think about making this part of internal
 *          logic that interprets BUILD files.
 *
 */
public final class CommandLineTargetPatternEvaluator implements TargetPatternEvaluator {

  private final RecursivePackageProvider packageProvider;
  private TargetPattern.Parser parser;

  private CommandLineTargetPatternEvaluator(RecursivePackageProvider packageProvider,
      PathFragment offset) {
    Preconditions.checkArgument(!offset.isAbsolute());
    this.packageProvider = packageProvider;
    parser = new TargetPattern.Parser(offset.getPathString());
  }

  /**
   * Construct a label parser with the given package providers. The parser will assume that every
   * label is relative to the package root.
   */
  public static CommandLineTargetPatternEvaluator create(RecursivePackageProvider packageProvider) {
    return new CommandLineTargetPatternEvaluator(packageProvider, new PathFragment(""));
  }

  @Override
  public void updateOffset(PathFragment relativeWorkingDirectory) {
    parser = new TargetPattern.Parser(relativeWorkingDirectory.getPathString());
  }

  @Override
  public String getOffset() {
    return parser.getRelativeDirectory();
  }

  @Override
  public ResolvedTargets<Target> parseTargetPatternList(final ErrorEventListener listener,
      List<String> targetPatterns, final FilteringPolicy policy, final boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    List<ResolvedTargets<Target>> patternResults = internalParseTargetPatternList(listener,
        targetPatterns, policy, keepGoing, /*allowNegativePatterns=*/true);

    // Now make sure we iterate over the parsing results in command-line order,
    // since the interaction of positive and negative patterns is
    // order-dependent.
    ResolvedTargets.Builder<Target> result = ResolvedTargets.builder();
    for (int i = 0; i < targetPatterns.size(); i++) {
      String pattern = targetPatterns.get(i);
      ResolvedTargets<Target> targets = patternResults.get(i);
      if (pattern.startsWith("-")) {
        result.filter(Predicates.not(Predicates.in(targets.getTargets())));
      } else {
        result.merge(targets);
      }
    }
    return result.build();
  }

  private List<ResolvedTargets<Target>> internalParseTargetPatternList(
      final ErrorEventListener listener, List<String> targetPatterns, final FilteringPolicy policy,
      final boolean keepGoing, boolean allowNegativePatterns)
          throws TargetParsingException, InterruptedException {
    // Use two thread pools: One for each target pattern, and one for recursive subpackage
    // visitation. Using a single pool would be deadlock-inducing, as we'd block on
    // completion from within worker threads themselves.
    final ExecutorService patternResolverPool = Executors.newFixedThreadPool(20,
        new ThreadFactoryBuilder().setNameFormat("pattern-resolver-%d").build());
    final ThreadPoolExecutor packageVisitorPool =
        ExecutorShutdownUtil.newSlackPool(100, "package-visitor");
    final List<Future<ResolvedTargets<Target>>> patternResults =
        Lists.newArrayListWithCapacity(targetPatterns.size());
    for (final String targetPattern : targetPatterns) {
      final boolean negative = allowNegativePatterns && targetPattern.startsWith("-");
      patternResults.add(
          patternResolverPool.submit(new Callable<ResolvedTargets<Target>>() {
            @Override
            public ResolvedTargets<Target> call() throws TargetParsingException {
              if (negative) {
                return parseTargetPattern(listener, targetPattern.substring(1),
                    FilteringPolicies.NO_FILTER, keepGoing, packageVisitorPool);
              } else {
                return parseTargetPattern(listener, targetPattern, policy, keepGoing,
                                          packageVisitorPool);
              }
            }
          }));
    }

    try {
      ExecutorShutdownUtil.throwingInterruptibleShutdown(patternResolverPool);
    } finally {
      // Try to shut down this executor too, even if the shutdown above was interrupted.
      ExecutorShutdownUtil.throwingInterruptibleShutdown(packageVisitorPool);
    }

    Preconditions.checkState(patternResults.size() == targetPatterns.size());

    // The result list contains the result in the order in which the patterns were passed in.
    List<ResolvedTargets<Target>> result = Lists.newArrayListWithCapacity(targetPatterns.size());
    for (Future<ResolvedTargets<Target>> future : patternResults) {
      ResolvedTargets<Target> targets = null;
      try {
        targets = Uninterruptibles.getUninterruptibly(future);
      } catch (ExecutionException e) {
        Throwables.propagateIfInstanceOf(e.getCause(), TargetParsingException.class);
        Throwables.propagateIfPossible(e.getCause());
        throw new RuntimeException(e);
      }
      result.add(targets);
    }
    return result;
  }

  @Override
  public ResolvedTargets<Target> parseTargetPattern(ErrorEventListener listener, String pattern,
      boolean keepGoing) throws TargetParsingException {
    return parseTargetPattern(listener, pattern, FilteringPolicies.NO_FILTER, keepGoing);
  }

  @Override
  public List<ResolvedTargets<Target>> preloadTargetPatterns(
      ErrorEventListener listener, List<String> patterns, boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    return internalParseTargetPatternList(listener, patterns, FilteringPolicies.NO_FILTER,
        keepGoing, /*allowNegativePatterns=*/false);
  }

  /**
   * Attempts to parse a single target pattern while consulting the package
   * cache to check for the existence of packages and directories and the build
   * targets in them.  Implements the specification described in the
   * class-level comment.
   */
  public ResolvedTargets<Target> parseTargetPattern(ErrorEventListener listener, String pattern,
      FilteringPolicy policy, boolean keepGoing) throws TargetParsingException {
    return parseTargetPattern(listener, pattern, policy, keepGoing, null);
  }

  /**
   * Attempts to parse a single target pattern while consulting the package
   * cache to check for the existence of packages and directories and the build
   * targets in them.  Implements the specification described in the
   * class-level comment.
   *
   * Uses the given thread pool to run package visitation under a subtree.
   */
  public ResolvedTargets<Target> parseTargetPattern(ErrorEventListener listener, String pattern,
      FilteringPolicy policy, boolean keepGoing, ThreadPoolExecutor packageVisitorPool)
      throws TargetParsingException {
    TargetPattern targetPattern = parser.parse(pattern);
    try {
      return targetPattern.eval(new PackageCacheBackedTargetPatternResolver(
          packageProvider, listener, keepGoing, policy, packageVisitorPool));
    } catch (TargetParsingException e) {
      if (keepGoing) {
        listener.error(null, e.getMessage());
        return ResolvedTargets.<Target>failed();
      }
      throw e;
    } catch (InterruptedException e) {
      // See bug "Loading phase is still not interruptible enough!": Ideally we would
      // propagate InterruptedException, but this has good bang for the buck; it assumes that
      // TargetParsingException always stops the build.
      throw new TargetParsingException("interrupted");
    } catch (TargetPatternResolver.MissingDepException e) {
      throw new AssertionError("unexpected missing dep", e);
    }
  }

  public static Label getSimpleTargetPattern(String pattern) {
    if (TargetPattern.Parser.isSimpleTargetPattern(pattern)) {
      try {
        return Label.parseAbsolute(pattern);
      } catch (Label.SyntaxException e) {
        return null;
      }
    }
    return null;
  }
}
