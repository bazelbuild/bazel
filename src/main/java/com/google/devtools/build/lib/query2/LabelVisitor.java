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

package com.google.devtools.build.lib.query2;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetEdgeObserver;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * <p>Visit the transitive closure of a label. Primarily used to "fault in"
 * packages to the packageProvider and ensure the necessary targets exists, in
 * advance of the configuration step, which is intolerant of missing
 * packages/targets.
 *
 * <p>LabelVisitor loads packages concurrently where possible, to increase I/O
 * parallelism.  However, the public interface is not thread-safe: calls to
 * public methods should not be made concurrently.
 *
 * <p>LabelVisitor is stateful: It remembers the previous visitation and can
 * check its validity on subsequent calls to sync() instead of doing the normal
 * visitation.
 *
 * <p>TODO(bazel-team): (2009) a small further optimization could be achieved if we
 * create tasks at the package (not individual label) level, since package
 * loading is the expensive step.  This would require additional bookkeeping to
 * maintain the list of labels that we need to visit once a package becomes
 * available.  Profiling suggests that there is still a potential benefit to be
 * gained: when the set of packages is known a-priori, loading a set of packages
 * that took 20 seconds can be done under 5 in the sequential case or 7 in the
 * current (parallel) case.
 *
 * <h4>Concurrency</h4>
 *
 * <p>The sync() methods of this class is thread-compatible. The accessor
 * ({@link #hasVisited} and similar must not be called until the concurrent phase
 * is over, i.e. all external calls to visit() methods have completed.
 */
final class LabelVisitor {

  /**
   * Attributes of a visitation which determine whether it is up-to-date or not.
   */
  private class VisitationAttributes {
    private Collection<Target> targetsToVisit;
    private boolean success = false;
    private int maxDepth = 0;

    /**
     * Returns true if and only if this visitation attribute is still up-to-date.
     */
    boolean current() {
      return targetsToVisit.equals(lastVisitation.targetsToVisit)
          && maxDepth <= lastVisitation.maxDepth;
    }
  }

  /*
   * Interrupts during the loading phase ===================================
   *
   * Bazel can be interrupted in the middle of the loading phase. The mechanics
   * of this are far from trivial, so there is an explanation of how they are
   * supposed to work. For a description how the same thing works in the
   * execution phase, see ParallelBuilder.java .
   *
   * The sequence of events that happen when the user presses Ctrl-C is the
   * following:
   *
   * 1. A SIGINT gets delivered to the Bazel client process.
   *
   * 2. The client process delivers the SIGINT to the server process.
   *
   * 3. The interruption state of the main thread is set to true.
   *
   * 4. Sooner or later, this results in an InterruptedException being thrown.
   * Usually this takes place because the main thread is interrupted during
   * AbstractQueueVisitor.awaitTermination(). The only exception to this is when
   * the interruption occurs during the loading of a package of a label
   * specified on the command line; in this case, the InterruptedException is
   * thrown during the loading of an individual package (see below where this
   * can occur)
   *
   * 5. The main thread calls ThreadPoolExecutor.shutdown(), which in turn
   * interrupts every worker thread. Then the main thread waits for their
   * termination.
   *
   * 6. An InterruptedException is thrown during the loading of an individual
   * package in the worker threads.
   *
   * 7. All worker threads terminate.
   *
   * 8. An InterruptedException is thrown from
   * AbstractQueueVisitor.awaitTermination()
   *
   * 9. This exception causes the execution of the currently running command to
   * terminate prematurely.
   *
   * The interruption of the loading of an individual package happens as follow:
   *
   * 1. We periodically check the interruption state of the thread in
   * UnixGlob.reallyGlob(). If it is interrupted, an InterruptedException is
   * thrown.
   *
   * 2. The stack is unwound until we are out of the part of the call stack
   * responsible for package loading. This either means that the worker thread
   * terminates or that the label parsing terminates if the package that is
   * being loaded was specified on the command line.
   */
  private final TargetProvider targetProvider;
  private final DependencyFilter edgeFilter;
  private final ConcurrentMap<Label, Integer> visitedTargets = new ConcurrentHashMap<>();

  private VisitationAttributes lastVisitation;

  /**
   * Constant for limiting the permitted depth of recursion.
   */
  private static final int RECURSION_LIMIT = 100;

  /**
   * Construct a LabelVisitor.
   *
   * @param targetProvider how to resolve labels to targets
   * @param edgeFilter which edges may be traversed
   */
  public LabelVisitor(TargetProvider targetProvider, DependencyFilter edgeFilter) {
    this.targetProvider = targetProvider;
    this.lastVisitation = new VisitationAttributes();
    this.edgeFilter = edgeFilter;
  }

  boolean syncWithVisitor(
      ExtendedEventHandler eventHandler,
      Collection<Target> targetsToVisit,
      boolean keepGoing,
      int parallelThreads,
      int maxDepth,
      TargetEdgeObserver... observers)
      throws InterruptedException {
    VisitationAttributes nextVisitation = new VisitationAttributes();
    nextVisitation.targetsToVisit = targetsToVisit;
    nextVisitation.maxDepth = maxDepth;

    if (!lastVisitation.success || !nextVisitation.current()) {
      try {
        nextVisitation.success = redoVisitation(eventHandler, nextVisitation, keepGoing,
            parallelThreads, maxDepth, observers);
        return nextVisitation.success;
      } finally {
        lastVisitation = nextVisitation;
      }
    } else {
      return true;
    }
  }

  // Does a bounded transitive visitation starting at the given top-level targets.
  private boolean redoVisitation(
      ExtendedEventHandler eventHandler,
      VisitationAttributes visitation,
      boolean keepGoing,
      int parallelThreads,
      int maxDepth,
      TargetEdgeObserver... observers)
      throws InterruptedException {
    visitedTargets.clear();

    Visitor visitor = new Visitor(eventHandler, keepGoing, parallelThreads, maxDepth, observers);

    Throwable uncaught = null;
    boolean result;
    try {
      visitor.visitTargets(visitation.targetsToVisit);
    } catch (Throwable t) {
      visitor.stopNewActions();
      uncaught = t;
    } finally {
      // Run finish() in finally block to ensure we don't leak threads on exceptions.
      result = visitor.finish();
    }
    Throwables.propagateIfPossible(uncaught);
    return result;
  }

  boolean hasVisited(Label target) {
    return visitedTargets.containsKey(target);
  }

  @VisibleForTesting class Visitor extends AbstractQueueVisitor {

    private final static String THREAD_NAME = "LabelVisitor";

    private final ExtendedEventHandler eventHandler;
    private final boolean keepGoing;
    private final int maxDepth;
    private final Iterable<TargetEdgeObserver> observers;
    private final TargetEdgeErrorObserver errorObserver;
    private final AtomicBoolean stopNewActions = new AtomicBoolean(false);

    public Visitor(
        ExtendedEventHandler eventHandler,
        boolean keepGoing,
        int parallelThreads,
        int maxDepth,
        TargetEdgeObserver... observers) {
      // Observing the loading phase of a typical large package (with all subpackages) shows
      // maximum thread-level concurrency of ~20. Limiting the total number of threads to 200 is
      // therefore conservative and should help us avoid hitting native limits.
      super(
          parallelThreads,
          1L,
          TimeUnit.SECONDS,
          !keepGoing,
          THREAD_NAME,
          ErrorClassifier.DEFAULT);
      this.eventHandler = eventHandler;
      this.maxDepth = maxDepth;
      this.errorObserver = new TargetEdgeErrorObserver();
      ImmutableList.Builder<TargetEdgeObserver> builder = ImmutableList.builder();
      builder.add(observers);
      builder.add(errorObserver);
      this.observers = builder.build();
      this.keepGoing = keepGoing;
    }

    /**
     * Visit the specified labels and follow the transitive closure of their outbound dependencies.
     *
     * @param targets the targets to visit
     */
    @ThreadSafe
    public void visitTargets(Iterable<Target> targets) throws InterruptedException {
      for (Target target : targets) {
        visit(null, null, target, 0, 0);
      }
    }

    @ThreadSafe
    public boolean finish() throws InterruptedException {
      awaitQuiescence(/*interruptWorkers=*/ true);
      return !errorObserver.hasErrors();
    }

    @Override
    protected boolean blockNewActions() {
      return (!keepGoing && errorObserver.hasErrors()) || super.blockNewActions() ||
          stopNewActions.get();
    }

    public void stopNewActions() {
      stopNewActions.set(true);
    }

    private void enqueueTarget(
        final Target from, final Attribute attr, final Label label, final int depth,
        final int count) {
      // Don't perform the targetProvider lookup if at the maximum depth already.
      if (depth >= maxDepth) {
        return;
      }

      // Avoid thread-related overhead when not crossing packages.
      // Can start a new thread when count reaches 100, to prevent infinite recursion.
      if (from != null && from.getLabel().getPackageFragment().equals(label.getPackageFragment())
          && !blockNewActions() && count < RECURSION_LIMIT) {
        newVisitRunnable(from, attr, label, depth, count + 1).run();
      } else {
        execute(newVisitRunnable(from, attr, label, depth, 0));
      }
    }

    private Runnable newVisitRunnable(final Target from, final Attribute attr, final Label label,
        final int depth, final int count) {
      return () -> {
        try {
          try {
            visit(from, attr, targetProvider.getTarget(eventHandler, label), depth + 1, count);
          } catch (NoSuchThingException e) {
            observeError(from, label, e);
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      };
    }

    private void visitTargetVisibility(Target target, int depth, int count) {
      Attribute attribute = null;
      if (target instanceof Rule) {
        Rule rule = (Rule) target;
        RuleClass ruleClass = rule.getRuleClassObject();
        if (!ruleClass.hasAttr("visibility", BuildType.NODEP_LABEL_LIST)) {
          return;
        }
        attribute = ruleClass.getAttributeByName("visibility");
        if (!edgeFilter.apply(rule, attribute)) {
          return;
        }
      }

      for (Label label : target.getVisibility().getDependencyLabels()) {
        enqueueTarget(target, attribute, label, depth, count);
      }
    }

    /**
     * Visit all the labels in a given rule.
     *
     * <p>Called in a worker thread if CONCURRENT.
     *
     * @param rule the rule to visit
     */
    @ThreadSafe
    private void visitRule(final Rule rule, final int depth, final int count)
        throws InterruptedException {
      // Follow all labels defined by this rule:
      AggregatingAttributeMapper.of(rule)
          .visitLabels()
          .stream()
          .filter(depEdge -> edgeFilter.apply(rule, depEdge.getAttribute()))
          .forEach(
              depEdge ->
                  enqueueTarget(rule, depEdge.getAttribute(), depEdge.getLabel(), depth, count));
    }

    @ThreadSafe
    private void visitPackageGroup(PackageGroup packageGroup, int depth, int count) {
      for (final Label include : packageGroup.getIncludes()) {
        enqueueTarget(packageGroup, null, include, depth, count);
      }
    }

    /**
     * Visits the target and its package.
     *
     * <p>Potentially blocking invocations into the package cache are enqueued in the worker pool if
     * CONCURRENT.
     */
    private void visit(Target from, Attribute attribute, final Target target, int depth, int count)
        throws InterruptedException {
      if (target == null) {
        throw new NullPointerException(
            String.format("'%s' attribute '%s'",
              from == null ? "(null)" : from.getLabel().toString(),
              attribute == null ? "(null)" : attribute.getName()));
      }
      if (depth > maxDepth) {
        return;
      }

      if (from != null) {
        observeEdge(from, attribute, target);
        visitAspectsIfRequired(from, attribute, target, depth, count);
      }
      visitTargetNode(target, depth, count);
    }

    private void visitAspectsIfRequired(
        Target from, Attribute attribute, final Target to, int depth, int count) {
      // TODO(bazel-team): The getAspects call below is duplicate work for each direct dep entailed
      // by an attribute's value. Consider doing a slight refactor where we make this method call
      // exactly once per Attribute.
      ImmutableList<Aspect> aspectsOfAttribute =
          (from instanceof Rule) ? attribute.getAspects((Rule) from) : ImmutableList.of();
      ImmutableMultimap<Attribute, Label> labelsFromAspects =
          AspectDefinition.visitAspectsIfRequired(from, aspectsOfAttribute, to, edgeFilter);
      // Create an edge from target to the attribute value.
      for (Map.Entry<Attribute, Label> entry : labelsFromAspects.entries()) {
        enqueueTarget(from, entry.getKey(), entry.getValue(), depth, count);
      }
    }

    /**
     * Visit the specified target. Called in a worker thread if CONCURRENT.
     *
     * @param target the target to visit
     */
    private void visitTargetNode(Target target, int depth, int count) throws InterruptedException {
      Integer minTargetDepth = visitedTargets.putIfAbsent(target.getLabel(), depth);
      if (minTargetDepth != null) {
        // The target was already visited at a greater depth.
        // The closure we are about to build is therefore a subset of what
        // has already been built, and we can skip it.
        // Also special case MAX_VALUE, where we never want to revisit targets.
        // (This avoids loading phase overhead outside of queries).
        if (maxDepth == Integer.MAX_VALUE || minTargetDepth <= depth) {
          return;
        }
        // Check again in case it was overwritten by another thread.
        synchronized (visitedTargets) {
          if (visitedTargets.get(target.getLabel()) <= depth) {
            return;
          }
          visitedTargets.put(target.getLabel(), depth);
        }
      }

      observeNode(target);
      if (target instanceof OutputFile) {
        Rule rule = ((OutputFile) target).getGeneratingRule();
        observeEdge(target, null, rule);
        // This is the only recursive call to visit which doesn't pass through enqueueTarget().
        visit(null, null, rule, depth + 1, count + 1);
        visitTargetVisibility(target, depth, count);
      } else if (target instanceof InputFile) {
        visitTargetVisibility(target, depth, count);
      } else if (target instanceof Rule) {
        visitTargetVisibility(target, depth, count);
        visitRule((Rule) target, depth, count);
      } else if (target instanceof PackageGroup) {
        visitPackageGroup((PackageGroup) target, depth, count);
      }
    }

    private void observeEdge(Target from, Attribute attribute, Target to) {
      for (TargetEdgeObserver observer : observers) {
        observer.edge(from, attribute, to);
      }
    }

    private void observeNode(Target target) {
      for (TargetEdgeObserver observer : observers) {
        observer.node(target);
      }
    }

    private void observeError(Target from, Label label, NoSuchThingException e)
        throws InterruptedException {
      for (TargetEdgeObserver observer : observers) {
        observer.missingEdge(from, label, e);
      }
    }
  }
}
