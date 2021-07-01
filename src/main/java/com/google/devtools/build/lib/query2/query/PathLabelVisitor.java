// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.query;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.LabelVisitationUtils;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.CompletionException;
import javax.annotation.Nullable;

/** Computes path queries given a {@link TargetProvider}. */
final class PathLabelVisitor {
  private final TargetProvider targetProvider;
  private final DependencyFilter edgeFilter;
  private final TargetEdgeErrorObserver errorObserver;

  /**
   * Construct a PathLabelVisitor.
   *
   * @param targetProvider how to resolve labels to targets
   * @param edgeFilter which edges may be traversed
   */
  public PathLabelVisitor(
      TargetProvider targetProvider,
      DependencyFilter edgeFilter,
      TargetEdgeErrorObserver errorObserver) {
    this.targetProvider = targetProvider;
    this.edgeFilter = edgeFilter;
    this.errorObserver = errorObserver;
  }

  public Iterable<Target> somePath(
      ExtendedEventHandler eventHandler, Iterable<Target> from, Iterable<Target> to)
      throws InterruptedException {
    Visitor visitor = new Visitor(eventHandler, VisitorMode.SOMEPATH);
    // TODO(ulfjack): It might be faster to stop the visitation once we see any 'to' Target.
    visitor.visitTargets(from);
    for (Target t : to) {
      if (visitor.hasVisited(t)) {
        ArrayDeque<Target> result = new ArrayDeque<>();
        Target at = t;
        // TODO(ulfjack): This can result in an infinite loop if there's a dependency cycle.
        while (true) {
          result.addFirst(at);
          List<Target> pred = visitor.getParents(at);
          if (pred == null) {
            break;
          }
          at = pred.get(0);
        }
        return result;
      }
    }
    return ImmutableList.of();
  }

  public Iterable<Target> allPaths(
      ExtendedEventHandler eventHandler, Iterable<Target> from, Iterable<Target> to)
      throws InterruptedException {
    Visitor visitor = new Visitor(eventHandler, VisitorMode.ALLPATHS);
    visitor.visitTargets(from);
    Set<Target> result = new HashSet<>();
    Queue<Target> workQueue = new ArrayDeque<>();
    // Add all 'to' targets to the work queue that are in the transitive closure of 'from' targets.
    for (Target t : to) {
      if (visitor.hasVisited(t)) {
        workQueue.add(t);
      }
    }
    while (!workQueue.isEmpty()) {
      Target at = workQueue.remove();
      if (result.add(at)) {
        List<Target> pred = visitor.getParents(at);
        if (pred != null) {
          workQueue.addAll(pred);
        }
      }
    }
    return result;
  }

  public Iterable<Target> samePkgDirectRdeps(
      ExtendedEventHandler eventHandler, Iterable<Target> from) throws InterruptedException {
    Visitor visitor = new Visitor(eventHandler, VisitorMode.SAME_PKG_DIRECT_RDEPS);
    for (Target t : from) {
      visitor.visitTargets(t.getPackage().getTargets().values());
    }
    Set<Target> result = new HashSet<>();
    for (Target t : from) {
      List<Target> pred = visitor.getParents(t);
      if (pred != null) {
        result.addAll(pred);
      }
    }
    return result;
  }

  public Iterable<Target> rdeps(
      ExtendedEventHandler eventHandler,
      Iterable<Target> from,
      Iterable<Target> universe,
      int depth)
      throws InterruptedException {
    Visitor visitor = new Visitor(eventHandler, VisitorMode.ALLPATHS);
    visitor.visitTargets(universe);

    Set<Target> result = new HashSet<>();
    Set<Target> at = new HashSet<>();
    // Add all 'from' targets to the work set that are in the transitive closure of 'universe'.
    for (Target t : from) {
      if (visitor.hasVisited(t)) {
        at.add(t);
      }
    }
    Set<Target> next = new HashSet<>();
    // In round i, we add all targets at depth i to result, so we need depth + 1 rounds. Note that
    // depth can be Integer.MAX_VALUE, so do not use "< depth + 1" here..
    for (int i = 0; i <= depth; i++) {
      if (at.isEmpty()) {
        break;
      }
      for (Target t : at) {
        if (result.add(t)) {
          List<Target> pred = visitor.getParents(t);
          if (pred != null) {
            next.addAll(pred);
          }
        }
      }
      at.clear();
      Set<Target> temp = at;
      at = next;
      next = temp;
    }
    return result;
  }

  private enum VisitorMode {
    DEPS,
    ALLPATHS,
    SOMEPATH,
    SAME_PKG_DIRECT_RDEPS
  }

  private static class Visit {
    private final Target from;
    private final Attribute attribute;
    private final Target target;

    private Visit(Target from, Attribute attribute, Target target) {
      if (target == null) {
        throw new NullPointerException(
            String.format(
                "'%s' attribute '%s'",
                from == null ? "(null)" : from.getLabel().toString(),
                attribute == null ? "(null)" : attribute.getName()));
      }
      this.from = from;
      this.attribute = attribute;
      this.target = target;
    }
  }

  private final class Visitor {
    private final ExtendedEventHandler eventHandler;
    private final VisitorMode mode;
    private final Set<Target> visited = new HashSet<>();
    private final Map<Target, List<Target>> parentMap = new HashMap<>();
    private final Queue<Visit> workQueue = new ArrayDeque<>();

    Visitor(ExtendedEventHandler eventHandler, VisitorMode mode) {
      this.eventHandler = eventHandler;
      this.mode = Preconditions.checkNotNull(mode);
    }

    public boolean hasVisited(Target target) {
      return visited.contains(target);
    }

    @Nullable
    public List<Target> getParents(Target target) {
      return parentMap.get(target);
    }

    /**
     * Visit the specified labels and follow the transitive closure of their outbound dependencies.
     *
     * @param targets the targets to visit
     */
    @ThreadSafe
    private void visitTargets(Iterable<Target> targets) throws InterruptedException {
      for (Target t : targets) {
        enqueue(null, null, t);
      }
      while (!workQueue.isEmpty()) {
        Visit visit = workQueue.remove();
        try {
          visit(visit.from, visit.attribute, visit.target);
        } catch (NoSuchThingException e) {
          errorObserver.missingEdge(visit.from, visit.target.getLabel(), e);
        }
      }
    }

    private void enqueue(Target from, Attribute attribute, Label label)
        throws InterruptedException, NoSuchThingException {
      Target target = targetProvider.getTarget(eventHandler, label);
      enqueue(from, attribute, target);
    }

    private void enqueue(Target from, Attribute attribute, Target target) {
      workQueue.add(new Visit(from, attribute, target));
    }

    private void visit(Target from, Attribute attribute, Target target)
        throws InterruptedException, NoSuchThingException {
      if (from != null) {
        switch (mode) {
          case DEPS:
            // Don't update parentMap; only use visited.
            break;
          case SAME_PKG_DIRECT_RDEPS:
            // Only track same-package dependencies.
            if (target
                .getLabel()
                .getPackageIdentifier()
                .equals(from.getLabel().getPackageIdentifier())) {
              if (!parentMap.containsKey(target)) {
                parentMap.put(target, new ArrayList<>());
              }
              parentMap.get(target).add(from);
            }
            // We only need to perform a single level of visitation. We have a non-null 'from'
            // target, and we're now at 'target' target, so we have one level, and can return here.
            return;
          case ALLPATHS:
            if (!parentMap.containsKey(target)) {
              parentMap.put(target, new ArrayList<>());
            }
            parentMap.get(target).add(from);
            break;
          case SOMEPATH:
            parentMap.putIfAbsent(target, ImmutableList.of(from));
            break;
        }

        visitAspectsIfRequired(from, attribute, target);
      }

      if (visited.add(target)) {
        visitEdgesOfTarget(target);
      }
    }

    private void visitEdgesOfTarget(Target target)
        throws InterruptedException, NoSuchThingException {
      try {
        LabelVisitationUtils.visitTarget(
            target,
            edgeFilter,
            (from, attribute, label) -> {
              try {
                enqueue(from, attribute, label);
              } catch (InterruptedException | NoSuchThingException e) {
                // Tunnel the exception, since we can't throw checked exceptions from here.
                throw new CompletionException(e);
              }
            });
      } catch (CompletionException e) {
        Throwables.propagateIfPossible(
            e.getCause(), InterruptedException.class, NoSuchThingException.class);
        throw e;
      }
    }

    private void visitAspectsIfRequired(Target from, Attribute attribute, final Target to)
        throws InterruptedException, NoSuchThingException {
      // TODO(bazel-team): The getAspects call below is duplicate work for each direct dep entailed
      // by an attribute's value. Additionally, we might end up enqueueing the same exact visitation
      // multiple times: consider the case where the same direct dependency is entailed by aspects
      // of *different* attributes. These visitations get culled later, but we still have to pay the
      // overhead for all that.

      if (!(from instanceof Rule) || !(to instanceof Rule)) {
        return;
      }
      Rule fromRule = (Rule) from;
      Rule toRule = (Rule) to;
      for (Aspect aspect : attribute.getAspects(fromRule)) {
        if (AspectDefinition.satisfies(
            aspect, toRule.getRuleClassObject().getAdvertisedProviders())) {
          Multimap<Attribute, Label> allLabels = HashMultimap.create();
          AspectDefinition.addAllAttributesOfAspect(fromRule, allLabels, aspect, edgeFilter);
          for (Map.Entry<Attribute, Label> e : allLabels.entries()) {
            enqueue(from, e.getKey(), e.getValue());
          }
        }
      }
    }
  }
}
