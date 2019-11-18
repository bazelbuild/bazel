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
//

package com.google.devtools.build.lib.bazel.rules.ninja.graph;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.lib.bazel.rules.ninja.file.CollectingListFuture;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class NinjaParallelGraphReducer {
  private final ImmutableSortedMap<PathFragment, NinjaTarget> targets;
  private final ImmutableList<Pair<NinjaTarget, AtomicBoolean>> list;
  private final ImmutableSortedMap<PathFragment, Integer> visited;
  private ImmutableList<NinjaTarget> reducedTargets;
  private final List<PathFragment> directlyRequestedOutputs;

  public NinjaParallelGraphReducer(Collection<NinjaTarget> targets,
      List<PathFragment> directlyRequestedOutputs) {
    Map<PathFragment, NinjaTarget> targetsMap = Maps.newHashMap();
    ImmutableSortedMap.Builder<PathFragment, Integer> visitedBuilder =
        ImmutableSortedMap.naturalOrder();
    ImmutableList.Builder<Pair<NinjaTarget, AtomicBoolean>> listBuilder = ImmutableList.builder();
    int cnt = 0;
    for (NinjaTarget target : targets) {
      listBuilder.add(Pair.of(target, new AtomicBoolean(false)));
      for (PathFragment output : target.getAllOutputs()) {
        targetsMap.put(output, target);
        visitedBuilder.put(output, cnt);
      }
      ++ cnt;
    }
    this.targets = ImmutableSortedMap.copyOf(targetsMap);
    this.visited = visitedBuilder.build();
    this.directlyRequestedOutputs = directlyRequestedOutputs;
    this.list = listBuilder.build();
  }

  private boolean markVisited(PathFragment fragment) {
    Integer index = visited.get(fragment);
    if (index == null) {
      return false;
    }
    AtomicBoolean atomicBoolean = Objects
        .requireNonNull(list.get(index).getSecond());
    return atomicBoolean.compareAndSet(false, true);
  }

  public void reduce(ListeningExecutorService service, int parallellism)
      throws InterruptedException {
    ConcurrentLinkedDeque<Collection<PathFragment>> queue = new ConcurrentLinkedDeque<>();
    cutAndAddToBlockingQueue(directlyRequestedOutputs, queue);
    directlyRequestedOutputs.forEach(this::markVisited);

    CollectingListFuture<Void, RuntimeException> future = new CollectingListFuture<>(
        RuntimeException.class);
    for (int i = 0; i < parallellism; i++) {
      future.add(service.submit(new Worker(queue)));
    }
    future.getResult();

    List<NinjaTarget> filteredList = list.parallelStream().filter(pair -> pair.getSecond().get())
        .map(Pair::getFirst)
        .collect(Collectors.toList());
    reducedTargets = ImmutableList.copyOf(filteredList);
  }

  public ImmutableList<NinjaTarget> getReducedTargets() {
    return reducedTargets;
  }

  private class Worker implements Callable<Void> {
    private final ConcurrentLinkedDeque<Collection<PathFragment>> queue;

    private Worker(ConcurrentLinkedDeque<Collection<PathFragment>> queue) {
      this.queue = queue;
    }

    @Override
    public Void call() throws Exception {
      while (true) {
        Collection<PathFragment> fragments = queue.poll();
        if (fragments == null) {
          break;
        }
        List<PathFragment> newFragments = Lists.newArrayList();
        for (PathFragment fragment : fragments) {
          Collection<PathFragment> inputs = targets.get(fragment).getAllInputs();
          for (PathFragment input : inputs) {
            if (NinjaParallelGraphReducer.this.markVisited(input)) {
              newFragments.add(input);
            }
          }
        }
        cutAndAddToBlockingQueue(newFragments, queue);
      }
      return null;
    }
  }

  private final static int PATHS_IN_TASK = 40;
  private static void cutAndAddToBlockingQueue(
      List<PathFragment> fragments,
      ConcurrentLinkedDeque<Collection<PathFragment>> blockingQueue) {
    if (fragments.isEmpty()) {
      return;
    }
    int num = (int) Math.floor((((double) fragments.size()) / PATHS_IN_TASK));
    int from = 0;
    int limit = Math.max(1, num);
    for (int i = 0; i < limit; i++) {
      int to = (i < (limit - 1)) ? (from + PATHS_IN_TASK) : fragments.size();
      blockingQueue.add(fragments.subList(from, to));
    }
  }
}
