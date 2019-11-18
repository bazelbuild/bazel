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

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.bazel.rules.ninja.graph.NinjaGraphReducer;
import com.google.devtools.build.lib.bazel.rules.ninja.graph.NinjaParallelGraphReducer;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.InputKind;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.OutputKind;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link NinjaParallelGraphReducer}.
 */
@RunWith(JUnit4.class)
public class NinjaGraphReducerTest {

  @Test
  public void testSimpleGraph() throws InterruptedException {
    ImmutableList<NinjaTarget> list = ImmutableList.of(
        create("out", "middle1", "middle2", "middle3"),
        create("middle1", "leaf1", "leaf2"),
        create("middle2", "leaf1", "leaf3"),
        create("extra", "leaf1", "leaf3"),
        create("extra3", "leaf1", "middle1")
    );
    NinjaGraphReducer reducer = new NinjaGraphReducer(list,
        Collections.singletonList(PathFragment.create("out")));
    reducer.reduce();
    assertThat(reducer.getReducedTargets()).hasSize(3);
    NinjaParallelGraphReducer pReducer = new NinjaParallelGraphReducer(list,
        Collections.singletonList(PathFragment.create("out")));
    ListeningExecutorService service =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(
                25,
                new ThreadFactoryBuilder()
                    .setNameFormat(NinjaGraphReducerTest.class.getSimpleName() + "-%d")
                    .build()));
    try {
      pReducer.reduce(service, 2);
    } finally {
      ExecutorUtil.interruptibleShutdown(service);
    }
    assertThat(pReducer.getReducedTargets()).hasSize(3);
  }

  private NinjaTarget create(String output, String... inputs) {
    NinjaTarget.Builder builder = NinjaTarget.builder();
    builder.setRuleName("command");
    builder.addOutputs(OutputKind.USUAL, Collections.singleton(PathFragment.create(output)));
    builder.addInputs(InputKind.USUAL,
        Arrays.stream(inputs).map(PathFragment::create).collect(Collectors.toList()));
    return builder.build();
  }

  @Test
  public void testReduceRegularBigGraph() throws InterruptedException {
    Pair<List<NinjaTarget>, List<PathFragment>> pair = generateRegularBigGraph(7);
    List<NinjaTarget> ninjaTargets = pair.getFirst();
    assertThat(ninjaTargets).isNotNull();
    System.out.printf("Generated %d targets.%n", ninjaTargets.size());

    ListeningExecutorService service =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(
                25,
                new ThreadFactoryBuilder()
                    .setNameFormat(NinjaGraphReducerTest.class.getSimpleName() + "-%d")
                    .build()));

    try {
      NinjaParallelGraphReducer pReducer = new NinjaParallelGraphReducer(ninjaTargets, pair.getSecond());
      NinjaGraphReducer sReducer = new NinjaGraphReducer(ninjaTargets, pair.getSecond());

      Stopwatch started = Stopwatch.createStarted();
      sReducer.reduce();
      System.out.println("Reduced in:" + started.elapsed(TimeUnit.MILLISECONDS));
      Stopwatch started1 = Stopwatch.createStarted();
      pReducer.reduce(service, 6);
      System.out.println("Reduced in parallel in:" + started1.elapsed(TimeUnit.MILLISECONDS));

      ImmutableList<NinjaTarget> reducedTargets = sReducer.getReducedTargets();
      assertThat(reducedTargets).hasSize(pReducer.getReducedTargets().size());

      System.out.println("Reduced to:" + reducedTargets.size());
      System.out.println("Reduced to:" + pReducer.getReducedTargets().size());
    } finally {
      ExecutorUtil.interruptibleShutdown(service);
    }
  }

  private static Pair<List<NinjaTarget>, List<PathFragment>> generateRegularBigGraph(int level) {
    ArrayDeque<Pair<Integer, NinjaTarget>> queue = new ArrayDeque<>();
    queue.add(Pair.of(level, createTarget(10, PathFragment.create("result"))));
    List<NinjaTarget> allTargets = Lists.newArrayList();
    List<PathFragment> level3 = Lists.newArrayList();
    while (!queue.isEmpty()) {
      Pair<Integer, NinjaTarget> pair = queue.removeFirst();
      int nextLevel = pair.getFirst() - 1;
      NinjaTarget target = pair.getSecond();
      allTargets.add(target);
      if (nextLevel == 2) {
        level3.add(Iterables.getFirst(target.getAllOutputs(), null));
      }
      if (nextLevel > 0) {
        for (PathFragment input : target.getAllInputs()) {
          queue.add(Pair.of(nextLevel, createTarget(10, input)));
        }
      }
    }
    Collections.shuffle(level3);
    return Pair.of(allTargets, level3.subList(0, level3.size()/2));
  }

  private static NinjaTarget createTarget(int numInputs, PathFragment output) {
    NinjaTarget.Builder builder = NinjaTarget.builder();
    builder.setRuleName("rule123");
    builder.addOutputs(OutputKind.USUAL, Collections.singleton(output));
    builder.addInputs(InputKind.USUAL, IntStream.range(0, numInputs)
        .mapToObj(i -> PathFragment.create(output.getPathString() + "/sub" + i))
        .collect(Collectors.toList()));
    return builder.build();
  }
}
