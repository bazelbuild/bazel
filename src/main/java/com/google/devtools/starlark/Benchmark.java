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
package com.google.devtools.starlark;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.*;

import java.util.Arrays;

/**
 * Benchmark for the Starlark interpreter.
 */
class Benchmark {

  private static final EventHandler PRINT_HANDLER =
      new EventHandler() {
        @Override
        public void handle(Event event) {
          if (event.getKind() == EventKind.ERROR) {
            System.err.println(event.getMessage());
          } else {
            System.out.println(event.getMessage());
          }
        }
      };

  private final Mutability mutability = Mutability.create("interpreter");
  private final Environment env =
      Environment.builder(mutability)
          .useDefaultSemantics()
          .setGlobals(Environment.DEFAULT_GLOBALS)
          .setEventHandler(PRINT_HANDLER)
          .build();

  /**
   * Execute a Starlark command.
   */
  private int execute(String content) {
    try {
      BuildFileAST.eval(env, content);
      return 0;
    } catch (EvalException e) {
      System.err.println(e.print());
      return 1;
    } catch (Exception e) {
      e.printStackTrace(System.err);
      return 1;
    }
  }

  private static final String[][] benchmarks = new String[][]{
      {
          "bubble_sort",
          "" +
              "def bubble_sort(array):\n" +
              "    array = list(array)\n" +
              "    for i in range(len(array)):\n" +
              "        for j in range((len(array) - i) - 1):\n" +
              "            if array[j] > array[j + 1]:\n" +
              "                array[j], array[j + 1] = array[j + 1], array[j]\n" +
              "    return array\n" +
              "\n" +
              "def bench():\n" +
              "    for i in range(100000):\n" +
              "        if [2, 3, 4, 5, 6, 7, 9] != bubble_sort([9, 3, 5, 4, 7, 2, 6]):\n" +
              "            fail()\n" +
              "\n" +
              "bench()\n",
      },
      {
          "sorted_strings",
          "" +
              "def foobar():\n" +
              "    return sorted([\"ab\", \"zzb\", \"aab\", \"aaa2\", \"zzz2\", \"ffv2\"])\n" +
              "\n" +
              "def bench():\n" +
              "    for i in range(1000000):\n" +
              "        foobar()\n" +
              "\n" +
              "bench()\n",
      },
  };

  private static void runNamedBenchmark(String arg) {
    for (String[] benchmark : benchmarks) {
      if (benchmark[0].equals(arg)) {
        runBenchmarks(new String[][]{benchmark});
        return;
      }
    }
    System.err.println("unknown benchmark: " + arg);
    System.exit(1);
  }

  private static String rightPad(String s, int len) {
    StringBuilder b = new StringBuilder(s);
    while (b.length() < len) {
      b.append(" ");
    }
    return b.toString();
  }

  private static void runBenchmarks(String[][] benchmarks) {
    long[] mins = new long[benchmarks.length];
    Arrays.fill(mins, Long.MAX_VALUE);

    int maxNameLength = Arrays.stream(benchmarks).mapToInt(p -> p[0].length()).max().getAsInt();

    for (int i = 0; ; ++i) {
      System.out.println("iteration " + i);
      for (int j = 0; j != benchmarks.length; ++j) {
        String name = benchmarks[j][0];
        String program = benchmarks[j][1];
        long start = System.currentTimeMillis();
        int ret = new Benchmark().execute(program);
        long d = System.currentTimeMillis() - start;
        if (ret != 0) {
          System.exit(ret);
        }
        mins[j] = Math.min(d, mins[j]);
        System.out.println(
            rightPad(name, maxNameLength) + ": " + String.format("%4dms; min %4dms", d, mins[j]));
      }
    }
  }

  public static void main(String[] args) {
    if (args.length == 1) {
      runNamedBenchmark(args[0]);
    } else if (args.length == 0) {
      runBenchmarks(benchmarks);
    } else {
      System.err.println("Usage: Benchmark [benchmark]");
      System.exit(1);
    }
  }
}
