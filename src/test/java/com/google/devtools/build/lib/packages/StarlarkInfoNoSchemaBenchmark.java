// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import net.starlark.java.eval.StarlarkInt;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/** Measures transient struct construction separately from construction followed by compaction. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(2)
public class StarlarkInfoNoSchemaBenchmark {
  @State(Scope.Thread)
  public static class StructState {
    @Param({"4", "16", "64"})
    public int fieldCount;

    @Param({"1", "128"})
    public int layouts;

    private Map<String, Object>[] inputs;
    private StarlarkInfo[] retained;
    private int next;

    @SuppressWarnings("unchecked")
    @Setup
    public void setup() {
      inputs = new Map[layouts];
      retained = new StarlarkInfo[layouts];
      for (int layout = 0; layout < layouts; layout++) {
        Map<String, Object> values = new HashMap<>();
        for (int i = fieldCount - 1; i >= 0; i--) {
          values.put("layout" + layout + "_field" + i, StarlarkInt.of(i));
        }
        inputs[layout] = values;
        retained[layout] = StarlarkInfo.create(StructProvider.STRUCT, values);
        retained[layout] = retained[layout].unsafeOptimizeMemoryLayout();
        retained[layout].getFieldNames();
      }
    }

    Map<String, Object> nextInput() {
      Map<String, Object> input = inputs[next];
      next = (next + 1) % inputs.length;
      return input;
    }
  }

  @Benchmark
  public StarlarkInfo construct(StructState state) {
    return StarlarkInfo.create(StructProvider.STRUCT, state.nextInput());
  }

  @Benchmark
  public StarlarkInfo constructAndCompact(StructState state) {
    return StarlarkInfo.create(StructProvider.STRUCT, state.nextInput())
        .unsafeOptimizeMemoryLayout();
  }
}
