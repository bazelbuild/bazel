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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;


@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 5, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 5, timeUnit = TimeUnit.SECONDS)
@Fork(2)
@State(Scope.Benchmark)
public class Bench {

  @State(Scope.Thread)
  public static class BenchState {
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

    private static final Charset CHARSET = StandardCharsets.ISO_8859_1;
    private static final Mutability mutability = Mutability.create("interpreter");
    private static final FuncallExpression ast = new FuncallExpression(Identifier.of(""), ImmutableList.of());
    private static final Environment env =
        Environment.builder(mutability)
            .useDefaultSemantics()
            .setGlobals(Environment.DEFAULT_GLOBALS)
            .setEventHandler(PRINT_HANDLER)
            .build();

    public static Map<String, UserDefinedFunction> benchmarks = new HashMap<>();

    @Setup(Level.Trial)
    public void doSetup() {
      int ret = executeFile(path);
      if (ret != 0) {
        System.exit(ret);
      }
      for (Map.Entry<String, Object> entry : env.getGlobals().getBindings().entrySet()) {
        if (entry.getKey().startsWith("bench_") && entry.getValue() instanceof UserDefinedFunction) {
          benchmarks.put(entry.getKey(), (UserDefinedFunction) entry.getValue());
        }
      }
    }

    private int executeFile(String path) {
      String content;
      try {
        content = new String(Files.readAllBytes(Paths.get(path)), CHARSET);
        return execute(content);
      } catch (Exception e) {
        e.printStackTrace();
        return 1;
      }
    }

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
  }

  @Param("")
  public static String path;

  @Param("")
  public String benchName;

  @Benchmark
  public Object testBenchmark(BenchState state) throws EvalException, InterruptedException {
    return state.benchmarks.get(benchName).call(Collections.emptyList(), Collections.emptyMap(), state.ast, state.env);
  }

  public static void main(String[] args) throws RunnerException {
    path = args[0];
    BenchState tmp = new BenchState();
    tmp.doSetup();

    Options opt = new OptionsBuilder()
        .include(Bench.class.getSimpleName())
        .param("benchName", tmp.benchmarks.keySet().toArray(new String[tmp.benchmarks.size()]))
        .param("path", path)
        .build();
    new Runner(opt).run();
  }

}
