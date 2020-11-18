// Copyright 2020 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableMap;
import com.sun.management.ThreadMXBean;
import java.io.File;
import java.lang.management.ManagementFactory;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.lib.json.Json;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;

// TODO(adonovan): document how to obtain a Java CPU profile.

// TODO(adonovan): mitigate the effects of JVM warmup.
// (See Oracle's JMH; we can't use it directly because it
// seems to be entirely driven by Java annotations,
// which is no good for a dynamic suite.)

/**
 * Script-based benchmarks of the Starlark evaluator.
 *
 * <p>Scripts in testdata/bench_*.star are executed, and then each function named {@code bench_*} is
 * repeatedly called and measured. The function has one parameter, b, a Benchmark, that provides
 * b.n, the number of iterations to execute. The function must have resource costs linear in b.n.
 * Typically, the function body is a loop of the form {@code for _ in range(b.n): ...}. Using b.n
 * for other purposes leads to meaningless results. For example, it would be a mistake to use it as
 * the length of a random list to be sorted, because sorting does not run in linear time.
 *
 * <p>A benchmark with significant set-up costs may reset the timer ({@code b.restart()}) before
 * entering its loop. Example:
 *
 * <pre>
 * def bench_my_func(b):
 *     """Description goes here."""
 *     my_setup()
 *     b.restart()
 *     for _ in range(b.n):
 *         my_func()
 * </pre>
 */
public final class Benchmarks {

  private static final String HELP =
      "Usage: Benchmarks [--help] [--filter regex] [--seconds float]\n"
          + "Runs Starlark benchmarks matching the filter for the specified (approximate) time,\n"
          + "and reports various performance measures.";

  private static boolean ok = true;

  public static void main(String[] args) throws Exception {
    Pattern filter = null; // default: all
    long budgetNanos = 1_000_000_000;

    // parse flags
    int i;
    for (i = 0; i < args.length; i++) {
      if (args[i].equals("--")) {
        i++;
        break;

      } else if (args[i].equals("--help")) {
        System.out.println(HELP);
        System.exit(0);

      } else if (args[i].equals("--filter")) {
        if (++i == args.length) {
          fail("--filter needs an argument");
        }
        try {
          filter = Pattern.compile(args[i]);
        } catch (PatternSyntaxException ex) {
          fail("for --filter, invalid regexp: %s", ex.getMessage());
        }

      } else if (args[i].equals("--seconds")) {
        if (++i == args.length) {
          fail("--seconds needs an argument");
        }
        try {
          budgetNanos = (long) (1e9 * Double.parseDouble(args[i]));
        } catch (NumberFormatException unused) {
          fail("for --seconds, got '%s', want floating-point number of seconds", args[i]);
        }
        if (!(0 <= budgetNanos && budgetNanos <= 1e13)) {
          fail("--seconds out of range");
        }

      } else {
        fail("unknown flag: %s", args[i]);
      }
    }
    if (i < args.length) {
      fail("unexpected arguments");
    }

    // Read testdata/bench_* files.
    File src = new File("third_party/bazel/src"); // blaze
    if (!src.exists()) {
      src = new File("src"); // bazel
    }
    File testdata = new File(src, "test/java/net/starlark/java/eval/testdata");
    for (File file : testdata.listFiles()) {
      if (!(file.getName().startsWith("bench_") && file.getName().endsWith(".star"))) {
        continue;
      }

      // parse & execute
      ParserInput input = ParserInput.readFile(file.toString());
      ImmutableMap.Builder<String, Object> predeclared = ImmutableMap.builder();
      predeclared.put("json", Json.INSTANCE);

      Module module = Module.withPredeclared(semantics, predeclared.build());
      try (Mutability mu = Mutability.create("test")) {
        StarlarkThread thread = new StarlarkThread(mu, semantics);
        Starlark.execFile(input, FileOptions.DEFAULT, module, thread);

      } catch (SyntaxError.Exception ex) {
        for (SyntaxError err : ex.errors()) {
          System.err.println(err); // includes location
          ok = false;
          continue;
        }
      } catch (EvalException ex) {
        System.err.println(ex.getMessageWithStack());
        ok = false;
        continue;

      } catch (
          @SuppressWarnings("InterruptedExceptionSwallowed")
          Throwable ex) {
        // unhandled exception (incl. InterruptedException)
        System.err.printf("in %s: %s\n", file, ex.getMessage());
        ex.printStackTrace();
        ok = false;
        continue;
      }

      // Sort bench_* functions by name.
      TreeMap<String, StarlarkFunction> benchmarks = new TreeMap<>();
      for (Map.Entry<String, Object> e : module.getExportedGlobals().entrySet()) {
        if (e.getKey().startsWith("bench_") && e.getValue() instanceof StarlarkFunction) {
          if (filter == null || filter.matcher(e.getKey()).find()) {
            benchmarks.put(e.getKey(), (StarlarkFunction) e.getValue());
          }
        }
      }
      if (benchmarks.isEmpty()) {
        if (filter == null) {
          System.err.printf("File %s: no bench_* functions\n", file);
          ok = false;
        }
        continue;
      }

      // Run benchmarks.
      System.out.printf("File %s:\n", file);
      System.out.printf(
          "%-20s %10s %10s %10s %10s %10s\n", //
          "benchmark", "ops", "cpu/op", "wall/op", "steps/op", "alloc/op");
      for (Map.Entry<String, StarlarkFunction> e : benchmarks.entrySet()) {
        String name = e.getKey();
        System.out.flush(); // help user identify a slow benchmark
        Benchmark b = run(name, e.getValue(), budgetNanos);
        if (b == null) {
          ok = false;
          continue;
        }
        System.out.printf(
            "%-20s %10d %10s %10s %10d %10s\n",
            name,
            b.count,
            formatDuration(((double) b.time) / b.count),
            formatDuration(((double) b.cpu) / b.count),
            b.steps / b.count,
            formatBytes(b.alloc / b.count));
      }
      System.out.println();
    }
    if (!ok) {
      System.exit(1);
    }
  }

  private static void fail(String format, Object... args) {
    System.err.printf(format, args);
    System.err.println();
    System.exit(1);
  }

  // Runs benchmark function f for the specified time budget
  // (which we may exceed by a factor of two).
  private static Benchmark run(String name, StarlarkFunction f, long budgetNanos) {
    Mutability mu = Mutability.create("test");
    StarlarkThread thread = new StarlarkThread(mu, semantics);

    Benchmark b = new Benchmark();

    // Keep doubling the number of iterations until we exceed the deadline.
    // TODO(adonovan): opt: extrapolate and predict the number of iterations
    // in the remaining time budget, being wary of extrapolation error.
    for (b.n = 1; b.time < budgetNanos; b.n <<= 1) {
      if (b.n <= 0) {
        System.err.printf(
            "In %s: function is too fast; likely a loop over `range(b.n)` is missing\n", name);
        return null;
      }

      try {
        b.start(thread);
        Starlark.fastcall(thread, f, new Object[] {b}, new Object[0]);
        b.stop(thread);

      } catch (EvalException ex) {
        System.err.println(ex.getMessageWithStack());
        return null;

      } catch (
          @SuppressWarnings("InterruptedExceptionSwallowed")
          Throwable ex) {
        // unhandled exception (incl. InterruptedException)
        System.err.printf("In %s: %s\n", name, ex.getMessage());
        ex.printStackTrace();
        return null;
      }
    }

    return b;
  }

  // The type of the parameter to each bench(b) function.
  // Provides n, the number of iterations.
  @StarlarkBuiltin(name = "Benchmark")
  private static class Benchmark implements StarlarkValue {

    // The cast assumes we use the "Sun" JVM, which measures per-thread allocation and CPU.
    private final ThreadMXBean threadMX = (ThreadMXBean) ManagementFactory.getThreadMXBean();

    // Starlark attributes
    private int n; // requested number of iterations

    // current span  (time0 != 0 => started)
    private long cpu0;
    private long alloc0;
    private long time0;
    private long steps0;

    // accumulators
    private int count; // iterations
    private long cpu; // CPU time (ns) in this thread
    private long alloc; // bytes allocated by this thread
    private long time; // wall time (ns)
    private long steps; // Starlark computation steps

    @StarlarkMethod(name = "n", doc = "Requested number of iterations.", structField = true)
    public int n() {
      return n;
    }

    @StarlarkMethod(name = "start", doc = "Starts the timer.", useStarlarkThread = true)
    public void start(StarlarkThread thread) throws EvalException {
      if (time0 != 0) {
        throw Starlark.errorf("timer already started");
      }

      this.cpu0 = threadMX.getCurrentThreadCpuTime();
      this.alloc0 = threadMX.getThreadAllocatedBytes(Thread.currentThread().getId());
      this.steps0 = thread.getExecutedSteps();
      this.time0 = System.nanoTime();
    }

    @StarlarkMethod(name = "stop", doc = "Starts the timer.", useStarlarkThread = true)
    public void stop(StarlarkThread thread) throws EvalException {
      if (time0 == 0) {
        throw Starlark.errorf("timer already stopped");
      }
      long time1 = System.nanoTime();
      long steps1 = thread.getExecutedSteps();
      long alloc1 = threadMX.getThreadAllocatedBytes(Thread.currentThread().getId());
      long cpu1 = threadMX.getCurrentThreadCpuTime();

      this.time += time1 - this.time0;
      this.steps += steps1 - this.steps0;
      this.alloc += alloc1 - this.alloc0;
      this.cpu += cpu1 - this.cpu0;

      this.count += this.n;

      time0 = 0; // stopped
    }

    @StarlarkMethod(name = "restart", doc = "Restarts the timer.", useStarlarkThread = true)
    public void restart(StarlarkThread thread) throws EvalException {
      time0 = 0; // stop, and discard current span
      start(thread);
    }

    @Override
    public void repr(Printer p) {
      p.append("<Benchmark>");
    }
  }

  private static String formatDuration(double ns) {
    // (Similar format to Go's time.Duration.)
    if (ns == 0) {
      return "0s";
    } else if (ns < 1e3) {
      return String.format("%dns", (long) ns);
    } else if (ns < 1e6) {
      return String.format("%.3gµs", ns / 1e3);
    } else if (ns < 1e9) {
      return String.format("%.6gms", ns / 1e6);
    } else {
      return String.format("%.3gs", ns / 1e9);
    }
  }

  private static String formatBytes(long bytes) {
    if (bytes == 0) {
      return "0B";
    } else if (bytes < 1e3) {
      return String.format("%dB", bytes);
    } else if (bytes < 1e6) {
      return String.format("%.3gKB", bytes / 1e3);
    } else if (bytes < 1e9) {
      return String.format("%.6gMB", bytes / 1e6);
    } else {
      return String.format("%.3gGB", bytes / 1e9);
    }
  }

  private static final StarlarkSemantics semantics = StarlarkSemantics.DEFAULT;

  private Benchmarks() {}
}
