package com.google.devtools.build.lib.util;

import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/** A benchmark for {@link RegexUtil#asOptimizedMatchingPredicate}. */
@BenchmarkMode(Mode.Throughput)
@State(Scope.Benchmark)
@Warmup(iterations = 4, time = 2, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 2, timeUnit = TimeUnit.SECONDS)
@Fork(value = 3)
public class RegexUtilBenchmark {
  @Param({
    "bazel-out/darwin_arm64-opt-exec-ST-fad1763555eb/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer",
    "bazel-out/darwin_arm64-fastbuild/testlogs/src/test/java/com/google/devtools/common/options/AllTests/test.log"
  })
  public String haystack;

  @Param({
    ".*/coverage\\.dat",
    ".*/my_action.outputs/.*",
    ".*/testlogs/.*/test\\.xml",
    ".*/testlogs/.*/attempt_[0-9]+\\.xml"
  })
  public String needle;

  private Pattern originalPattern;
  private Predicate<String> optimizedMatcher;

  @Setup
  public void compile() {
    originalPattern = Pattern.compile(needle);
    optimizedMatcher = RegexUtil.asOptimizedMatchingPredicate(originalPattern);
  }

  @Benchmark
  public boolean baseline() {
    return originalPattern.matcher(haystack).matches();
  }

  @Benchmark
  public boolean optimized() {
    return optimizedMatcher.test(haystack);
  }
}
