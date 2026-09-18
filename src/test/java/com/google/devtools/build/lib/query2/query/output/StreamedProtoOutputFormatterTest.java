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

package com.google.devtools.build.lib.query2.query.output;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.common.options.OptionsParser;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link StreamedProtoOutputFormatter}. */
@RunWith(TestParameterInjector.class)
public final class StreamedProtoOutputFormatterTest extends PackageLoadingTestCase {
  private static final String PARALLEL_FLAG = "--experimental_parallel_streamed_proto_output";
  private static final String OUTPUT_THREAD_PREFIX = "streamed-proto-output-";
  private static final int TEST_TIMEOUT_SECONDS = 10;
  private static final int THREAD_TERMINATION_TIMEOUT_SECONDS = 5;

  private Target ruleA;
  private Target ruleB;
  private ImmutableList<Target> representativeTargets;

  @Before
  public void createTargets() throws Exception {
    scratch.file(
        "pkg/defs.bzl",
        """
        def _aspect_impl(target, ctx):
            return []

        test_aspect = aspect(
            implementation = _aspect_impl,
            attrs = {"_aspect_dep": attr.label(default = ":b")},
        )

        def _rule_impl(ctx):
            return []

        test_rule = rule(
            implementation = _rule_impl,
            attrs = {"deps": attr.label_list(aspects = [test_aspect])},
        )
        """);
    scratch.file("pkg/input.txt", "input");
    scratch.file(
        "pkg/BUILD",
        """
        load(":defs.bzl", "test_rule")

        test_rule(
            name = "a",
            deps = [":b"],
        )

        test_rule(name = "b")

        filegroup(
            name = "files",
            srcs = ["input.txt"],
        )

        genrule(
            name = "generated",
            srcs = ["input.txt"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
        )

        package_group(
            name = "group",
            packages = ["//pkg"],
        )
        """);

    ruleA = getTarget("//pkg:a");
    ruleB = getTarget("//pkg:b");
    representativeTargets =
        ImmutableList.of(
            ruleA,
            ruleB,
            getTarget("//pkg:files"),
            getTarget("//pkg:input.txt"),
            getTarget("//pkg:out.txt"),
            getTarget("//pkg:group"));
  }

  @Test
  public void parallelOutputIsByteIdenticalWhenWorkersCompleteOutOfOrder() throws Exception {
    assumeTrue(Runtime.getRuntime().availableProcessors() > 1);
    int targetsPerChunk = StreamedProtoOutputFormatter.TARGETS_PER_CHUNK;
    ImmutableList<Target> targets =
        ImmutableList.<Target>builderWithExpectedSize(2 * targetsPerChunk)
            .addAll(repeatTargets(targetsPerChunk, ruleA))
            .addAll(repeatTargets(targetsPerChunk, ruleB))
            .build();
    byte[] serial =
        format(
            new StreamedProtoOutputFormatter(),
            options(),
            AspectResolver.Mode.CONSERVATIVE,
            targets);
    OutOfOrderFormatter parallelFormatter = new OutOfOrderFormatter(ruleA, ruleB);

    byte[] parallel =
        format(
            parallelFormatter, options(PARALLEL_FLAG), AspectResolver.Mode.CONSERVATIVE, targets);

    assertThat(parallel).isEqualTo(serial);
    assertThat(parseTargets(parallel)).containsExactlyElementsIn(parseTargets(serial)).inOrder();
    assertThat(parallelFormatter.firstChunkStarted.getCount()).isEqualTo(0);
    assertThat(parallelFormatter.laterChunkStarted.getCount()).isEqualTo(0);
    assertThat(parallelFormatter.threadNames.size()).isAtLeast(2);
    assertNoParallelOutputThreads();
  }

  @Test
  public void parallelOutputMatchesSerialForAspectMode(
      @TestParameter({"OFF", "CONSERVATIVE", "PRECISE"}) AspectResolver.Mode mode)
      throws Exception {
    ImmutableList<Target> targets =
        repeatTargets(350, representativeTargets.toArray(Target[]::new));

    byte[] serial = format(new StreamedProtoOutputFormatter(), options(), mode, targets);
    byte[] parallel =
        format(new StreamedProtoOutputFormatter(), options(PARALLEL_FLAG), mode, targets);

    assertThat(parallel).isEqualTo(serial);
    assertNoParallelOutputThreads();
  }

  @Test
  public void emptyInputWritesNothing() throws Exception {
    byte[] output =
        format(
            new StreamedProtoOutputFormatter(),
            options(PARALLEL_FLAG),
            AspectResolver.Mode.CONSERVATIVE,
            ImmutableList.of());

    assertThat(output).isEmpty();
    assertNoParallelOutputThreads();
  }

  @Test
  public void flagDisabledUsesSerialCallback() throws Exception {
    RecordingFormatter formatter = new RecordingFormatter();

    format(formatter, options(), AspectResolver.Mode.CONSERVATIVE, representativeTargets);

    assertThat(formatter.threadNames).containsExactly(Thread.currentThread().getName());
    assertNoParallelOutputThreads();
  }

  @Test
  public void ruleClassesForcesSerialAndPreservesFirstSeenInfo() throws Exception {
    ImmutableList<Target> targets = ImmutableList.of(ruleA, ruleB);
    byte[] serial =
        format(
            new StreamedProtoOutputFormatter(),
            options("--proto:rule_classes"),
            AspectResolver.Mode.CONSERVATIVE,
            targets);
    RecordingFormatter formatter = new RecordingFormatter();

    byte[] parallel =
        format(
            formatter,
            options(PARALLEL_FLAG, "--proto:rule_classes"),
            AspectResolver.Mode.CONSERVATIVE,
            targets);

    assertThat(parallel).isEqualTo(serial);
    assertThat(formatter.threadNames).containsExactly(Thread.currentThread().getName());
    ImmutableList<Build.Target> parsed = parseTargets(parallel);
    assertThat(parsed.get(0).getRule().getRuleClassKey())
        .isEqualTo(parsed.get(1).getRule().getRuleClassKey());
    assertThat(parsed.get(0).getRule().hasRuleClassInfo()).isTrue();
    assertThat(parsed.get(1).getRule().hasRuleClassInfo()).isFalse();
    assertNoParallelOutputThreads();
  }

  @Test
  public void outputIOExceptionIsRethrownByIdentityAndWorkersExit() throws Exception {
    IOException expected = new IOException("expected output failure");
    OutputStream failingOutput =
        new OutputStream() {
          @Override
          public void write(int value) throws IOException {
            throw expected;
          }
        };

    IOException actual =
        assertThrows(
            IOException.class,
            () ->
                format(
                    new StreamedProtoOutputFormatter(),
                    options(PARALLEL_FLAG),
                    AspectResolver.Mode.CONSERVATIVE,
                    repeatTargets(350, representativeTargets.toArray(Target[]::new)),
                    failingOutput));

    assertThat(actual).isSameInstanceAs(expected);
    assertNoParallelOutputThreads();
  }

  @Test
  public void workerInterruptedExceptionIsRethrownByIdentity() throws Exception {
    InterruptedException expected = new InterruptedException("expected worker interruption");

    InterruptedException actual =
        assertThrows(
            InterruptedException.class,
            () ->
                format(
                    new InterruptingFormatter(expected),
                    options(PARALLEL_FLAG),
                    AspectResolver.Mode.CONSERVATIVE,
                    ImmutableList.of(ruleA)));

    assertThat(actual).isSameInstanceAs(expected);
    assertNoParallelOutputThreads();
  }

  @Test
  public void uncheckedWorkerExceptionIsRethrownByIdentity() throws Exception {
    IllegalStateException expected = new IllegalStateException("expected worker failure");

    IllegalStateException actual =
        assertThrows(
            IllegalStateException.class,
            () ->
                format(
                    new UncheckedFailingFormatter(expected),
                    options(PARALLEL_FLAG),
                    AspectResolver.Mode.CONSERVATIVE,
                    ImmutableList.of(ruleA)));

    assertThat(actual).isSameInstanceAs(expected);
    assertNoParallelOutputThreads();
  }

  @Test
  public void interruptingCallbackThreadCancelsWorkersAndWaitsForExit() throws Exception {
    BlockingFormatter formatter = new BlockingFormatter();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread callbackThread =
        Thread.ofPlatform()
            .name("streamed-proto-test-callback")
            .unstarted(
                () -> {
                  try {
                    format(
                        formatter,
                        options(PARALLEL_FLAG),
                        AspectResolver.Mode.CONSERVATIVE,
                        repeatTargets(350, ruleA));
                  } catch (Throwable t) {
                    failure.set(t);
                  }
                });

    callbackThread.start();
    try {
      assertThat(formatter.workerStarted.await(TEST_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
      callbackThread.interrupt();
      callbackThread.join(TimeUnit.SECONDS.toMillis(TEST_TIMEOUT_SECONDS));
      assertThat(callbackThread.isAlive()).isFalse();
    } finally {
      formatter.releaseWorkers.countDown();
      callbackThread.interrupt();
      callbackThread.join(TimeUnit.SECONDS.toMillis(TEST_TIMEOUT_SECONDS));
    }

    assertThat(failure.get()).isInstanceOf(InterruptedException.class);
    assertThat(formatter.workerExited.await(TEST_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
    assertNoParallelOutputThreads();
  }

  private byte[] format(
      StreamedProtoOutputFormatter formatter,
      QueryOptions queryOptions,
      AspectResolver.Mode mode,
      Iterable<Target> targets)
      throws Exception {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    format(formatter, queryOptions, mode, targets, output);
    return output.toByteArray();
  }

  private void format(
      StreamedProtoOutputFormatter formatter,
      QueryOptions queryOptions,
      AspectResolver.Mode mode,
      Iterable<Target> targets,
      OutputStream output)
      throws Exception {
    formatter.setOptions(
        queryOptions,
        mode.createResolver(getPackageManager(), reporter),
        DigestHashFunction.SHA256.getHashFunction());
    formatter.setEventHandler(reporter);
    OutputFormatterCallback.processAllTargets(
        formatter.createPostFactoStreamCallback(output, queryOptions, LabelPrinter.legacy()),
        targets);
  }

  private static QueryOptions options(String... flags) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(QueryOptions.class).build();
    parser.parse(flags);
    return parser.getOptions(QueryOptions.class);
  }

  private static ImmutableList<Target> repeatTargets(int count, Target... targets) {
    ImmutableList.Builder<Target> repeated = ImmutableList.builderWithExpectedSize(count);
    for (int i = 0; i < count; i++) {
      repeated.add(targets[i % targets.length]);
    }
    return repeated.build();
  }

  private static ImmutableList<Build.Target> parseTargets(byte[] output) throws IOException {
    ByteArrayInputStream input = new ByteArrayInputStream(output);
    ImmutableList.Builder<Build.Target> targets = ImmutableList.builder();
    Build.Target target;
    while ((target = Build.Target.parseDelimitedFrom(input)) != null) {
      targets.add(target);
    }
    return targets.build();
  }

  private static void assertNoParallelOutputThreads() throws InterruptedException {
    long deadlineNanos =
        System.nanoTime() + TimeUnit.SECONDS.toNanos(THREAD_TERMINATION_TIMEOUT_SECONDS);
    while (System.nanoTime() < deadlineNanos) {
      if (findLiveParallelOutputThreads().isEmpty()) {
        return;
      }
      Thread.sleep(10);
    }
    assertThat(findLiveParallelOutputThreads()).isEmpty();
  }

  private static List<String> findLiveParallelOutputThreads() {
    return Thread.getAllStackTraces().keySet().stream()
        .filter(Thread::isAlive)
        .map(Thread::getName)
        .filter(name -> name.startsWith(OUTPUT_THREAD_PREFIX))
        .toList();
  }

  private static class RecordingFormatter extends StreamedProtoOutputFormatter {
    final Set<String> threadNames = ConcurrentHashMap.newKeySet();

    @Override
    public Build.Target toTargetProtoBuffer(Target target, LabelPrinter labelPrinter)
        throws InterruptedException {
      threadNames.add(Thread.currentThread().getName());
      return super.toTargetProtoBuffer(target, labelPrinter);
    }
  }

  private static final class OutOfOrderFormatter extends RecordingFormatter {
    private final Target slowTarget;
    private final Target releaseTarget;
    final CountDownLatch firstChunkStarted = new CountDownLatch(1);
    final CountDownLatch laterChunkStarted = new CountDownLatch(1);
    private final AtomicInteger releaseTargetConversions = new AtomicInteger();

    OutOfOrderFormatter(Target slowTarget, Target releaseTarget) {
      this.slowTarget = slowTarget;
      this.releaseTarget = releaseTarget;
    }

    @Override
    public Build.Target toTargetProtoBuffer(Target target, LabelPrinter labelPrinter)
        throws InterruptedException {
      if (Thread.currentThread().getName().startsWith(OUTPUT_THREAD_PREFIX)) {
        if (target == slowTarget) {
          firstChunkStarted.countDown();
          assertThat(laterChunkStarted.await(TEST_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
        }
      }
      Build.Target proto = super.toTargetProtoBuffer(target, labelPrinter);
      if (Thread.currentThread().getName().startsWith(OUTPUT_THREAD_PREFIX)
          && target == releaseTarget
          && releaseTargetConversions.incrementAndGet()
              == StreamedProtoOutputFormatter.TARGETS_PER_CHUNK) {
        laterChunkStarted.countDown();
      }
      return proto;
    }
  }

  private static final class InterruptingFormatter extends StreamedProtoOutputFormatter {
    private final InterruptedException failure;

    InterruptingFormatter(InterruptedException failure) {
      this.failure = failure;
    }

    @Override
    public Build.Target toTargetProtoBuffer(Target target, LabelPrinter labelPrinter)
        throws InterruptedException {
      throw failure;
    }
  }

  private static final class UncheckedFailingFormatter extends StreamedProtoOutputFormatter {
    private final IllegalStateException failure;

    UncheckedFailingFormatter(IllegalStateException failure) {
      this.failure = failure;
    }

    @Override
    public Build.Target toTargetProtoBuffer(Target target, LabelPrinter labelPrinter) {
      throw failure;
    }
  }

  private static final class BlockingFormatter extends StreamedProtoOutputFormatter {
    final CountDownLatch workerStarted = new CountDownLatch(1);
    final CountDownLatch workerExited = new CountDownLatch(1);
    final CountDownLatch releaseWorkers = new CountDownLatch(1);

    @Override
    public Build.Target toTargetProtoBuffer(Target target, LabelPrinter labelPrinter)
        throws InterruptedException {
      workerStarted.countDown();
      try {
        releaseWorkers.await();
      } finally {
        workerExited.countDown();
      }
      return super.toTargetProtoBuffer(target, labelPrinter);
    }
  }
}
