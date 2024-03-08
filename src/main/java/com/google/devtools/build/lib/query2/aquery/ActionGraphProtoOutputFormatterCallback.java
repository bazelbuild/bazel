// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.aquery;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionException;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryConsumingOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.MonolithicOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.StreamedConsumingOutputHandler;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;

/** Default output callback for aquery, prints proto output. */
public class ActionGraphProtoOutputFormatterCallback extends AqueryThreadsafeCallback {
  // Arbitrarily chosen. Large enough for good performance, small enough not to cause OOMs.
  private static final int BLOCKING_QUEUE_SIZE = Runtime.getRuntime().availableProcessors() * 2;
  private final OutputType outputType;
  private final ActionGraphDump actionGraphDump;
  private final AqueryActionFilter actionFilters;
  private final AqueryOutputHandler aqueryOutputHandler;

  /**
   * Pseudo-arbitrarily chosen buffer size for output. Chosen to be large enough to fit a handful of
   * messages without needing to flush to the underlying output, which may not be buffered.
   */
  private static final int OUTPUT_BUFFER_SIZE = 16384;

  ActionGraphProtoOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      TargetAccessor<ConfiguredTargetValue> accessor,
      OutputType outputType,
      AqueryActionFilter actionFilters) {
    super(eventHandler, options, out, accessor);
    this.outputType = outputType;
    this.actionFilters = actionFilters;
    this.aqueryOutputHandler = constructAqueryOutputHandler(outputType, out, printStream);
    this.actionGraphDump =
        new ActionGraphDump(
            options.includeCommandline,
            options.includeArtifacts,
            options.includeSchedulingDependencies,
            this.actionFilters,
            options.includeParamFiles,
            options.includeFileWriteContents,
            aqueryOutputHandler,
            eventHandler);
  }

  public static AqueryOutputHandler constructAqueryOutputHandler(
      OutputType outputType, OutputStream out, PrintStream printStream) {
    switch (outputType) {
      case BINARY:
      case DELIMITED_BINARY:
      case TEXT:
        return new StreamedConsumingOutputHandler(
            outputType,
            out,
            CodedOutputStream.newInstance(out, OUTPUT_BUFFER_SIZE),
            printStream,
            new LinkedBlockingQueue<>(BLOCKING_QUEUE_SIZE));
      case JSON:
        return new MonolithicOutputHandler(printStream);
    }
    // The above cases are exhaustive.
    throw new AssertionError("Wrong output type: " + outputType);
  }

  @Override
  public String getName() {
    return outputType.formatName();
  }

  @Override
  public void close(boolean failFast) throws IOException {
    if (!failFast) {
      try (SilentCloseable c = Profiler.instance().profile("aqueryOutputHandler.close")) {
        aqueryOutputHandler.close();
      }
    }
  }

  @Override
  public void processOutput(Iterable<ConfiguredTargetValue> partialResult)
      throws IOException, InterruptedException {
    if (aqueryOutputHandler instanceof AqueryConsumingOutputHandler) {
      processOutputInParallel(partialResult);
      return;
    }

    try (SilentCloseable c = Profiler.instance().profile("process partial result")) {
      // Enabling includeParamFiles should enable includeCommandline by default.
      options.includeCommandline |= options.includeParamFiles;

      for (ConfiguredTargetValue configuredTargetValue : partialResult) {
        processSingleEntry(configuredTargetValue);
      }
    } catch (CommandLineExpansionException | TemplateExpansionException e) {
      throw new IOException(e.getMessage());
    }
  }

  private void processSingleEntry(ConfiguredTargetValue configuredTargetValue)
      throws CommandLineExpansionException,
          InterruptedException,
          IOException,
          TemplateExpansionException {
    if (!(configuredTargetValue instanceof RuleConfiguredTargetValue)) {
      // We have to include non-rule values in the graph to visit their dependencies, but they
      // don't have any actions to print out.
      return;
    }
    actionGraphDump.dumpConfiguredTarget((RuleConfiguredTargetValue) configuredTargetValue);
    if (options.useAspects) {
      for (AspectValue aspectValue : accessor.getAspectValues(configuredTargetValue)) {
        actionGraphDump.dumpAspect(aspectValue, configuredTargetValue);
      }
    }
  }

  private void processOutputInParallel(Iterable<ConfiguredTargetValue> partialResult)
      throws IOException, InterruptedException {
    AqueryConsumingOutputHandler aqueryConsumingOutputHandler =
        (AqueryConsumingOutputHandler) aqueryOutputHandler;
    try (SilentCloseable c = Profiler.instance().profile("process partial result")) {
      // Enabling includeParamFiles should enable includeCommandline by default.
      options.includeCommandline |= options.includeParamFiles;
      ForkJoinPool executor =
          NamedForkJoinPool.newNamedPool("aquery", Runtime.getRuntime().availableProcessors());

      try {
        Future<Void> consumerFuture = executor.submit(aqueryConsumingOutputHandler.startConsumer());
        List<Future<Void>> futures = executor.invokeAll(toTasks(partialResult));
        for (Future<Void> future : futures) {
          future.get();
        }
        aqueryConsumingOutputHandler.stopConsumer(/* discardRemainingTasks= */ false);
        // Get any possible exception from the consumer.
        consumerFuture.get();
      } catch (ExecutionException e) {
        aqueryConsumingOutputHandler.stopConsumer(/* discardRemainingTasks= */ true);
        Throwable cause = Throwables.getRootCause(e);
        if (cause instanceof CommandLineExpansionException
            || cause instanceof TemplateExpansionException) {
          // This is kinda weird, but keeping it in line with the status quo for now.
          // TODO(b/266179316): Clean this up.
          throw new IOException(cause.getMessage());
        }
        throwIfInstanceOf(cause, IOException.class);
        throwIfInstanceOf(cause, InterruptedException.class);
        throwIfUnchecked(cause);
        throw new IllegalStateException("Unexpected exception type: ", e);
      } finally {
        executor.shutdown();
      }
    }
  }

  private ImmutableList<AqueryOutputTask> toTasks(Iterable<ConfiguredTargetValue> values) {
    ImmutableList.Builder<AqueryOutputTask> tasks = ImmutableList.builder();
    for (ConfiguredTargetValue value : values) {
      tasks.add(new AqueryOutputTask(value));
    }
    return tasks.build();
  }

  private final class AqueryOutputTask implements Callable<Void> {

    private final ConfiguredTargetValue configuredTargetValue;

    AqueryOutputTask(ConfiguredTargetValue configuredTargetValue) {
      this.configuredTargetValue = configuredTargetValue;
    }

    @Override
    public Void call()
        throws CommandLineExpansionException,
            TemplateExpansionException,
            IOException,
            InterruptedException {
      processSingleEntry(configuredTargetValue);
      return null;
    }
  }
}
