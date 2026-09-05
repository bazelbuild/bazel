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
package com.google.devtools.build.lib.query2.query.output;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.MoreFutures;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

/**
 * An output formatter that outputs a protocol buffer representation of a query result and outputs
 * the proto bytes to the output print stream. By taking the bytes and calling {@code mergeFrom()}
 * on a {@code Build.QueryResult} object the full result can be reconstructed.
 */
public class StreamedProtoOutputFormatter extends ProtoOutputFormatter {
  private static final int OUTPUT_PARALLELISM = Runtime.getRuntime().availableProcessors();

  // At the issue's reported average target size, this produces roughly 0.5 MB chunks: large enough
  // to amortize task overhead without retaining the roughly 5 MB chunks used by the previous PR.
  @VisibleForTesting static final int TARGETS_PER_CHUNK = 50;

  // Two pending chunks per worker keep serialization busy while this callback writes the head
  // chunk. Retained serialized bytes are approximately MAX_PENDING_CHUNKS * TARGETS_PER_CHUNK *
  // average serialized target bytes, plus worker-local buffers and temporary array copies.
  private static final int MAX_PENDING_CHUNKS = 2 * OUTPUT_PARALLELISM;

  private static final ThreadFactory THREAD_FACTORY =
      Thread.ofPlatform().name("streamed-proto-output-", 0).factory();

  @Override
  public String getName() {
    return "streamed_proto";
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options, LabelPrinter labelPrinter) {
    // rule_class_info is emitted for the first rule in the stream with a given key. Parallel target
    // construction would race that stateful first-seen decision even though writes remain ordered.
    if (options.getParallelStreamedProtoOutput() && !options.getProtoRuleClasses()) {
      return new ParallelCallback(out, labelPrinter);
    }
    return new OutputFormatterCallback<Target>() {
      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        for (Target target : partialResult) {
          toTargetProtoBuffer(target, labelPrinter).writeDelimitedTo(out);
        }
      }
    };
  }

  /** Serializes bounded chunks in parallel and writes their bytes in input order. */
  private final class ParallelCallback extends OutputFormatterCallback<Target> {
    private final OutputStream out;
    private final LabelPrinter labelPrinter;
    private final ExecutorService executor =
        Executors.newFixedThreadPool(OUTPUT_PARALLELISM, THREAD_FACTORY);

    private ParallelCallback(OutputStream out, LabelPrinter labelPrinter) {
      this.out = out;
      this.labelPrinter = labelPrinter;
    }

    @Override
    public void processOutput(Iterable<Target> partialResult)
        throws IOException, InterruptedException {
      // Only this thread writes. Draining futures from the front preserves input order even when
      // later chunks finish first. The bounded deque also stops consuming input when output lags.
      Deque<Future<byte[]>> pendingChunks = new ArrayDeque<>();
      try {
        for (List<Target> targets : Iterables.partition(partialResult, TARGETS_PER_CHUNK)) {
          if (pendingChunks.size() == MAX_PENDING_CHUNKS) {
            writeChunk(pendingChunks.removeFirst());
          }
          pendingChunks.addLast(executor.submit(() -> serializeChunk(targets)));
        }
        while (!pendingChunks.isEmpty()) {
          writeChunk(pendingChunks.removeFirst());
        }
      } finally {
        pendingChunks.forEach(future -> future.cancel(/* mayInterruptIfRunning= */ true));
      }
    }

    private byte[] serializeChunk(List<Target> targets) throws IOException, InterruptedException {
      // Serialization is part of the measured bottleneck, and bytes retain less live object graph
      // than completed Build.Target protos while they wait for ordered output. Target proto sizes
      // vary substantially, so a workload-specific initial capacity could overallocate heavily.
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      for (Target target : targets) {
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }
        toTargetProtoBuffer(target, labelPrinter).writeDelimitedTo(bytes);
      }
      return bytes.toByteArray();
    }

    private void writeChunk(Future<byte[]> chunk) throws IOException, InterruptedException {
      out.write(
          MoreFutures.waitForFutureAndGetWithCheckedException(
              chunk, /* cancelOnInterrupt= */ true, IOException.class));
    }

    @Override
    public void close(boolean failFast) {
      // processOutput drains all work on success. On failure, cancel remaining work without
      // replacing the primary query, output, or interruption failure during cleanup.
      if (ExecutorUtil.uninterruptibleShutdownNow(executor)) {
        Thread.currentThread().interrupt();
      }
    }
  }
}
