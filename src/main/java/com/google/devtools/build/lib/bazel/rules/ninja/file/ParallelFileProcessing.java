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

package com.google.devtools.build.lib.bazel.rules.ninja.file;

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Parallel file processing implementation.
 * See comment to {@link #processFile(Path, Supplier, BiPredicate, int, int)}.
 */
public class ParallelFileProcessing {
  private static final int BLOCK_SIZE = 25 * 1024 * 1024;
  private static final int MIN_BLOCK_SIZE = 10 * 1024 * 1024;
  /**
   * Number of threads to use in parallel tokenizing; during experiments, it did not help
   * to increase the number of threads above 25 even when the actual number of cores was
   * much higher: the limiting factor is probably merging of fragment bounds.
   */
  private static final int NUM_THREADS =
      Math.min(25, Runtime.getRuntime().availableProcessors() - 1);
  private final Path path;
  private final Supplier<TokenConsumer> tokenConsumerFactory;
  private final BiPredicate<Byte, Byte> predicate;
  private final int blockSize;
  private final int numThreads;
  private final int minBlockSize;

  private ParallelFileProcessing(Path path,
      Supplier<TokenConsumer> tokenConsumerFactory,
      BiPredicate<Byte, Byte> predicate,
      int blockSize,
      int numThreads) throws IOException {
    this.path = path;
    this.tokenConsumerFactory = tokenConsumerFactory;
    this.predicate = predicate;
    long size = Files.size(path);
    if (blockSize <= 0) {
      blockSize = (int) Math.min(BLOCK_SIZE, size);
    }
    this.blockSize = blockSize;
    minBlockSize = Math.min(blockSize, MIN_BLOCK_SIZE);
    if (numThreads <= 0) {
      numThreads = NUM_THREADS;
    }
    this.numThreads = numThreads;
  }

  /**
   * Processes file in parallel: {@link java.nio.channels.FileChannel} is used to read
   * contents into a sequence of buffers.
   * Each buffer is split into chunks, which are tokenized in parallel by
   * {@link BufferTokenizer}, using the <code>predicate</code>.
   * Fragments of tokens (on the bounds of buffer fragments) are assembled by
   * {@link TokenAssembler}.
   * The resulting tokens (each can be further parsed independently) are passed to
   * {@link TokenConsumer}.
   *
   * The main ideas behind this implementation are:
   *
   * 1) using the NIO with direct buffer allocation for reading from file,
   * (a quote from ByteBuffer's javadoc:
   * "Given a direct byte buffer, the Java virtual machine will make a best effort to perform native
   * I/O operations directly upon it. That is, it will attempt to avoid copying the buffer's content
   * to (or from) an intermediate buffer before (or after) each invocation of one of the underlying
   * operating system's native I/O operations.")
   *
   * 2) utilizing the possibilities for parallel processing, since splitting into tokens and
   * parsing them can be done in high degree independently.
   *
   * 3) not creating additional copies of character buffers for keeping tokens and
   * only then specific objects.
   *
   * 4) for absorbing the results, it is possible to create a consumer for each tokenizer,
   * and escape synchronization, summarizing the results after all parallel work is done,
   * of use just one shared consumer with synchronized data structures.
   *
   * Please see a comment about performance test results:
   * {@link com.google.devtools.build.lib.bazel.rules.ninja.ParallelFileProcessingTest}.
   *
   * @param path path to file to process
   * @param tokenConsumerFactory factory of {@link TokenConsumer} for further processing / parsing
   * @param predicate predicate for separating tokens
   * @param blockSize size of the buffer for reading from channel, -1 for using the default value.
   * @param numThreads number of threads for parallel tokenizing, -1 for default value.
   * @throws GenericParsingException thrown by further processing in <code>tokenConsumer</code>
   * @throws IOException thrown by file reading
   */
  public static void processFile(Path path,
      Supplier<TokenConsumer> tokenConsumerFactory,
      BiPredicate<Byte, Byte> predicate,
      int blockSize,
      int numThreads)
      throws GenericParsingException, IOException, ExecutionException, InterruptedException {
    new ParallelFileProcessing(path, tokenConsumerFactory, predicate, blockSize, numThreads)
        .processFileImpl();
  }

  private void processFileImpl()
      throws InterruptedException, ExecutionException, IOException, GenericParsingException {
    TokenAssembler assembler = new TokenAssembler(tokenConsumerFactory.get(), predicate);

    List<Pair<Integer, ByteBufferFragment>> fragments;
    try (SeekableByteChannel ch = Files.newByteChannel(path);
        ExecutorHelper service = new ExecutorHelper(numThreads)) {
      int offset = 0;
      boolean keepReading = true;
      while (keepReading && ch.isOpen()) {
        ByteBuffer bb = ByteBuffer.allocateDirect(blockSize);
        keepReading = readFromChannel(ch, bb);
        if (bb.position() > 0) {
          bb.flip();
          tokenizeFragments(bb.asReadOnlyBuffer(), offset, service);
          offset += bb.limit();
        }
      }
      fragments = service.get();
    }

    assembler.wrapUp(fragments);
  }

  private boolean readFromChannel(SeekableByteChannel ch, ByteBuffer bb) throws IOException {
    // Continue reading until we filled the minimum buffer size.
    while (ch.isOpen() && bb.position() < minBlockSize) {
      // Stop if we reached the end of stream.
      if (ch.read(bb) < 0) {
        return false;
      }
    }
    return true;
  }

  private void tokenizeFragments(ByteBuffer bb, int offset, ExecutorHelper service) {
    int fragmentSize = (int) Math.ceil((double) bb.limit() / numThreads);
    int from = 0;
    while (from < bb.limit()) {
      int to = Math.min(bb.limit(), from + fragmentSize);
      TokenConsumer consumer = tokenConsumerFactory.get();
      service.accept(new BufferTokenizer(bb, consumer, predicate, offset, from, to));
      from += fragmentSize;
    }
  }

  private static class ExecutorHelper
      implements Consumer<Callable<List<Pair<Integer, ByteBufferFragment>>>>, AutoCloseable {
    private final List<ListenableFuture<List<Pair<Integer, ByteBufferFragment>>>> futures;
    private final ListeningExecutorService executorService;

    private ExecutorHelper(int numThreads) {
      // A separate pool works better then ForkJoinTask.commonPool().
      executorService = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads,
          new ThreadFactoryBuilder().setNameFormat("file-indexer-thread-%d").build()));
      futures = Lists.newArrayList();
    }

    @Override
    public void accept(Callable<List<Pair<Integer, ByteBufferFragment>>> callable) {
      futures.add(executorService.submit(callable));
    }

    public List<Pair<Integer, ByteBufferFragment>> get()
        throws ExecutionException, InterruptedException {
      List<List<Pair<Integer, ByteBufferFragment>>> listOfLists = Futures.allAsList(futures).get();
      return listOfLists.stream()
          .flatMap(List::stream)
          .collect(Collectors.toList());
    }

    @Override
    public void close() {
      ExecutorUtil.interruptibleShutdown(executorService);
    }
  }
}
