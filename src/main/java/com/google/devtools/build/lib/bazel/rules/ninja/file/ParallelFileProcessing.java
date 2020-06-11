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

import com.google.common.util.concurrent.ListeningExecutorService;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Parallel file processing implementation. See comment to {@link #processFile(ReadableByteChannel,
 * BlockParameters, Supplier, ListeningExecutorService)}.
 */
public class ParallelFileProcessing {
  private final ReadableByteChannel channel;
  private final BlockParameters parameters;
  private final Supplier<DeclarationConsumer> tokenConsumerFactory;
  private final ListeningExecutorService executorService;

  private ParallelFileProcessing(
      ReadableByteChannel channel,
      BlockParameters parameters,
      Supplier<DeclarationConsumer> tokenConsumerFactory,
      ListeningExecutorService executorService) {
    this.channel = channel;
    this.parameters = parameters;
    this.tokenConsumerFactory = tokenConsumerFactory;
    this.executorService = executorService;
  }

  /**
   * Processes file in parallel: {@link java.nio.channels.FileChannel} is used to read contents into
   * a sequence of buffers. Each buffer is split into chunks, which are tokenized in parallel by
   * {@link FileFragmentSplitter}, using the <code>predicate</code>. Fragments of tokens (on the
   * bounds of buffer fragments) are assembled by {@link DeclarationAssembler}. The resulting tokens
   * (each can be further parsed independently) are passed to {@link DeclarationConsumer}.
   *
   * <p>The main ideas behind this implementation are:
   *
   * <p>1) using the NIO with direct buffer allocation for reading from file, (a quote from
   * ByteBuffer's javadoc: "Given a direct byte buffer, the Java virtual machine will make a best
   * effort to perform native I/O operations directly upon it. That is, it will attempt to avoid
   * copying the buffer's content to (or from) an intermediate buffer before (or after) each
   * invocation of one of the underlying operating system's native I/O operations.")
   *
   * <p>2) utilizing the possibilities for parallel processing, since splitting into tokens and
   * parsing them can be done in high degree independently.
   *
   * <p>3) not creating additional copies of character buffers for keeping tokens and only then
   * specific objects.
   *
   * <p>4) for absorbing the results, it is possible to create a consumer for each tokenizer, and
   * escape synchronization, summarizing the results after all parallel work is done, of use just
   * one shared consumer with synchronized data structures.
   *
   * <p>Please see a comment about performance test results: {@link
   * com.google.devtools.build.lib.bazel.rules.ninja.ParallelFileProcessingTest}.
   *
   * @param channel open {@link ReadableByteChannel} to file to process. The lifetime of the channel
   *     is outside the scope of this method. Channel should not be closed by this method.
   * @param parameters {@link BlockParameters} with sizes of read and tokenize blocks
   * @param tokenConsumerFactory factory of {@link DeclarationConsumer} for further processing /
   *     parsing
   * @param executorService executorService to use for parallel tokenization tasks
   * @param predicate predicate for separating tokens
   * @throws GenericParsingException thrown by further processing in <code>tokenConsumer</code>
   * @throws IOException thrown by file reading
   */
  public static void processFile(
      ReadableByteChannel channel,
      BlockParameters parameters,
      Supplier<DeclarationConsumer> tokenConsumerFactory,
      ListeningExecutorService executorService)
      throws GenericParsingException, IOException, InterruptedException {
    new ParallelFileProcessing(channel, parameters, tokenConsumerFactory, executorService)
        .processFileImpl();
  }

  private void processFileImpl() throws InterruptedException, IOException, GenericParsingException {
    if (parameters.readBlockSize == 0) {
      // Return immediately, if the file is empty.
      return;
    }
    DeclarationAssembler assembler = new DeclarationAssembler(tokenConsumerFactory.get());

    CollectingListFuture<List<FileFragment>, GenericParsingException> future =
        new CollectingListFuture<>(GenericParsingException.class);
    List<List<FileFragment>> listOfLists;
    long offset = 0;
    boolean keepReading = true;
    while (keepReading) {
      ByteBuffer bb = ByteBuffer.allocateDirect(parameters.getReadBlockSize());
      keepReading = readFromChannel(channel, bb);
      if (bb.position() > 0) {
        bb.flip();
        tokenizeFragments(bb.asReadOnlyBuffer(), offset, future);
        offset += bb.limit();
      }
    }
    listOfLists = future.getResult();
    List<FileFragment> fragments =
        listOfLists.stream().flatMap(List::stream).collect(Collectors.toList());

    assembler.wrapUp(fragments);
  }

  private boolean readFromChannel(ReadableByteChannel ch, ByteBuffer bb) throws IOException {
    // Continue reading until we filled the minimum buffer size.
    while (bb.position() < parameters.getMinReadBlockSize()) {
      // Stop if we reached the end of stream.
      if (ch.read(bb) < 0) {
        return false;
      }
    }
    return true;
  }

  private void tokenizeFragments(
      ByteBuffer bb,
      long offset,
      CollectingListFuture<List<FileFragment>, GenericParsingException> future) {
    int from = 0;
    int blockSize = parameters.getTokenizeBlockSize();
    while (from < bb.limit()) {
      int to = Math.min(bb.limit(), from + blockSize);
      if (bb.limit() - to < BlockParameters.MIN_TOKENIZE_BLOCK_SIZE) {
        // Do not create the last block too small, rather join it with the previous block.
        to = bb.limit();
      }
      DeclarationConsumer consumer = tokenConsumerFactory.get();
      FileFragment fragment = new FileFragment(bb, offset, from, to);
      FileFragmentSplitter tokenizer = new FileFragmentSplitter(fragment, consumer);
      future.add(executorService.submit(tokenizer));
      from = to;
    }
  }

  /** Sizes of blocks for reading from file and parsing for {@link ParallelFileProcessing}. */
  public static class BlockParameters {

    private static final int READ_BLOCK_SIZE = 25 * 1024 * 1024;
    private static final int MIN_READ_BLOCK_SIZE = 10 * 1024 * 1024;
    private static final int TOKENIZE_BLOCK_SIZE = 1024 * 1024;
    private static final int MIN_TOKENIZE_BLOCK_SIZE = 100;

    /** Size of the reading buffer. */
    private int readBlockSize;
    /**
     * Minimum size of the reading buffer - read() calls will be repeated until the reading buffer
     * has at least minReadBlockSize bytes, only after that the contents will be passed for
     * tokenization.
     */
    private int minReadBlockSize;
    /**
     * Size of the piece for the tokenization task. The read buffer will be split into pieces of
     * tokenizeBlockSize size, and passed for tokenization in parallel.
     */
    private int tokenizeBlockSize;

    /** @param fileSize size of the file we are going to parse */
    public BlockParameters(long fileSize) {
      readBlockSize = (int) Math.min(READ_BLOCK_SIZE, fileSize);
      minReadBlockSize = Math.min(MIN_READ_BLOCK_SIZE, (int) Math.ceil((double) fileSize / 2));
      tokenizeBlockSize =
          Math.max(MIN_TOKENIZE_BLOCK_SIZE, Math.min(TOKENIZE_BLOCK_SIZE, minReadBlockSize / 4));
    }

    public int getReadBlockSize() {
      return readBlockSize;
    }

    /**
     * Sets the size of readBlockSize and adjusts other block sizes so that they together make
     * sense.
     */
    public BlockParameters setReadBlockSize(int readBlockSize) {
      if (readBlockSize > 0) {
        this.readBlockSize = readBlockSize;
        minReadBlockSize = Math.min(minReadBlockSize, (int) Math.ceil((double) readBlockSize / 2));
        tokenizeBlockSize =
            Math.max(MIN_TOKENIZE_BLOCK_SIZE, Math.min(tokenizeBlockSize, minReadBlockSize / 4));
      }
      return this;
    }

    public int getTokenizeBlockSize() {
      return tokenizeBlockSize;
    }

    public int getMinReadBlockSize() {
      return minReadBlockSize;
    }
  }
}
