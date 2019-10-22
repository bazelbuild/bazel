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
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

/**
 * Parallel file processing implementation.
 * See comment to {@link #processFile(Path, BlockParameters, AbstractDeclarationConsumerFactory,
 * ListeningExecutorService, SeparatorPredicate)}.
 */
public class ParallelFileProcessing<T extends DeclarationConsumer> implements Callable<Set<T>> {
  private final Path path;
  private final BlockParameters parameters;
  private final AbstractDeclarationConsumerFactory<T> declarationParserFactory;
  private final ListeningExecutorService executorService;
  private final SeparatorPredicate predicate;

  public ParallelFileProcessing(
      Path path,
      BlockParameters parameters,
      AbstractDeclarationConsumerFactory<T> declarationParserFactory,
      ListeningExecutorService executorService,
      SeparatorPredicate predicate) {
    this.path = path;
    this.parameters = parameters;
    this.declarationParserFactory = declarationParserFactory;
    this.executorService = executorService;
    this.predicate = predicate;
  }

  /**
   * Processes file in parallel: {@link java.nio.channels.FileChannel} is used to read contents into
   * a sequence of buffers. Each buffer is split into chunks, which are tokenized in parallel by
   * {@link BufferSplitter}, using the <code>predicate</code>. Fragments of tokens (on the bounds of
   * buffer fragments) are assembled by {@link DeclarationAssembler}. The resulting tokens (each can
   * be further parsed independently) are passed to {@link DeclarationConsumer}.
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
   * @param path to the file to be processed
   * @param parameters {@link BlockParameters} with sizes of read and tokenize blocks
   * @param tokenConsumerFactory factory of {@link DeclarationConsumer} for further processing /
   *     parsing
   * @param executorService executorService to use for parallel tokenization tasks
   * @param predicate predicate for separating tokens
   * @throws GenericParsingException thrown by further processing in <code>tokenConsumer</code>
   * @throws IOException thrown by file reading
   */
  public static <T extends DeclarationConsumer> void processFile(
      Path path,
      BlockParameters parameters,
      AbstractDeclarationConsumerFactory<T> tokenConsumerFactory,
      ListeningExecutorService executorService,
      SeparatorPredicate predicate)
      throws GenericParsingException, IOException, InterruptedException {
    new ParallelFileProcessing<T>(
            path, parameters, tokenConsumerFactory, executorService, predicate)
        .call();
  }

  @Override
  public Set<T> call() throws InterruptedException, IOException, GenericParsingException {
    try (ReadableByteChannel channel = path.createChannel()) {
      long fileSize = path.getFileSize();
      if (fileSize < parameters.getReadBlockSize()) {
        parameters.setReadBlockSize((int) fileSize);
      }
      DeclarationAssembler assembler =
          new DeclarationAssembler(declarationParserFactory.createParser(), predicate);

      CollectingListFuture<List<BufferEdge>, GenericParsingException> future =
          new CollectingListFuture<>(GenericParsingException.class);
      List<List<BufferEdge>> listOfLists;
      int offset = 0;
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
      List<BufferEdge> fragments =
          listOfLists.stream().flatMap(List::stream).collect(Collectors.toList());

      assembler.wrapUp(fragments);
      return declarationParserFactory.getParsers();
    }
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
      int offset,
      CollectingListFuture<List<BufferEdge>, GenericParsingException> future) {
    int from = 0;
    int blockSize = parameters.getTokenizeBlockSize();
    while (from < bb.limit()) {
      int to = Math.min(bb.limit(), from + blockSize);
      DeclarationConsumer consumer = declarationParserFactory.createParser();
      ByteBufferFragment fragment = new ByteBufferFragment(bb, from, to);
      BufferSplitter tokenizer = new BufferSplitter(fragment, consumer, predicate, offset);
      future.add(executorService.submit(tokenizer));
      from += blockSize;
    }
  }

  /** Sizes of blocks for reading from file and parsing for {@link ParallelFileProcessing}. */
  public static class BlockParameters {
    private static final int READ_BLOCK_SIZE = 25 * 1024 * 1024;
    private static final int MIN_READ_BLOCK_SIZE = 10 * 1024 * 1024;
    private static final int TOKENIZE_BLOCK_SIZE = 1024 * 1024;

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

    public BlockParameters() {
      readBlockSize = READ_BLOCK_SIZE;
      minReadBlockSize = MIN_READ_BLOCK_SIZE;
      tokenizeBlockSize = TOKENIZE_BLOCK_SIZE;
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
        tokenizeBlockSize = Math.min(tokenizeBlockSize, minReadBlockSize / 4);
      }
      return this;
    }

    public int getTokenizeBlockSize() {
      return tokenizeBlockSize;
    }

    /** Sets tokenizeBlockSize, if it is less than readBlockSize. */
    public BlockParameters setTokenizeBlockSize(int tokenizeBlockSize) {
      if (tokenizeBlockSize > 0 && tokenizeBlockSize <= readBlockSize) {
        this.tokenizeBlockSize = tokenizeBlockSize;
      }
      return this;
    }

    public int getMinReadBlockSize() {
      return minReadBlockSize;
    }
  }
}
