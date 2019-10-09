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

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Parallel file processing implementation.
 * See comment to {@link #processFile(Path, TokenConsumer, SeparatorPredicate, int, int)}.
 */
public class ParallelFileProcessing {
  private static final int BLOCK_SIZE = 30 * 1024 * 1024;
  private static final int NUM_THREADS = 4;

  private ParallelFileProcessing() {
  }

  /**
   * Processes file in parallel: {@link java.nio.channels.FileChannel} is used to read
   * contents into a sequence of buffers.
   * Each buffer is split into chunks, which are tokenized in parallel by
   * {@link BufferTokenizer}, using the {@link SeparatorPredicate}.
   * Fragments of tokens (on the bounds of buffer fragments) are assembled by
   * {@link TokenAssembler}.
   * The resulting tokens (each can be further parsed independently) are passed to
   * {@link TokenConsumer}.
   *
   * The main ideas behind this implementation is:
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
   * Please see a comment about performance test results:
   * {@link com.google.devtools.common.options.ParamsFilePreProcessorTest}.
   *
   * @param path path to file to process
   * @param tokenConsumer token consumer for further processing / parsing
   * @param predicate predicate for separating tokens
   * @param blockSize size of the buffer for reading from channel, -1 for using the default value.
   * @param numThreads number of threads for parallel tokenizing, -1 for default value.
   * @throws GenericParsingException thrown by further processing in <code>tokenConsumer</code>
   * @throws IOException thrown by file reading
   */
  public static void processFile(Path path,
      TokenConsumer tokenConsumer,
      SeparatorPredicate predicate,
      int blockSize,
      int numThreads) throws GenericParsingException, IOException {
    TokenAssembler assembler = new TokenAssembler(tokenConsumer, predicate);
    BufferTokenizerFactory factory = (buffer, offset, start, end) ->
        new BufferTokenizer(buffer, assembler, predicate, offset, start, end);

    readAndTokenize(path, factory, blockSize, numThreads);

    assembler.wrapUp();
  }

  private static void readAndTokenize(Path path,
      BufferTokenizerFactory tokenizerFactory,
      int blockSize,
      int numThreads) throws IOException {
    if (blockSize < 0) {
      blockSize = BLOCK_SIZE;
    }
    if (numThreads < 0) {
      numThreads = NUM_THREADS;
    }

    ExecutorService executorService = Executors.newFixedThreadPool(numThreads,
        new ThreadFactoryBuilder().setNameFormat("file-indexer-thread-%d").build());
    try (SeekableByteChannel ch = Files.newByteChannel(path)) {
      int numRead = 1;
      int offset = 0;
      while (numRead >= 0 && ch.isOpen()) {
        ByteBuffer bb = ByteBuffer.allocateDirect(blockSize);
        numRead = ch.read(bb);
        if (numRead > 0) {
          bb.limit(bb.position());
          bb.position(0);
          CharBuffer charBuffer = StandardCharsets.ISO_8859_1.decode(bb);
          Preconditions.checkArgument(charBuffer.hasArray());
          tokenize(offset, charBuffer.array(), executorService, tokenizerFactory, numThreads);
          offset += charBuffer.length();
        }
      }
    } finally {
      ExecutorUtil.interruptibleShutdown(executorService);
    }
  }

  private static void tokenize(int offset, char[] buffer, ExecutorService service,
      BufferTokenizerFactory factory, int numThreads) {
    int size = buffer.length / numThreads;
    for (int index = 0; index < numThreads; index++) {
      int from = index * size;
      int to = (index == numThreads - 1) ? buffer.length : (from + size);
      service.submit(factory.create(buffer, offset, from, to));
    }
  }
}
