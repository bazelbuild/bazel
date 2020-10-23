// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkArgument;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.concurrent.Executors.newFixedThreadPool;

import com.android.dx.command.dexer.DxContext;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.Weigher;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.dexer.Dexing.DexingKey;
import com.google.devtools.build.android.dexer.Dexing.DexingOptions;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import javax.annotation.Nullable;

/**
 * Tool used by Bazel that converts a Jar file of .class files into a .zip file of .dex files,
 * one per .class file, which we call a <i>dex archive</i>.
 */
class DexBuilder {

  private static final long ONE_MEG = 1_000_000L;

  /**
   * Commandline options.
   */
  public static class Options extends OptionsBase {
    @Option(
      name = "input_jar",
      defaultValue = "null",
      category = "input",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help = "Input file to read classes and jars from."
    )
    public Path inputJar;

    @Option(
      name = "output_zip",
      defaultValue = "null",
      category = "output",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = PathConverter.class,
      abbrev = 'o',
      help = "Output file to write."
    )
    public Path outputZip;

    @Option(
      name = "max_threads",
      defaultValue = "8",
      category = "misc",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "How many threads (besides the main thread) to use at most."
    )
    public int maxThreads;

    @Option(
      name = "persistent_worker",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.HIDDEN},
      help = "Run as a Bazel persistent worker."
    )
    public boolean persistentWorker;
  }

  public static void main(String[] args) throws Exception {
    if (args.length == 1 && args[0].startsWith("@")) {
      args = Files.readAllLines(Paths.get(args[0].substring(1)), ISO_8859_1).toArray(new String[0]);
    }

    OptionsParser optionsParser =
        OptionsParser.builder().optionsClasses(Options.class, DexingOptions.class).build();
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);
    if (options.persistentWorker) {
      runPersistentWorker();
    } else {
      buildDexArchive(options, new Dexing(optionsParser.getOptions(DexingOptions.class)));
    }
  }

  @VisibleForTesting
  static void buildDexArchive(Options options, Dexing dexing) throws Exception {
    checkArgument(options.maxThreads > 0,
        "--max_threads must be strictly positive, was: %s", options.maxThreads);
    try (ZipFile in = new ZipFile(options.inputJar.toFile())) {
      // Heuristic: use at most 1 thread per 1000 files in the input Jar
      int threads = Math.min(options.maxThreads, in.size() / 1000 + 1);
      ExecutorService executor = newFixedThreadPool(threads);
      try (ZipOutputStream out = createZipOutputStream(options.outputZip)) {
        produceDexArchive(in, out, executor, threads <= 1, dexing, null);
      } finally {
        executor.shutdown();
      }
    }
  }

  /**
   * Implements a persistent worker process for use with Bazel (see {@code WorkerSpawnStrategy}).
   */
  private static void runPersistentWorker() throws IOException {
    ExecutorService executor = newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    Cache<DexingKey, byte[]> dexCache = CacheBuilder.newBuilder()
        // Use at most 200 MB for cache and leave at least 25 MB of heap space alone. For reference:
        // .class & class.dex files are around 1-5 KB, so this fits ~30K-35K class-dex pairs.
        .maximumWeight(Math.min(Runtime.getRuntime().maxMemory() - 25 * ONE_MEG, 200 * ONE_MEG))
        .weigher(new Weigher<DexingKey, byte[]>() {
          @Override
          public int weigh(DexingKey key, byte[] value) {
            return key.classfileContent().length + value.length;
          }
        })
        .build();
    try {
      while (true) {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);
        if (request == null) {
          return;
        }

        // Redirect dx's output so we can return it in response
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos, /*autoFlush*/ true);
        DxContext context = new DxContext(ps, ps);
        // Make sure that we exit nonzero in case uncaught errors occur during processRequest.
        int exitCode = 1;
        try {
          processRequest(executor, dexCache, context, request.getArgumentsList());
          exitCode = 0; // success!
        } catch (Exception e) {
          // Deliberate catch-all so we can capture a stack trace.
          // TODO(bazel-team): Consider canceling any outstanding futures created for this request
          e.printStackTrace(ps);
        } catch (Error e) {
          e.printStackTrace();
          e.printStackTrace(ps); // try capturing the error, may fail if out of memory
          throw e; // rethrow to kill the worker
        } finally {
          // Try sending a response no matter what
          String output;
          try {
            output = baos.toString();
          } catch (Throwable t) { // most likely out of memory, so log with minimal memory needs
            t.printStackTrace();
            output = "check worker log for exceptions";
          }
          WorkResponse.newBuilder()
              .setOutput(output)
              .setExitCode(exitCode)
              .setRequestId(request.getRequestId())
              .build()
              .writeDelimitedTo(System.out);
          System.out.flush();
        }
      }
    } finally {
      executor.shutdown();
    }
  }

  private static void processRequest(
      ExecutorService executor,
      Cache<DexingKey, byte[]> dexCache,
      DxContext context,
      List<String> args)
      throws OptionsParsingException, IOException, InterruptedException, ExecutionException {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class, DexingOptions.class)
            .allowResidue(false)
            .build();
    optionsParser.parse(args);
    Options options = optionsParser.getOptions(Options.class);
    try (ZipFile in = new ZipFile(options.inputJar.toFile());
        ZipOutputStream out = createZipOutputStream(options.outputZip)) {
      produceDexArchive(
          in,
          out,
          executor,
          /*convertOnReaderThread*/ false,
          new Dexing(context, optionsParser.getOptions(DexingOptions.class)),
          dexCache);
    }
  }

  private static ZipOutputStream createZipOutputStream(Path path) throws IOException {
    return new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(path)));
  }

  private static void produceDexArchive(
      ZipFile in,
      ZipOutputStream out,
      ExecutorService executor,
      boolean convertOnReaderThread,
      Dexing dexing,
      @Nullable Cache<DexingKey, byte[]> dexCache)
      throws InterruptedException, ExecutionException, IOException {
    // If we only have one thread in executor, we give a "direct" executor to the stuffer, which
    // will convert .class files to .dex inline on the same thread that reads the input jar.
    // This is an optimization that makes sure we can start writing the output file below while
    // the stuffer is still working its way through the input.
    DexConversionEnqueuer enqueuer = new DexConversionEnqueuer(in,
        convertOnReaderThread ? MoreExecutors.newDirectExecutorService() : executor,
        new DexConverter(dexing),
        dexCache);
    Future<?> enqueuerTask = executor.submit(enqueuer);
    while (true) {
      // Wait for next future in the queue *and* for that future to finish.  To guarantee
      // deterministic output we just write out the files in the order they appear, which is
      // the same order as in the input zip.
      ZipEntryContent file = enqueuer.getFiles().take().get();
      if (file == null) {
        // "done" marker indicating no more files coming.
        // Make sure enqueuer terminates normally (any wait should be minimal).  This in
        // particular surfaces any exceptions thrown in the enqueuer.
        enqueuerTask.get();
        break;
      }
      out.putNextEntry(file.getEntry());
      out.write(file.getContent());
      out.closeEntry();
    }
  }

  private DexBuilder() {
  }
}
