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
package com.google.devtools.build.android.r8;

import static com.google.common.base.Verify.verify;
import static java.lang.Math.min;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.android.tools.r8.ByteDataView;
import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.CompilationMode;
import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.DexIndexedConsumer;
import com.android.tools.r8.DiagnosticsHandler;
import com.android.tools.r8.SyntheticInfoConsumer;
import com.android.tools.r8.SyntheticInfoConsumerData;
import com.android.tools.r8.origin.ArchiveEntryOrigin;
import com.android.tools.r8.origin.PathOrigin;
import com.android.tools.r8.references.ClassReference;
import com.google.auto.value.AutoValue;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.Weigher;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.worker.ProtoWorkerMessageProcessor;
import com.google.devtools.build.lib.worker.WorkRequestHandler;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import javax.annotation.Nullable;

/**
 * Tool used by Bazel that converts a Jar file of .class files into a .zip file of .dex files, one
 * per .class file, which we call a <i>dex archive</i>.
 *
 * <p>D8 version of DexBuilder.
 */
public class CompatDexBuilder {
  private static final long ONE_MEG = 1024 * 1024;

  private static class ContextConsumer implements SyntheticInfoConsumer {

    // After compilation this will be non-null iff the compiled class is a D8 synthesized class.
    ClassReference sythesizedPrimaryClass = null;

    // If the above is non-null then this will be the non-synthesized context class that caused
    // D8 to synthesize the above class.
    ClassReference contextOfSynthesizedClass = null;

    @Nullable
    String getContextMapping() {
      if (sythesizedPrimaryClass != null) {
        return sythesizedPrimaryClass.getBinaryName()
            + ";"
            + contextOfSynthesizedClass.getBinaryName();
      }
      return null;
    }

    @Override
    public synchronized void acceptSyntheticInfo(SyntheticInfoConsumerData data) {
      verify(
          sythesizedPrimaryClass == null || sythesizedPrimaryClass.equals(data.getSyntheticClass()),
          "The single input classfile should ensure this has one value.");
      verify(
          contextOfSynthesizedClass == null
              || contextOfSynthesizedClass.equals(data.getSynthesizingContextClass()),
          "The single input classfile should ensure this has one value.");
      sythesizedPrimaryClass = data.getSyntheticClass();
      contextOfSynthesizedClass = data.getSynthesizingContextClass();
    }

    @Override
    public void finished() {
      // Do nothing.
    }
  }

  private static class DexConsumer implements DexIndexedConsumer {

    final ContextConsumer contextConsumer = new ContextConsumer();
    byte[] bytes;

    @Override
    public synchronized void accept(
        int fileIndex, ByteDataView data, Set<String> descriptors, DiagnosticsHandler handler) {
      verify(bytes == null, "Should not have been populated until now");
      bytes = data.copyByteData();
    }

    byte[] getBytes() {
      return bytes;
    }

    void setBytes(byte[] byteCode) {
      this.bytes = byteCode;
    }

    ContextConsumer getContextConsumer() {
      return contextConsumer;
    }

    @Override
    public void finished(DiagnosticsHandler handler) {
      // Do nothing.
    }
  }

  public static void main(String[] args)
      throws IOException, InterruptedException, ExecutionException, OptionsParsingException {
    CompatDexBuilder compatDexBuilder = new CompatDexBuilder();
    if (ImmutableSet.copyOf(args).contains("--persistent_worker")) {
      ByteArrayOutputStream buf = new ByteArrayOutputStream();
      PrintStream ps = new PrintStream(buf, true);
      PrintStream realStdOut = System.out;
      PrintStream realStdErr = System.err;

      // Redirect all stdout and stderr output for logging.
      System.setOut(ps);
      System.setErr(ps);

      // Set up dexer cache
      Cache<DexingKeyR8, byte[]> dexCache =
          CacheBuilder.newBuilder()
              // Use at most 200 MB for cache and leave at least 25 MB of heap space alone. For
              // reference:
              // .class & class.dex files are around 1-5 KB, so this fits ~30K-35K class-dex pairs.
              .maximumWeight(min(Runtime.getRuntime().maxMemory() - 25 * ONE_MEG, 200 * ONE_MEG))
              .weigher(
                  new Weigher<DexingKeyR8, byte[]>() {
                    @Override
                    public int weigh(DexingKeyR8 key, byte[] value) {
                      return key.classfileContent().length + value.length;
                    }
                  })
              .build();
      try {
        WorkRequestHandler workerHandler =
            new WorkRequestHandler.WorkRequestHandlerBuilder(
                    new WorkRequestHandler.WorkRequestCallback(
                        (request, pw) ->
                            compatDexBuilder.processRequest(
                                dexCache, request.getArgumentsList(), pw, buf)),
                    realStdErr,
                    new ProtoWorkerMessageProcessor(System.in, realStdOut))
                .setCpuUsageBeforeGc(Duration.ofSeconds(10))
                .build();
        workerHandler.processRequests();
      } catch (IOException e) {
        realStdErr.println(e.getMessage());
        System.exit(1);
      } finally {
        System.setOut(realStdOut);
        System.setErr(realStdErr);
      }
    } else {
      // Dex cache has no value in non-persistent mode, so pass it as null.
      compatDexBuilder.dexEntries(/* dexCache= */ null, Arrays.asList(args));
    }
  }

  private int processRequest(
      @Nullable Cache<DexingKeyR8, byte[]> dexCache,
      List<String> args,
      PrintWriter pw,
      ByteArrayOutputStream buf) {
    try {
      dexEntries(dexCache, args);
      return 0;
    } catch (OptionsParsingException e) {
      pw.println("CompatDexBuilder raised OptionsParsingException: " + e.getMessage());
      return 1;
    } catch (IOException | InterruptedException | ExecutionException e) {
      e.printStackTrace();
      return 1;
    } finally {
      // Write the captured buffer to the work response
      synchronized (buf) {
        String captured = buf.toString(UTF_8).trim();
        buf.reset();
        pw.print(captured);
      }
    }
  }

  @SuppressWarnings("JdkObsolete")
  private void dexEntries(@Nullable Cache<DexingKeyR8, byte[]> dexCache, List<String> args)
      throws IOException, InterruptedException, ExecutionException, OptionsParsingException {
    List<String> flags = new ArrayList<>();
    String input = null;
    String output = null;
    int minSdkVersionFlag = Constants.MIN_API_LEVEL;
    int numberOfThreads = min(8, Runtime.getRuntime().availableProcessors());
    boolean noLocals = false;

    for (String arg : args) {
      if (arg.startsWith("@")) {
        flags.addAll(Files.readAllLines(Paths.get(arg.substring(1))));
      } else {
        flags.add(arg);
      }
    }

    for (int i = 0; i < flags.size(); i++) {
      String flag = flags.get(i);
      if (flag.startsWith("--positions=")) {
        String positionsValue = flag.substring("--positions=".length());
        if (positionsValue.startsWith("throwing") || positionsValue.startsWith("important")) {
          noLocals = true;
        }
        continue;
      }
      if (flag.startsWith("--num-threads=")) {
        numberOfThreads = Integer.parseInt(flag.substring("--num-threads=".length()));
        continue;
      }
      switch (flag) {
        case "--input_jar":
          input = flags.get(++i);
          break;
        case "--output_zip":
          output = flags.get(++i);
          break;
        case "--verify-dex-file":
        case "--no-verify-dex-file":
        case "--show_flags":
        case "--no-optimize":
        case "--nooptimize":
        case "--help":
          // Ignore
          break;
        case "--nolocals":
          noLocals = true;
          break;
        case "--min_sdk_version":
          minSdkVersionFlag = Integer.parseInt(flags.get(++i));
          break;
        default:
          throw new OptionsParsingException("Unsupported option: " + flag);
      }
    }

    if (input == null) {
      throw new OptionsParsingException("No input jar specified");
    }

    if (output == null) {
      throw new OptionsParsingException("No output jar specified");
    }

    ExecutorService executor = Executors.newWorkStealingPool(numberOfThreads);
    try (ZipOutputStream out =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(output))))) {

      List<ZipEntry> toDex = new ArrayList<>();

      try (ZipFile zipFile = new ZipFile(input, UTF_8)) {
        final CompilationMode compilationMode =
            noLocals ? CompilationMode.RELEASE : CompilationMode.DEBUG;
        final Enumeration<? extends ZipEntry> entries = zipFile.entries();
        while (entries.hasMoreElements()) {
          ZipEntry entry = entries.nextElement();
          if (!entry.getName().endsWith(".class")) {
            try (InputStream stream = zipFile.getInputStream(entry)) {
              ZipUtils.addEntry(entry.getName(), stream, out);
            }
          } else {
            toDex.add(entry);
          }
        }

        final int minSdkVersion = minSdkVersionFlag;
        List<Future<DexConsumer>> futures = new ArrayList<>(toDex.size());
        for (ZipEntry classEntry : toDex) {
          futures.add(
              executor.submit(
                  () ->
                      dexEntry(
                          dexCache,
                          zipFile,
                          classEntry,
                          compilationMode,
                          minSdkVersion,
                          executor)));
        }
        StringBuilder contextMappingBuilder = new StringBuilder();
        for (int i = 0; i < futures.size(); i++) {
          ZipEntry entry = toDex.get(i);
          DexConsumer consumer = futures.get(i).get();
          ZipUtils.addEntry(entry.getName() + ".dex", consumer.getBytes(), ZipEntry.STORED, out);
          String mapping = consumer.getContextConsumer().getContextMapping();
          if (mapping != null) {
            contextMappingBuilder.append(mapping).append('\n');
          }
        }
        String contextMapping = contextMappingBuilder.toString();
        if (!contextMapping.isEmpty()) {
          ZipUtils.addEntry(
              "META-INF/synthetic-contexts.map",
              contextMapping.getBytes(UTF_8),
              ZipEntry.STORED,
              out);
        }
      }
    } finally {
      executor.shutdown();
    }
  }

  private DexConsumer dexEntry(
      @Nullable Cache<DexingKeyR8, byte[]> dexCache,
      ZipFile zipFile,
      ZipEntry classEntry,
      CompilationMode mode,
      int minSdkVersion,
      ExecutorService executor)
      throws IOException, CompilationFailedException {
    DexConsumer consumer = new DexConsumer();
    D8Command.Builder builder = D8Command.builder();
    builder
        .setProgramConsumer(consumer)
        .setSyntheticInfoConsumer(consumer.getContextConsumer())
        .setMode(mode)
        .setMinApiLevel(minSdkVersion)
        .setDisableDesugaring(true)
        .setIntermediate(true);
    byte[] cachedDexBytes = null;
    byte[] classFileBytes = null;
    try (InputStream stream = zipFile.getInputStream(classEntry)) {
      classFileBytes = ByteStreams.toByteArray(stream);
      if (dexCache != null) {
        // If the cache exists, check for cache validity.
        cachedDexBytes =
            dexCache.getIfPresent(DexingKeyR8.create(mode, minSdkVersion, classFileBytes));
      }
      if (cachedDexBytes != null) {
        // Cache hit: quit early and return the data
        consumer.setBytes(cachedDexBytes);
        return consumer;
      }
      builder.addClassProgramData(
          classFileBytes,
          new ArchiveEntryOrigin(
              classEntry.getName(), new PathOrigin(Paths.get(zipFile.getName()))));
    }
    D8.run(builder.build(), executor);
    // After dexing finishes, store the dexed output into the cache.
    if (dexCache != null) {
      dexCache.put(DexingKeyR8.create(mode, minSdkVersion, classFileBytes), consumer.getBytes());
    }
    return consumer;
  }

  /**
   * Generates a unique key for the R8 dex builder (CompatDexBuilder) as a key for the dex builder's
   * runtime cache.
   */
  @AutoValue
  public abstract static class DexingKeyR8 {
    public static DexingKeyR8 create(
        CompilationMode compilationMode, int minSdkVersion, byte[] classfileContent) {
      return new AutoValue_CompatDexBuilder_DexingKeyR8(
          compilationMode, minSdkVersion, classfileContent);
    }

    public abstract CompilationMode compilationMode();

    public abstract int minSdkVersion();

    @SuppressWarnings("mutable")
    public abstract byte[] classfileContent();
  }
}

