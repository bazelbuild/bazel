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
import com.android.tools.r8.origin.ArchiveEntryOrigin;
import com.android.tools.r8.origin.PathOrigin;
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

/**
 * Tool used by Bazel that converts a Jar file of .class files into a .zip file of .dex files, one
 * per .class file, which we call a <i>dex archive</i>.
 *
 * <p>D8 version of DexBuilder.
 */
public class CompatDexBuilder {

  private static class DexConsumer implements DexIndexedConsumer {

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
      try {
        WorkRequestHandler workerHandler =
            new WorkRequestHandler.WorkRequestHandlerBuilder(
                    new WorkRequestHandler.WorkRequestCallback(
                        (request, pw) ->
                            compatDexBuilder.processRequest(request.getArgumentsList(), pw, buf)),
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
      compatDexBuilder.dexEntries(Arrays.asList(args));
    }
  }

  private int processRequest(List<String> args, PrintWriter pw, ByteArrayOutputStream buf) {
    try {
      dexEntries(args);
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
  private void dexEntries(List<String> args)
      throws IOException, InterruptedException, ExecutionException, OptionsParsingException {
    List<String> flags = new ArrayList<>();
    String input = null;
    String output = null;
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

        List<Future<DexConsumer>> futures = new ArrayList<>(toDex.size());
        for (ZipEntry classEntry : toDex) {
          futures.add(
              executor.submit(() -> dexEntry(zipFile, classEntry, compilationMode, executor)));
        }
        for (int i = 0; i < futures.size(); i++) {
          ZipEntry entry = toDex.get(i);
          DexConsumer consumer = futures.get(i).get();
          ZipUtils.addEntry(entry.getName() + ".dex", consumer.getBytes(), ZipEntry.STORED, out);
        }
      }
    } finally {
      executor.shutdown();
    }
  }

  private DexConsumer dexEntry(
      ZipFile zipFile, ZipEntry classEntry, CompilationMode mode, ExecutorService executor)
      throws IOException, CompilationFailedException {
    DexConsumer consumer = new DexConsumer();
    D8Command.Builder builder = D8Command.builder();
    builder
        .setProgramConsumer(consumer)
        .setMode(mode)
        .setMinApiLevel(13) // H_MR2.
        .setDisableDesugaring(true)
        .setIntermediate(true);
    try (InputStream stream = zipFile.getInputStream(classEntry)) {
      builder.addClassProgramData(
          ByteStreams.toByteArray(stream),
          new ArchiveEntryOrigin(
              classEntry.getName(), new PathOrigin(Paths.get(zipFile.getName()))));
    }
    D8.run(builder.build(), executor);
    return consumer;
  }
}
