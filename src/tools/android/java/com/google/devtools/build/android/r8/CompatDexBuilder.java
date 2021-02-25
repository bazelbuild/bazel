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
import com.google.common.io.ByteStreams;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
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

  private String input;
  private String output;
  private int numberOfThreads = min(8, Runtime.getRuntime().availableProcessors());
  private boolean noLocals;

  public static void main(String[] args)
      throws IOException, InterruptedException, ExecutionException {
    new CompatDexBuilder().run(args);
  }

  @SuppressWarnings("JdkObsolete")
  private void run(String[] args) throws IOException, InterruptedException, ExecutionException {
    List<String> flags = new ArrayList<>();

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
          System.err.println("Unsupported option: " + flag);
          System.exit(1);
      }
    }

    if (input == null) {
      System.err.println("No input jar specified");
      System.exit(1);
    }

    if (output == null) {
      System.err.println("No output jar specified");
      System.exit(1);
    }

    ExecutorService executor = Executors.newWorkStealingPool(numberOfThreads);
    try (ZipOutputStream out =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(output))))) {

      List<ZipEntry> toDex = new ArrayList<>();

      try (ZipFile zipFile = new ZipFile(input, UTF_8)) {
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
          futures.add(executor.submit(() -> dexEntry(input, zipFile, classEntry, executor)));
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
      String classpath, ZipFile zipFile, ZipEntry classEntry, ExecutorService executor)
      throws IOException, CompilationFailedException {
    DexConsumer consumer = new DexConsumer();
    D8Command.Builder builder = D8Command.builder();
    builder
        .addClasspathFiles(Paths.get(classpath))
        .setProgramConsumer(consumer)
        .setMode(noLocals ? CompilationMode.RELEASE : CompilationMode.DEBUG)
        .setMinApiLevel(13) // H_MR2.
        .setDisableDesugaring(true);
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
