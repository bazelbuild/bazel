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
package com.google.devtools.build.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Ordering;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;

import com.android.SdkConstants;
import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.res2.MergingException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.Flushable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

/**
 * Writer for UnwrittenMergedAndroidData.
 */
public class AndroidDataWriter implements Flushable, AndroidDataWritingVisitor {

  private static final class WriteValuesXmlTask implements Callable<Boolean> {

    private final Path valuesPath;
    private final Map<FullyQualifiedName, Iterable<String>> valueFragments;

    WriteValuesXmlTask(Path valuesPath, Map<FullyQualifiedName, Iterable<String>> valueFragments) {
      this.valuesPath = valuesPath;
      this.valueFragments = valueFragments;
    }

    @Override
    public Boolean call() throws Exception {
      // TODO(corysmith): replace the xml writing with a real xml writing library.
      Files.createDirectories(valuesPath.getParent());
      try (BufferedWriter writer =
          Files.newBufferedWriter(
              valuesPath,
              StandardCharsets.UTF_8,
              StandardOpenOption.CREATE_NEW,
              StandardOpenOption.WRITE)) {
        writer.write(START_RESOURCES);
        for (FullyQualifiedName key :
            Ordering.natural().immutableSortedCopy(valueFragments.keySet())) {
          for (String line : valueFragments.get(key)) {
            writer.write(line);
            writer.write(LINE_END);
          }
        }
        writer.write(END_RESOURCES);
      }
      return Boolean.TRUE;
    }
  }

  private final class CopyTask implements Callable<Boolean> {

    private final Path sourcePath;

    private final Path destinationPath;

    private CopyTask(Path sourcePath, Path destinationPath) {
      this.sourcePath = sourcePath;
      this.destinationPath = destinationPath;
    }

    @Override
    public Boolean call() throws Exception {
      Files.createDirectories(destinationPath.getParent());
      Files.copy(sourcePath, destinationPath, StandardCopyOption.REPLACE_EXISTING);
      return Boolean.TRUE;
    }
  }

  public static final char[] START_RESOURCES =
      ("<resources xmlns:xliff=\"" + XmlResourceValues.XLIFF_NAMESPACE + "\">").toCharArray();
  public static final char[] END_RESOURCES = "</resources>".toCharArray();
  private static final char[] LINE_END = "\n".toCharArray();
  private static final PngCruncher NOOP_CRUNCHER =
      new PngCruncher() {
        @Override
        public void crunchPng(@NonNull File source, @NonNull File destination)
            throws InterruptedException, LoggedErrorException, IOException {
          Files.createDirectories(destination.toPath().getParent());
          Files.copy(source.toPath(), destination.toPath());
        }
      };

  private final Path destination;
  private final Map<String, Map<FullyQualifiedName, Iterable<String>>> valueFragments =
      new HashMap<>();
  private final Path resourceDirectory;
  private final Path assetDirectory;
  private final PngCruncher cruncher;
  private final List<ListenableFuture<Boolean>> writeTasks = new ArrayList<>();
  private final ListeningExecutorService executorService;

  private AndroidDataWriter(
      Path destination,
      Path resourceDirectory,
      Path assetsDirectory,
      PngCruncher cruncher,
      ListeningExecutorService executorService) {
    this.destination = destination;
    this.resourceDirectory = resourceDirectory;
    this.assetDirectory = assetsDirectory;
    this.cruncher = cruncher;
    this.executorService = executorService;
  }

  /**
   * Creates a new, naive writer for testing.
   *
   * This writer has "assets" and a "res" directory from the destination directory, as well as a
   * noop png cruncher and a {@link ExecutorService} of 1 thread.
   *
   * @param destination The base directory to derive all paths.
   * @return A new {@link AndroidDataWriter}.
   */
  @VisibleForTesting
  static AndroidDataWriter createWithDefaults(Path destination) {
    return createWith(
        destination,
        destination.resolve("res"),
        destination.resolve("assets"),
        NOOP_CRUNCHER,
        MoreExecutors.newDirectExecutorService());
  }

  /**
   * Creates a new writer.
   *
   * @param manifestDirectory The base directory for the AndroidManifest.
   * @param resourceDirectory The directory to copy resources into.
   * @param assetsDirectory The directory to copy assets into.
   * @param cruncher The cruncher for png files. If the cruncher is null, it will be replaced with a
   *    noop cruncher.
   * @param executorService An execution service for multi-threaded writing.
   * @return A new {@link AndroidDataWriter}.
   */
  public static AndroidDataWriter createWith(
      Path manifestDirectory,
      Path resourceDirectory,
      Path assetsDirectory,
      @Nullable PngCruncher cruncher,
      ListeningExecutorService executorService) {
    return new AndroidDataWriter(
        manifestDirectory,
        resourceDirectory,
        assetsDirectory,
        cruncher == null ? NOOP_CRUNCHER : cruncher,
        executorService);
  }

  @Override
  public Path copyManifest(Path sourceManifest) throws IOException {
    // aapt won't read any manifest that is not named AndroidManifest.xml,
    // so we hard code it here.
    Path destinationManifest = destination.resolve("AndroidManifest.xml");
    copy(sourceManifest, destinationManifest);
    return destinationManifest;
  }

  public Path assetDirectory() {
    return assetDirectory;
  }

  public Path resourceDirectory() {
    return resourceDirectory;
  }

  @Override
  public void copyAsset(Path source, String relativeDestinationPath) throws IOException {
    copy(source, assetDirectory.resolve(relativeDestinationPath));
  }

  @Override
  public void copyResource(Path source, String relativeDestinationPath)
      throws IOException, MergingException {
    Path destinationPath = resourceDirectory.resolve(relativeDestinationPath);
    if (!source.getParent().getFileName().toString().startsWith(SdkConstants.FD_RES_RAW)
        && source.getFileName().toString().endsWith(SdkConstants.DOT_PNG)) {
      try {
        Files.createDirectories(destinationPath.getParent());
        cruncher.crunchPng(source.toFile(), destinationPath.toFile());
      } catch (InterruptedException | LoggedErrorException e) {
        throw new MergingException(e);
      }
    } else {
      copy(source, destinationPath);
    }
  }

  private void copy(final Path sourcePath, final Path destinationPath) {
    writeTasks.add(executorService.submit(new CopyTask(sourcePath, destinationPath)));
  }

  /**
   * Finalizes all operations and flushes the buffers.
   */
  @Override
  public void flush() throws IOException {
    for (Entry<String, Map<FullyQualifiedName, Iterable<String>>> entry :
        valueFragments.entrySet()) {
      writeTasks.add(
          executorService.submit(
              new WriteValuesXmlTask(
                  resourceDirectory().resolve(entry.getKey()), entry.getValue())));
    }
    FailedFutureAggregator.forIOExceptionsWithMessage("Failures during writing.")
        .aggregateAndMaybeThrow(writeTasks);

    writeTasks.clear();
    valueFragments.clear();
  }

  @Override
  public void writeToValuesXml(FullyQualifiedName key, Iterable<String> xmlFragment) {
    String valuesPathString = key.valuesPath();
    if (!valueFragments.containsKey(valuesPathString)) {
      valueFragments.put(
          valuesPathString, new TreeMap<FullyQualifiedName, Iterable<String>>(Ordering.natural()));
    }
    valueFragments.get(valuesPathString).put(key, xmlFragment);
  }
}
