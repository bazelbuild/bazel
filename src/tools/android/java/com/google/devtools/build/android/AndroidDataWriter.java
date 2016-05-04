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

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.logging.Level.SEVERE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Ordering;

import com.android.SdkConstants;
import com.android.annotations.NonNull;
import com.android.annotations.Nullable;
import com.android.ide.common.internal.LoggedErrorException;
import com.android.ide.common.internal.PngCruncher;

import java.io.File;
import java.io.Flushable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Writer for UnwrittenMergedAndroidData.
 */
// TODO(corysmith): Profile this class on large datasets and look for bottlenecks, as this class
// does all the IO.
public class AndroidDataWriter implements Flushable, AndroidDataWritingVisitor {

  private static final byte[] START_RESOURCES = "<resources>".getBytes(UTF_8);
  private static final byte[] END_RESOURCES = "</resources>".getBytes(UTF_8);
  private static final byte[] LINE_END = "\n".getBytes(UTF_8);
  private static final Logger logger = Logger.getLogger(AndroidDataWriter.class.getName());
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
  private final Map<FullyQualifiedName, Iterable<String>> valueFragments = new HashMap<>();
  private Path resourceDirectory;
  private Path assetDirectory;
  private final PngCruncher cruncher;

  private AndroidDataWriter(
      Path destination, Path resourceDirectory, Path assetsDirectory, PngCruncher cruncher) {
    this.destination = destination;
    this.resourceDirectory = resourceDirectory;
    this.assetDirectory = assetsDirectory;
    this.cruncher = cruncher;
  }

  /**
   * Creates a new, naive writer for testing.
   *
   * This writer has "assets" and a "res" directory from the destination directory, as well as a 
   * noop png cruncher.
   *
   * @param destination The base directory to derive all paths.
   * @return A new {@link AndroidDataWriter}.
   */
  @VisibleForTesting
  static AndroidDataWriter from(Path destination) {
    return createWith(
        destination, destination.resolve("res"), destination.resolve("assets"), NOOP_CRUNCHER);
  }

  /**
   * Creates a new writer.
   *
   * @param manifestDirectory The base directory for the AndroidManifest.
   * @param resourceDirectory The directory to copy resources into.
   * @param assetsDirectory The directory to copy assets into.
   * @param cruncher The cruncher for png files. If the cruncher is null, it will be replaced with a
   *     noop cruncher.
   * @return A new {@link AndroidDataWriter}.
   */
  public static AndroidDataWriter createWith(
      Path manifestDirectory,
      Path resourceDirectory,
      Path assetsDirectory,
      @Nullable PngCruncher cruncher) {
    return new AndroidDataWriter(
        manifestDirectory,
        resourceDirectory,
        assetsDirectory,
        cruncher == null ? NOOP_CRUNCHER : cruncher);
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
      throws IOException {
    Path destinationPath = resourceDirectory.resolve(relativeDestinationPath);
    if (source.getParent().getFileName().toString().startsWith(SdkConstants.DRAWABLE_FOLDER)
        && source.getFileName().toString().endsWith(SdkConstants.DOT_PNG)) {
      try {
        Files.createDirectories(destinationPath.getParent());
        cruncher.crunchPng(source.toFile(), destinationPath.toFile());
      } catch (InterruptedException | LoggedErrorException e) {
        // TODO(corysmith): Change to a MergingException
        throw new RuntimeException(e);
      }
    } else {
      copy(source, destinationPath);
    }
  }

  private void copy(Path sourcePath, Path destinationPath) throws IOException {
    Files.createDirectories(destinationPath.getParent());
    Files.copy(sourcePath, destinationPath, StandardCopyOption.REPLACE_EXISTING);
  }

  /**
   * Finalizes all operations and flushes the buffers.
   */
  @Override
  public void flush() throws IOException {
    // TODO(corysmith): replace the xml writing with a real xml writing library.
    Map<Path, FileChannel> channels = new HashMap<>();
    try {
      for (FullyQualifiedName key : Ordering.natural().sortedCopy(valueFragments.keySet())) {
        Path valuesPath = resourceDirectory().resolve(key.valuesPath());
        FileChannel channel;
        if (!channels.containsKey(valuesPath)) {
          Files.createDirectories(valuesPath.getParent());
          channel =
              FileChannel.open(valuesPath, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
          channel.write(ByteBuffer.wrap(START_RESOURCES));
          channels.put(valuesPath, channel);
        } else {
          channel = channels.get(valuesPath);
        }
        for (String line : valueFragments.get(key)) {
          channel.write(ByteBuffer.wrap(line.getBytes(UTF_8)));
          channel.write(ByteBuffer.wrap(LINE_END));
        }
      }
    } finally {
      List<Exception> suppressedExceptions = new ArrayList<>();
      for (FileChannel channel : channels.values()) {
        try {
          channel.write(ByteBuffer.wrap(END_RESOURCES));
          channel.close();
        } catch (IOException e) {
          logger.log(SEVERE, "Error during writing", e);
          suppressedExceptions.add(e);
        }
      }
      if (!suppressedExceptions.isEmpty()) {
        throw new IOException("IOException(s) thrown during writing. See logs.");
      }
    }
    valueFragments.clear();
  }

  @Override
  public void writeToValuesXml(FullyQualifiedName key, Iterable<String> xmlFragment) {
    valueFragments.put(key, xmlFragment);
  }
}
