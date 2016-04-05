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

import com.google.common.collect.Ordering;

import java.io.Flushable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * Writer for UnwrittenMergedAndroidData.
 */
// TODO(corysmith): Profile this class on large datasets and look for bottlenecks, as this class
// does all the IO.
public class AndroidDataWriter implements Flushable, AndroidDataWritingVisitor {
  private final Path destination;
  private final Map<FullyQualifiedName, Iterable<String>> valueFragments = new HashMap<>();
  private Path resourceDirectory;
  private Path assetDirectory;

  private AndroidDataWriter(Path destination, Path resourceDirectory, Path assetsDirectory) {
    this.destination = destination;
    this.resourceDirectory = resourceDirectory;
    this.assetDirectory = assetsDirectory;
  }

  public static AndroidDataWriter from(Path destination) {
    return new AndroidDataWriter(
        destination, destination.resolve("res"), destination.resolve("assets"));
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
  public void copyResource(Path source, String relativeDestinationPath) throws IOException {
    copy(source, resourceDirectory.resolve(relativeDestinationPath));
  }

  private void copy(Path sourcePath, Path destinationPath) throws IOException {
    Files.createDirectories(destinationPath.getParent());
    Files.copy(sourcePath, destinationPath);
  }

  /**
   * Finalizes all operations and flushes the buffers.
   */
  @Override
  public void flush() throws IOException {
    Path values = Files.createDirectories(resourceDirectory().resolve("values"));
    try (FileChannel channel =
            FileChannel.open(
                values.resolve("values.xml"),
                StandardOpenOption.CREATE_NEW,
                StandardOpenOption.WRITE)) {
      channel.write(ByteBuffer.wrap("<resources>".getBytes(UTF_8)));
      for (FullyQualifiedName key : Ordering.natural().sortedCopy(valueFragments.keySet())) {
        for (String line : valueFragments.get(key)) {
          channel.write(ByteBuffer.wrap(line.getBytes(UTF_8)));
        }
      }
      channel.write(ByteBuffer.wrap("</resources>".getBytes(UTF_8)));
    }
    valueFragments.clear();
  }

  @Override
  public void writeToValuesXml(FullyQualifiedName key, Iterable<String> xmlFragment) {
    valueFragments.put(key, xmlFragment);
  }
}
