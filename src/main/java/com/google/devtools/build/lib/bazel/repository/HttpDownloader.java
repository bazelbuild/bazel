// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

/**
 * Helper class for downloading a file from a URL.
 */
public class HttpDownloader {
  private static final int BUFFER_SIZE = 2048;

  private final URL url;
  private final String sha256;
  private final Path outputDirectory;

  HttpDownloader(URL url, String sha256, Path outputDirectory) {
    this.url = url;
    this.sha256 = sha256;
    this.outputDirectory = outputDirectory;
  }

  /**
   * Attempt to download a file from the repository's URL. Returns the path to the file downloaded.
   */
  public Path download() throws IOException {
    String filename = new PathFragment(url.getPath()).getBaseName();
    if (filename.isEmpty()) {
      filename = "temp";
    }
    Path destination = outputDirectory.getRelative(filename);

    try (OutputStream outputStream = destination.getOutputStream()) {
      ReadableByteChannel rbc = getChannel(url);
      ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_SIZE);
      while (rbc.read(byteBuffer) > 0) {
        byteBuffer.flip();
        while (byteBuffer.hasRemaining()) {
          outputStream.write(byteBuffer.get());
        }
        byteBuffer.flip();
      }
    } catch (IOException e) {
      throw new IOException(
          "Error downloading " + url + " to " + destination + ": " + e.getMessage());
    }

    String downloadedSha256;
    try {
      downloadedSha256 = getHash(Hashing.sha256().newHasher(), destination);
    } catch (IOException e) {
      throw new IOException(
          "Could not hash file " + destination + ": " + e.getMessage() + ", expected SHA-256 of "
              + sha256 + ")");
    }
    if (!downloadedSha256.equals(sha256)) {
      throw new IOException(
          "Downloaded file at " + destination + " has SHA-256 of " + downloadedSha256
              + ", does not match expected SHA-256 (" + sha256 + ")");
    }
    return destination;
  }

  @VisibleForTesting
  protected ReadableByteChannel getChannel(URL url) throws IOException {
    return Channels.newChannel(url.openStream());
  }

  public static String getHash(Hasher hasher, Path path) throws IOException {
    byte byteBuffer[] = new byte[BUFFER_SIZE];
    try (InputStream stream = path.getInputStream()) {
      int numBytesRead = stream.read(byteBuffer);
      while (numBytesRead != -1) {
        if (numBytesRead != 0) {
          // If more than 0 bytes were read, add them to the hash.
          hasher.putBytes(byteBuffer, 0, numBytesRead);
        }
        numBytesRead = stream.read(byteBuffer);
      }
    }
    return hasher.hash().toString();
  }
}
