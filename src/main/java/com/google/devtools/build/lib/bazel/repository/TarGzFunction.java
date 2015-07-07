// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.io.CountingInputStream;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nullable;

/**
 * Creates a  repository by unarchiving a .tar.gz file.
 */
public class TarGzFunction implements SkyFunction {

  public static final SkyFunctionName NAME = SkyFunctionName.create("TAR_GZ_FUNCTION");

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    DecompressorDescriptor descriptor = (DecompressorDescriptor) skyKey.argument();

    try (GZIPInputStream gzipStream = new GZIPInputStream(
        new FileInputStream(descriptor.archivePath().getPathFile()))) {
      TarInputStream inputStream = new TarInputStream(gzipStream);
      while (inputStream.available() != 0) {
        TarEntry entry = inputStream.getNextEntry();
        Path filename = descriptor.repositoryPath().getRelative(entry.getFilename());
        FileSystemUtils.createDirectoryAndParents(filename.getParentDirectory());
        if (entry.isDirectory()) {
          FileSystemUtils.createDirectoryAndParents(filename);
        } else {
          Files.copy(entry, filename.getPathFile().toPath());
          filename.chmod(entry.getPermissions());
        }
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    return new DecompressorValue(descriptor.repositoryPath());
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static class TarInputStream {
    private final CountingInputStream inputStream;
    private String nextFilename;

    public TarInputStream(InputStream inputStream) {
      this.inputStream = new CountingInputStream(inputStream);
    }

    public int available() throws IOException {
      nextFilename = TarEntry.parseFilename(inputStream);
      if (nextFilename.isEmpty()) {
        // We've probably reached the padding at the end of the .tar file.
        return 0;
      }
      return inputStream.available();
    }

    public TarEntry getNextEntry() throws IOException {
      return new TarEntry(nextFilename, inputStream);
    }
  }

  private static class TarEntry extends InputStream {
    private static final int BUFFER_SIZE = 512;
    private static final int FILENAME_SIZE = 100;
    private static final int PERMISSIONS_SIZE = 8;
    private static final int FILE_SIZE = 12;
    private static final int TYPE_SIZE = 1;
    private static final int USTAR_SIZE = 6;

    public enum FileType {
      NORMAL, DIRECTORY
    }

    private final String filename;
    private final int permissions;
    private final FileType type;
    private final CountingInputStream inputStream;
    // Tar format pads the content to blocks of 512 bytes, so when we're done reading the file skip
    // this many bytes to arrive at the next file.
    private final int finalSkip;
    private long bytesRemaining;
    private boolean done;

    public TarEntry(String filename, CountingInputStream inputStream) throws IOException {
      byte buffer[] = new byte[BUFFER_SIZE];

      this.filename = filename;

      // Permissions.
      if (inputStream.read(buffer, 0, PERMISSIONS_SIZE) != PERMISSIONS_SIZE) {
        throw new IOException("Error reading tar file (could not read permissions for " + filename
            + ")");
      }

      String permissionsString;
      if (buffer[PERMISSIONS_SIZE - 2] == ' ') {
        // The permissions look like 000644 \0 (OS X, sigh).
        permissionsString = new String(buffer, 0, PERMISSIONS_SIZE - 2);
      } else {
        // The permissions look like 0000644\0 (Linux).
        permissionsString = new String(buffer, 0, PERMISSIONS_SIZE - 1);
      }
      try {
        permissions = Integer.parseInt(permissionsString, 8);
      } catch (NumberFormatException e) {
        throw new IOException("Error reading tar file (could not parse permissions of " + filename
            + "): [" + permissionsString + "]");
      }

      // User & group IDs.
      inputStream.skip(16);

      // File size.
      if (inputStream.read(buffer, 0, FILE_SIZE) != FILE_SIZE) {
        throw new IOException("Error reading tar file (could not read file size of " + filename
            + ")");
      }

      // 12345678901\0 in base 8, bizarly.
      bytesRemaining = Long.parseLong(new String(buffer, 0, FILE_SIZE - 1), 8);
      if (bytesRemaining % 512 == 0) {
        if (bytesRemaining == 0) {
          done = true;
        }
        finalSkip = 0;
      } else {
        done = false;
        finalSkip = (int) (512 - bytesRemaining % 512);
      }

      // Timestamp and checksum.
      // TODO(kchodorow): actually check the checksum.
      inputStream.skip(20);

      if (inputStream.read(buffer, 0, TYPE_SIZE) != TYPE_SIZE) {
        throw new IOException("Error reading tar file (could not read file type of " + filename
            + ")");
      }
      char type = (char) buffer[0];
      if (type == '0') {
        this.type = FileType.NORMAL;
      } else if (type == '5') {
        this.type = FileType.DIRECTORY;
      } else {
        // TODO(kchodorow): support links.
        throw new IOException("Error reading tar file (unknown file type " + type + " for file "
            + filename + ")");
      }

      // Skip name of linked file.
      inputStream.skip(100);

      // USTAR constant.
      if (inputStream.read(buffer, 0, USTAR_SIZE) != USTAR_SIZE
          || !new String(buffer, 0, USTAR_SIZE - 1).equals("ustar")) {
        // TODO(kchodorow): support old-style tar format.
        throw new IOException("Error reading tar file (" + filename + " did not specify 'ustar')");
      }

      // Skip the rest of the ustar preamble.
      inputStream.skip(249);
      // We're now at position 512.

      // Ready to read content.
      this.inputStream = inputStream;
    }

    private static String parseFilename(InputStream inputStream) throws IOException {
      byte buffer[] = new byte[FILENAME_SIZE];

      if (inputStream.read(buffer, 0, FILENAME_SIZE) != FILENAME_SIZE) {
        throw new IOException("Error reading tar file (could not read filename)");
      }

      int actualFilenameLength = 0;
      while (actualFilenameLength < FILENAME_SIZE) {
        if (buffer[actualFilenameLength] == 0) {
          break;
        }
        actualFilenameLength++;
      }
      return new String(buffer, 0, actualFilenameLength);
    }

    public String getFilename() {
      return filename;
    }

    public int getPermissions() {
      return permissions;
    }

    public boolean isDirectory() {
      return type == FileType.DIRECTORY;
    }

    @Override
    public int read() throws IOException {
      if (--bytesRemaining < 0) {
        if (!done) {
          inputStream.skip(finalSkip);
        }
        done = true;
        return -1;
      }
      return inputStream.read();
    }
  }
}
