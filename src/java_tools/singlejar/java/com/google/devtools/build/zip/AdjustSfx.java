// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.zip;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Utility to adjust the central directory offsets of a self-extracting zip archive created by
 * prepending a launcher (or any preamble) to a zip file. Similar to "zip -qA", but reads from
 * input_file and writes to output_file.
 */
public final class AdjustSfx {

  private AdjustSfx() {}

  public static void main(String[] args) throws IOException {
    if (args.length != 2) {
      System.err.println("Usage: AdjustSfx <input_file> <output_file>");
      System.exit(1);
    }
    File inputFile = new File(args[0]);
    File outputFile = new File(args[1]);

    try (ZipReader reader = new ZipReader(inputFile, UTF_8, false);
        InputStream in = new FileInputStream(inputFile);
        OutputStream out = new BufferedOutputStream(new FileOutputStream(outputFile))) {

      ZipFileData zipData = reader.getZipData();
      long eocdLocation = reader.findEndOfCentralDirectoryRecord();
      long actualCenEnd =
          (zipData.isZip64() && zipData.getZip64EndOfCentralDirectoryOffset() > 0)
              ? zipData.getZip64EndOfCentralDirectoryOffset()
              : eocdLocation;
      long actualCenStart = actualCenEnd - zipData.getCentralDirectorySize();
      long adjustOffset = actualCenStart - zipData.getCentralDirectoryOffset();

      if (adjustOffset != 0) {
        for (ZipFileEntry entry : zipData.getEntries()) {
          entry.setLocalHeaderOffset(entry.getLocalHeaderOffset() + adjustOffset);
        }
        zipData.setCentralDirectoryOffset(zipData.getCentralDirectoryOffset() + adjustOffset);
        if (zipData.isZip64() && zipData.getZip64EndOfCentralDirectoryOffset() > 0) {
          zipData.setZip64EndOfCentralDirectoryOffset(
              zipData.getZip64EndOfCentralDirectoryOffset() + adjustOffset);
        }
      }

      byte[] buf = new byte[8192];
      long remaining = actualCenStart;
      while (remaining > 0) {
        int toRead = (int) Math.min(buf.length, remaining);
        int bytesRead = in.read(buf, 0, toRead);
        if (bytesRead < 0) {
          throw new IOException("Unexpected EOF reading prefix data from " + inputFile);
        }
        out.write(buf, 0, bytesRead);
        remaining -= bytesRead;
      }

      CentralDirectory.write(zipData, zipData.isZip64(), out);
    }
  }
}
