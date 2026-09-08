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

import com.google.devtools.build.zip.ZipFileEntry.Compression;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

/**
 * Hermetic packaging tool for Bazel release archives (package.zip).
 *
 * <p>Guarantees required archive entry ordering (A-server.jar first, sorted files, install_base_key
 * last), normalized 1980-01-01 timestamps, automatic unpacking of embedded_tools zips and platform
 * tar archives, generation of build-label.txt from server jar properties, and synchronous Win32
 * kernel flushing (FileChannel.force(true)) before process termination.
 */
public final class PackageZipper {
  private PackageZipper() {}

  private static final class PackageEntry implements Comparable<PackageEntry> {
    final String name;
    final byte[] data;

    PackageEntry(String name, byte[] data) {
      this.name = name;
      this.data = data;
    }

    @Override
    public int compareTo(PackageEntry other) {
      return this.name.compareTo(other.name);
    }
  }

  public static void main(String[] args) throws IOException {
    boolean fastCompression = false;
    List<String> positionalArgs = new ArrayList<>();
    for (String arg : args) {
      if (arg.equals("--fast")) {
        fastCompression = true;
      } else {
        positionalArgs.add(arg);
      }
    }

    if (positionalArgs.size() < 3) {
      System.err.println(
          "Usage: PackageZipper [--fast] <output_zip> <server_jar> <install_base_key> [file...]");
      System.exit(1);
    }

    Path outputPath = Path.of(positionalArgs.get(0));
    Path serverJar = Path.of(positionalArgs.get(1));
    Path installBaseKey = Path.of(positionalArgs.get(2));
    int compressionLevel = fastCompression ? Deflater.BEST_SPEED : Deflater.BEST_COMPRESSION;

    List<Path> packageFiles = new ArrayList<>();
    for (int i = 3; i < positionalArgs.size(); i++) {
      Path p = Path.of(positionalArgs.get(i));
      if (Files.isRegularFile(p)) {
        packageFiles.add(p);
      }
    }

    String buildLabel = "no_version";
    try (ZipFile serverZip = new ZipFile(serverJar.toFile(), UTF_8)) {
      ZipEntry buildDataEntry = serverZip.getEntry("build-data.properties");
      if (buildDataEntry != null) {
        try (InputStream in = serverZip.getInputStream(buildDataEntry)) {
          Properties props = new Properties();
          props.load(in);
          String label = props.getProperty("build.label");
          if (label != null && !label.isEmpty()) {
            buildLabel = label;
          }
        }
      }
    }

    List<PackageEntry> entries = new ArrayList<>();
    entries.add(new PackageEntry("build-label.txt", buildLabel.getBytes(UTF_8)));

    for (Path file : packageFiles) {
      String fileName = file.getFileName().toString();
      if (fileName.startsWith("embedded_tools") && fileName.endsWith(".zip")) {
        try (ZipFile zipFile = new ZipFile(file.toFile(), UTF_8)) {
          List<? extends ZipEntry> zipEntries = Collections.list(zipFile.entries());
          for (ZipEntry ze : zipEntries) {
            if (ze.isDirectory()) {
              continue;
            }
            try (InputStream in = zipFile.getInputStream(ze)) {
              byte[] data = readAllBytes(in);
              entries.add(new PackageEntry("embedded_tools/" + ze.getName(), data));
            }
          }
        }
      } else if (fileName.endsWith(".tar")
          || fileName.endsWith(".tar.gz")
          || fileName.endsWith(".tgz")) {
        try (InputStream fis = Files.newInputStream(file);
            TarArchiveInputStream tarIn = new TarArchiveInputStream(fis)) {
          TarArchiveEntry te;
          while ((te = tarIn.getNextEntry()) != null) {
            if (te.isDirectory()) {
              continue;
            }
            byte[] data = readAllBytes(tarIn);
            entries.add(new PackageEntry(te.getName(), data));
          }
        }
      } else {
        entries.add(new PackageEntry(fileName, Files.readAllBytes(file)));
      }
    }

    Collections.sort(entries);

    if (outputPath.getParent() != null) {
      Files.createDirectories(outputPath.getParent());
    }

    try (FileOutputStream fos = new FileOutputStream(outputPath.toFile())) {
      ZipWriter writer = new ZipWriter(fos, UTF_8, true);

      // 1. Mandatory first entry: A-server.jar (for Bazel client JVM bootstrapper)
      writeBytesEntry(writer, "A-server.jar", Files.readAllBytes(serverJar), compressionLevel);

      // 2. Sorted intermediate package entries (including build-label.txt, embedded_tools/,
      // platforms/, tools, binaries)
      for (PackageEntry pe : entries) {
        writeBytesEntry(writer, pe.name, pe.data, compressionLevel);
      }

      // 3. Mandatory last entry: install_base_key
      writeBytesEntry(
          writer, "install_base_key", Files.readAllBytes(installBaseKey), compressionLevel);

      writer.finish();

      // Explicitly force NTFS MFT EOF and dirty cache pages to commit
      // synchronously before process exit.
      fos.getChannel().force(true);

      writer.close();
    }
  }

  private static byte[] readAllBytes(InputStream in) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[65536];
    int n;
    while ((n = in.read(buffer)) > 0) {
      out.write(buffer, 0, n);
    }
    return out.toByteArray();
  }

  private static void writeBytesEntry(
      ZipWriter writer, String entryName, byte[] rawBytes, int compressionLevel)
      throws IOException {
    CRC32 crc32 = new CRC32();
    crc32.update(rawBytes);
    long crc = crc32.getValue();

    byte[] compressedBytes = deflate(rawBytes, compressionLevel);

    ZipFileEntry entry = new ZipFileEntry(entryName);
    // DOS timestamps carry no time zone and are always interpreted in the local one, so the
    // instant has to be derived from the local zone to record the same date everywhere.
    entry.setTime(ZipUtil.DOS_EPOCH);
    entry.setVersion((short) 20);

    if (compressedBytes.length >= rawBytes.length && rawBytes.length > 0) {
      entry.setMethod(Compression.STORED);
      entry.setSize(rawBytes.length);
      entry.setCompressedSize(rawBytes.length);
      entry.setCrc(crc);
      writer.putNextEntry(entry);
      writer.write(rawBytes);
    } else {
      entry.setMethod(Compression.DEFLATED);
      entry.setSize(rawBytes.length);
      entry.setCompressedSize(compressedBytes.length);
      entry.setCrc(crc);
      writer.putNextEntry(entry);
      writer.write(compressedBytes);
    }
    writer.closeEntry();
  }

  private static byte[] deflate(byte[] data, int compressionLevel) {
    Deflater deflater = new Deflater(compressionLevel, true);
    deflater.setInput(data);
    deflater.finish();
    ByteArrayOutputStream out = new ByteArrayOutputStream(data.length);
    byte[] buf = new byte[65536];
    while (!deflater.finished()) {
      int count = deflater.deflate(buf);
      out.write(buf, 0, count);
    }
    deflater.end();
    return out.toByteArray();
  }
}
