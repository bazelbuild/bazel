/* Copyright 2016 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.integration;

import org.brotli.dec.BrotliInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Decompress files and (optionally) checks their checksums.
 *
 * <p> File are read from ZIP archive passed as an array of bytes. Multiple checkers negotiate about
 * task distribution via shared AtomicInteger counter.
 * <p> All entries are expected to be valid brotli compressed streams and output CRC64 checksum
 * is expected to match the checksum hex-encoded in the first part of entry name.
 */
public class BundleChecker implements Runnable {
  private final AtomicInteger nextJob;
  private final InputStream input;
  private final boolean sanityCheck;

  /**
   * @param sanityCheck do not calculate checksum and ignore {@link IOException}.
   */
  public BundleChecker(InputStream input, AtomicInteger nextJob, boolean sanityCheck) {
    this.input = input;
    this.nextJob = nextJob;
    this.sanityCheck = sanityCheck;
  }

  private long decompressAndCalculateCrc(ZipInputStream input) throws IOException {
    /* Do not allow entry readers to close the whole ZipInputStream. */
    FilterInputStream entryStream = new FilterInputStream(input) {
      @Override
      public void close() {}
    };

    BrotliInputStream decompressedStream = new BrotliInputStream(entryStream);
    long crc;
    try {
      crc = BundleHelper.fingerprintStream(decompressedStream);
    } finally {
      decompressedStream.close();
    }
    return crc;
  }

  @Override
  public void run() {
    String entryName = "";
    ZipInputStream zis = new ZipInputStream(input);
    try {
      int entryIndex = 0;
      ZipEntry entry;
      int jobIndex = nextJob.getAndIncrement();
      while ((entry = zis.getNextEntry()) != null) {
        if (entry.isDirectory()) {
          continue;
        }
        if (entryIndex++ != jobIndex) {
          zis.closeEntry();
          continue;
        }
        entryName = entry.getName();
        long entryCrc = BundleHelper.getExpectedFingerprint(entryName);
        try {
          if (entryCrc != decompressAndCalculateCrc(zis) && !sanityCheck) {
            throw new RuntimeException("CRC mismatch");
          }
        } catch (IOException iox) {
          if (!sanityCheck) {
            throw new RuntimeException("Decompression failed", iox);
          }
        }
        zis.closeEntry();
        entryName = "";
        jobIndex = nextJob.getAndIncrement();
      }
      zis.close();
      input.close();
    } catch (Throwable ex) {
      throw new RuntimeException(entryName, ex);
    }
  }

  public static void main(String[] args) throws FileNotFoundException {
    int argsOffset = 0;
    boolean sanityCheck = false;
    if (args.length != 0) {
      if (args[0].equals("-s")) {
        sanityCheck = true;
        argsOffset = 1;
      }
    }
    if (args.length == argsOffset) {
      throw new RuntimeException("Usage: BundleChecker [-s] <fileX.zip> ...");
    }
    for (int i = argsOffset; i < args.length; ++i) {
      new BundleChecker(new FileInputStream(args[i]), new AtomicInteger(0), sanityCheck).run();
    }
  }
}
