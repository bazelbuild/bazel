/* Copyright 2016 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.integration;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Utilities to work test files bundles in zip archive.
 */
public class BundleHelper {
  private BundleHelper() { }

  public static List<String> listEntries(InputStream input) throws IOException {
    List<String> result = new ArrayList<String>();
    ZipInputStream zis = new ZipInputStream(input);
    ZipEntry entry;
    try {
      while ((entry = zis.getNextEntry()) != null) {
        if (!entry.isDirectory()) {
          result.add(entry.getName());
        }
        zis.closeEntry();
      }
    } finally {
      zis.close();
    }
    return result;
  }

  public static byte[] readStream(InputStream input) throws IOException {
    ByteArrayOutputStream result = new ByteArrayOutputStream();
    byte[] buffer = new byte[65536];
    int bytesRead;
    while ((bytesRead = input.read(buffer)) != -1) {
      result.write(buffer, 0, bytesRead);
    }
    return result.toByteArray();
  }

  public static byte[] readEntry(InputStream input, String entryName) throws IOException {
    ZipInputStream zis = new ZipInputStream(input);
    ZipEntry entry;
    try {
      while ((entry = zis.getNextEntry()) != null) {
        if (entry.getName().equals(entryName)) {
          byte[] result = readStream(zis);
          zis.closeEntry();
          return result;
        }
        zis.closeEntry();
      }
    } finally {
      zis.close();
    }
    /* entry not found */
    return null;
  }

  /** ECMA CRC64 polynomial. */
  private static final long CRC_64_POLY =
      new BigInteger("C96C5795D7870F42", 16).longValue();

  /**
   * Rolls CRC64 calculation.
   *
   * <p> {@code CRC64(data) = -1 ^ updateCrc64((... updateCrc64(-1, firstBlock), ...), lastBlock);}
   * <p> This simple and reliable checksum is chosen to make is easy to calculate the same value
   * across the variety of languages (C++, Java, Go, ...).
   */
  public static long updateCrc64(long crc, byte[] data, int offset, int length) {
    for (int i = offset; i < offset + length; ++i) {
      long c = (crc ^ (long) (data[i] & 0xFF)) & 0xFF;
      for (int k = 0; k < 8; k++) {
        c = ((c & 1) == 1) ? CRC_64_POLY ^ (c >>> 1) : c >>> 1;
      }
      crc = c ^ (crc >>> 8);
    }
    return crc;
  }

  /**
   * Calculates CRC64 of stream contents.
   */
  public static long fingerprintStream(InputStream input) throws IOException {
    byte[] buffer = new byte[65536];
    long crc = -1;
    while (true) {
      int len = input.read(buffer);
      if (len <= 0) {
        break;
      }
      crc = updateCrc64(crc, buffer, 0, len);
    }
    return ~crc;
  }

  public static long getExpectedFingerprint(String entryName) {
    int dotIndex = entryName.indexOf('.');
    String entryCrcString = (dotIndex == -1) ? entryName : entryName.substring(0, dotIndex);
    return new BigInteger(entryCrcString, 16).longValue();
  }
}
