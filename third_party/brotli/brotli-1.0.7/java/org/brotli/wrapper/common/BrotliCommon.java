/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.common;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * JNI wrapper for brotli common.
 */
public class BrotliCommon {
  public static final int RFC_DICTIONARY_SIZE = 122784;

  /* 96cecd2ee7a666d5aa3627d74735b32a */
  private static final byte[] RFC_DICTIONARY_MD5 = {
    -106, -50, -51, 46, -25, -90, 102, -43, -86, 54, 39, -41, 71, 53, -77, 42
  };

  /* 72b41051cb61a9281ba3c4414c289da50d9a7640 */
  private static final byte[] RFC_DICTIONARY_SHA_1 = {
    114, -76, 16, 81, -53, 97, -87, 40, 27, -93, -60, 65, 76, 40, -99, -91, 13, -102, 118, 64
  };

  /* 20e42eb1b511c21806d4d227d07e5dd06877d8ce7b3a817f378f313653f35c70 */
  private static final byte[] RFC_DICTIONARY_SHA_256 = {
    32, -28, 46, -79, -75, 17, -62, 24, 6, -44, -46, 39, -48, 126, 93, -48,
    104, 119, -40, -50, 123, 58, -127, 127, 55, -113, 49, 54, 83, -13, 92, 112
  };

  private static boolean isDictionaryDataSet;
  private static final Object mutex = new Object();

  /**
   * Checks if the given checksum matches MD5 checksum of the RFC dictionary.
   */
  public static boolean checkDictionaryDataMd5(byte[] digest) {
    return Arrays.equals(RFC_DICTIONARY_MD5, digest);
  }

  /**
   * Checks if the given checksum matches SHA-1 checksum of the RFC dictionary.
   */
  public static boolean checkDictionaryDataSha1(byte[] digest) {
    return Arrays.equals(RFC_DICTIONARY_SHA_1, digest);
  }

  /**
   * Checks if the given checksum matches SHA-256 checksum of the RFC dictionary.
   */
  public static boolean checkDictionaryDataSha256(byte[] digest) {
    return Arrays.equals(RFC_DICTIONARY_SHA_256, digest);
  }

  /**
   * Copy bytes to a new direct ByteBuffer.
   *
   * Direct byte buffers are used to supply native code with large data chunks.
   */
  public static ByteBuffer makeNative(byte[] data) {
    ByteBuffer result = ByteBuffer.allocateDirect(data.length);
    result.put(data);
    return result;
  }

  /**
   * Copies data and sets it to be brotli dictionary.
   */
  public static void setDictionaryData(byte[] data) {
    if (data.length != RFC_DICTIONARY_SIZE) {
      throw new IllegalArgumentException("invalid dictionary size");
    }
    synchronized (mutex) {
      if (isDictionaryDataSet) {
        return;
      }
      setDictionaryData(makeNative(data));
    }
  }

  /**
   * Reads data and sets it to be brotli dictionary.
   */
  public static void setDictionaryData(InputStream src) throws IOException {
    synchronized (mutex) {
      if (isDictionaryDataSet) {
        return;
      }
      ByteBuffer copy = ByteBuffer.allocateDirect(RFC_DICTIONARY_SIZE);
      byte[] buffer = new byte[4096];
      int readBytes;
      while ((readBytes = src.read(buffer)) != -1) {
        if (copy.remaining() < readBytes) {
          throw new IllegalArgumentException("invalid dictionary size");
        }
        copy.put(buffer, 0, readBytes);
      }
      if (copy.remaining() != 0) {
        throw new IllegalArgumentException("invalid dictionary size " + copy.remaining());
      }
      setDictionaryData(copy);
    }
  }

  /**
   * Sets data to be brotli dictionary.
   */
  public static void setDictionaryData(ByteBuffer data) {
    if (!data.isDirect()) {
      throw new IllegalArgumentException("direct byte buffer is expected");
    }
    if (data.capacity() != RFC_DICTIONARY_SIZE) {
      throw new IllegalArgumentException("invalid dictionary size");
    }
    synchronized (mutex) {
      if (isDictionaryDataSet) {
        return;
      }
      if (!CommonJNI.nativeSetDictionaryData(data)) {
        throw new RuntimeException("setting dictionary failed");
      }
      isDictionaryDataSet = true;
    }
  }
}
