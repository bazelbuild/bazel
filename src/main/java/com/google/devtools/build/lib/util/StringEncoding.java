package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.protobuf.ByteString;
import java.nio.charset.Charset;

public final class StringEncoding {

  /**
   * Reencodes a string using Bazel's internal raw byte encoding into the equivalent representation
   * for Java stdlib functions, if necessary.
   */
  public static String internalToPlatform(String s) {
    return needsReencodeForPlatform(s) ? new String(s.getBytes(ISO_8859_1), UTF_8) : s;
  }

  public static String internalToPlatform(ByteString bytes) {
    return internalToPlatform(bytes.toString(ISO_8859_1));
  }

  /**
   * Reencodes a string obtained from Java stdlib functions into Bazel's internal raw byte encoding,
   * if necessary.
   */
  public static String platformToInternal(String s) {
    return needsReencodeForPlatform(s) ? new String(s.getBytes(UTF_8), ISO_8859_1) : s;
  }

  public static String platformToInternal(ByteString bytes) {
    return bytes.toString(ISO_8859_1);
  }

  public static String internalToUnicode(String s) {
    return needsReencodeForUnicode(s) ? new String(s.getBytes(ISO_8859_1), UTF_8) : s;
  }

  public static String unicodeToInternal(String s) {
    return needsReencodeForUnicode(s) ? new String(s.getBytes(UTF_8), ISO_8859_1) : s;
  }

  /**
   * Even though Bazel attempts to force the default charset to be ISO-8859-1, which makes String
   * identical to the "bag of raw bytes" that a UNIX path is, the JVM may still use a different
   * encoding for file paths and command-line arguments, e.g. on macOS (always UTF-8). When
   * interoperating with Java APIs, we thus need to reencode paths to the JVM's native encoding.
   */
  private static final Charset SUN_JNU_ENCODING =
      Charset.forName(System.getProperty("sun.jnu.encoding"));

  // This only exists for RemoteWorker, which directly uses the RE APIs UTF-8-encoded string with
  // the JavaIoFileSystem and thus shouldn't be subject to any reencoding.
  private static final boolean BAZEL_UNICODE_STRINGS =
      Boolean.getBoolean("bazel.internal.UnicodeStringss");

  private static boolean needsReencodeForPlatform(String s) {
    if (SUN_JNU_ENCODING == ISO_8859_1 && OS.getCurrent() == OS.LINUX) {
      return false;
    }
    return needsReencodeForUnicode(s);
  }

  private static boolean needsReencodeForUnicode(String s) {
    if (BAZEL_UNICODE_STRINGS) {
      return false;
    }
    return !StringUnsafe.getInstance().isAscii(s);
  }

  private StringEncoding() {}
}
