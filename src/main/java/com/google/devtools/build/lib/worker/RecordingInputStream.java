// Copyright 2017 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.lib.worker;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.base.Utf8;
import com.google.common.io.BaseEncoding;
import com.google.common.primitives.Bytes;
import java.io.ByteArrayOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * An input stream filter that records the first X bytes read from its wrapped stream.
 *
 * <p>The number bytes to record can be set via {@link #startRecording(int)}}, which also discards
 * any already recorded data. The recorded data can be retrieved via {@link
 * #getRecordedDataAsString()}.
 */
final class RecordingInputStream extends FilterInputStream {

  private static final Pattern NON_PRINTABLE_CHARS =
      Pattern.compile("[^\\p{Print}\\t\\r\\n]", Pattern.UNICODE_CHARACTER_CLASS);
  /** In hexdump output, the maximum number of lines to output. */
  private static final int MAX_HEX_LINES = 64;
  /** In hexdump output, the number of bytes that fit on one line. */
  private static final int BYTES_PER_HEX_LINE = 16;
  /** In hexdump output, the number of bytes that is grouped together in blocks. */
  private static final int BYTES_PER_HEX_BLOCK = 8;

  private ByteArrayOutputStream recordedData;
  private int maxRecordedSize;

  RecordingInputStream(InputStream in) {
    super(in);
  }

  /**
   * Returns the maximum number of bytes that can still be recorded in our buffer (but not more
   * than {@code size}).
   */
  private int getRecordableBytes(int size) {
    if (recordedData == null) {
      return 0;
    }
    return Math.min(maxRecordedSize - recordedData.size(), size);
  }

  @Override
  public int read() throws IOException {
    int bytesRead = super.read();
    if (getRecordableBytes(bytesRead) > 0) {
      recordedData.write(bytesRead);
    }
    return bytesRead;
  }

  @Override
  public int read(byte[] b) throws IOException {
    return this.read(b, 0, b.length);
  }

  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    int bytesRead = super.read(b, off, len);
    int recordableBytes = getRecordableBytes(bytesRead);
    if (recordableBytes > 0) {
      recordedData.write(b, off, recordableBytes);
    }
    return bytesRead;
  }

  public void startRecording(int maxSize) {
    recordedData = new ByteArrayOutputStream(maxSize);
    maxRecordedSize = maxSize;
  }

  /**
   * Reads whatever remaining data is available on the input stream if we still have space left in
   * the recording buffer, in order to maximize the usefulness of the recorded data for the
   * caller.
   */
  public void readRemaining() {
    try {
      byte[] dummy = new byte[getRecordableBytes(available())];
      read(dummy);
    } catch (IOException e) {
      // Ignore.
    }
  }

  /**
   * Returns the recorded data as a string, where non-printable characters are replaced with a '?'
   * symbol. Or, if the data is not UTF-8, or has non-printable chars in the start,returns hex
   * values formatted similarly to `hexdump -C`
   */
  public String getRecordedDataAsString() {
    byte[] bytes = recordedData.toByteArray();
    String input = new String(bytes, UTF_8);
    // TODO: Why do we get so much noise?
    if (Utf8.isWellFormed(bytes)
        && !NON_PRINTABLE_CHARS
            .matcher(
                input.substring(0, Math.min(input.length(), BYTES_PER_HEX_LINE * MAX_HEX_LINES)))
            .find()) {
      return NON_PRINTABLE_CHARS.matcher(input).replaceAll("?");
    } else {
      List<byte[]> chunks = new ArrayList<>(MAX_HEX_LINES);
      while (chunks.size() * BYTES_PER_HEX_LINE < bytes.length && chunks.size() < MAX_HEX_LINES) {
        chunks.add(
            Arrays.copyOfRange(
                bytes,
                chunks.size() * BYTES_PER_HEX_LINE,
                Math.min((1 + chunks.size()) * BYTES_PER_HEX_LINE, bytes.length)));
      }
      boolean isTruncated = bytes.length > BYTES_PER_HEX_LINE * MAX_HEX_LINES;
      List<String> lines = chunks.stream().map(this::formatHexLine).collect(Collectors.toList());
      return String.format(
          "Not UTF-8, printing %sas hex\n%s\n",
          (isTruncated ? "first 1024 bytes " : ""), Joiner.on('\n').join(lines));
    }
  }

  /** Formats a single array of 16 bytes as a hexdump-style line. */
  private String formatHexLine(byte[] bytes) {
    String rawHex = BaseEncoding.base16().encode(bytes);
    // Adds spaces between hex representation of each char
    String separatedHex = Joiner.on(' ').join(Splitter.fixedLength(2).split(rawHex));
    // Adds extra space between each block of 8 hex bytes (two hex chars and one space each).
    String groupedHex =
        Joiner.on(' ').join(Splitter.fixedLength(3 * BYTES_PER_HEX_BLOCK).split(separatedHex));
    // Adds ASCII-safe display of text on the right
    String textDisplay =
        Bytes.asList(bytes).stream()
            .map(b -> b >= 32 ? Character.toString((char) ((byte) b)) : ".")
            .collect(Collectors.joining());
    // Adds space in text display between blocks of 8 hex bytes.
    String splitText =
        Joiner.on(' ').join(Splitter.fixedLength(BYTES_PER_HEX_BLOCK).split(textDisplay));
    return String.format("%-50s|%-17s|", groupedHex, splitText);
  }
}
