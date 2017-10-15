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

import java.io.ByteArrayOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.regex.Pattern;

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
    int bytesRead = super.read(b);
    int recordableBytes = getRecordableBytes(bytesRead);
    if (recordableBytes > 0) {
      recordedData.write(b, 0, recordableBytes);
    }
    return bytesRead;
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
   * symbol.
   */
  public String getRecordedDataAsString() {
    String recordedString = new String(recordedData.toByteArray(), UTF_8);
    return NON_PRINTABLE_CHARS.matcher(recordedString).replaceAll("?").trim();
  }
}
