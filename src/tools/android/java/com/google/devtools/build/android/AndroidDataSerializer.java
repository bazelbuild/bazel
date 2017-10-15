// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.base.Stopwatch;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Serializes {@link DataKey},{@link DataValue} entries to a binary file.
 */
public class AndroidDataSerializer {
  private static final Logger logger = Logger.getLogger(AndroidDataSerializer.class.getName());

  private final NavigableMap<DataKey, DataValue> entries = new TreeMap<>();

  public static AndroidDataSerializer create() {
    return new AndroidDataSerializer();
  }

  private AndroidDataSerializer() {}

  /**
   * Writes all of the collected DataKey -> DataValue.
   *
   * The binary format will be: <pre>
   * {@link Header}
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataKey} keys...
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataValue} entries...
   * </pre>
   *
   * The key and values will be written in comparable order, allowing for the optimization of not
   * converting the DataValue from binary, only writing it into a merged serialized binary.
   */
  public void flushTo(Path out) throws IOException {
    Stopwatch timer = Stopwatch.createStarted();
    // Ensure the parent directory exists, if any.
    if (out.getParent() != null) {
      Files.createDirectories(out.getParent());
    }
    try (OutputStream outStream =
        new BufferedOutputStream(
            Files.newOutputStream(out, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE))) {

      // Set the header for the deserialization process.
      SerializeFormat.Header.Builder headerBuilder = Header.newBuilder()
          .setEntryCount(entries.size());

      // Create table of source paths to allow references in the serialization format via an index.
      ByteArrayOutputStream sourceTableOutputStream = new ByteArrayOutputStream(2048);
      DataSourceTable sourceTable =
          DataSourceTable.createAndWrite(entries, sourceTableOutputStream, headerBuilder);

      headerBuilder
          .build()
          .writeDelimitedTo(outStream);

      writeKeyValuesTo(entries, outStream, sourceTable, sourceTableOutputStream.toByteArray());
    }
    logger.fine(
        String.format("Serialized merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }

  private void writeKeyValuesTo(
      NavigableMap<DataKey, DataValue> map,
      OutputStream outStream,
      DataSourceTable sourceTable,
      byte[] sourceTableBytes)
      throws IOException {
    Set<Entry<DataKey, DataValue>> entries = map.entrySet();
    int[] orderedValueSizes = new int[entries.size()];
    int valueSizeIndex = 0;
    // Serialize all the values in sorted order to a intermediate buffer, so that the keys
    // can be associated with a value size.
    // TODO(corysmith): Tune the size of the byte array.
    ByteArrayOutputStream valuesOutputStream = new ByteArrayOutputStream(2048);
    for (Map.Entry<DataKey, DataValue> entry : entries) {
      orderedValueSizes[valueSizeIndex++] = entry.getValue()
          .serializeTo(entry.getKey(), sourceTable, valuesOutputStream);
    }
    // Serialize all the keys in sorted order
    valueSizeIndex = 0;
    for (Map.Entry<DataKey, DataValue> entry : entries) {
      entry.getKey().serializeTo(outStream, orderedValueSizes[valueSizeIndex++]);
    }
    // write the source table
    outStream.write(sourceTableBytes);
    // write the values to the output stream.
    outStream.write(valuesOutputStream.toByteArray());
  }

  /** Queues the key and value for serialization as a entries entry. */
  public void queueForSerialization(DataKey key, DataValue value) {
    entries.put(key, value);
  }
}
