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

import com.google.common.base.Preconditions;
import com.google.common.collect.Ordering;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import com.google.devtools.build.android.proto.SerializeFormat.ProtoSource;
import com.google.protobuf.InvalidProtocolBufferException;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.NavigableSet;
import java.util.TreeMap;

/**
 * Writes an {@link UnwrittenMergedAndroidData} to a binary file.
 */
public class AndroidDataSerializer {

  // TODO(corysmith): We might need a more performant comparasion methodology than toString.
  private final NavigableMap<DataKey, DataValue> primary =
      new TreeMap<DataKey, DataValue>(Ordering.usingToString());

  private final NavigableMap<DataKey, DataValue> deps =
      new TreeMap<DataKey, DataValue>(Ordering.usingToString());
  private Path manifest;

  public static AndroidDataSerializer create() {
    return new AndroidDataSerializer();
  }

  private AndroidDataSerializer() {}

  /**
   * Writes all of the collected DataKey -> DataValue.
   *
   * The binary format will be:
   * <pre>
   * {@link Header}
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataKey} primary...
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataValue} primary...
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataKey} transitive...
   * {@link com.google.devtools.build.android.proto.SerializeFormat.DataValue} transitive...
   * </pre>
   *
   * The key and values will be written in comparable order, allowing for the optimization of not
   * converting the DataValue from binary, only writing it into a merged serialized binary.
   */
  public void flushTo(Path out) throws IOException {
    Preconditions.checkNotNull(manifest, "Manifest is required to serialize AndroidData.");
    try (OutputStream outStream =
        new BufferedOutputStream(
            Files.newOutputStream(out, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE))) {
      // Set the header for the deserialization process.
      Header.newBuilder()
          .setPrimaryEntries(primary.size())
          .setTransitiveEntries(deps.size())
          .setManifestPath(ProtoSource.newBuilder().setFilename(manifest.toString()))
          .build()
          .writeDelimitedTo(outStream);

      writeKeyValuesTo(primary, outStream);
      writeKeyValuesTo(deps, outStream);
    }
  }

  private void writeKeyValuesTo(NavigableMap<DataKey, DataValue> map, OutputStream outStream)
      throws IOException {
    NavigableSet<DataKey> keys = map.navigableKeySet();
    int[] orderedValueSizes = new int[keys.size()];
    int valueSizeIndex = 0;
    // Serialize all the values in sorted order to a intermediate buffer, so that the keys
    // can be associated with a value size.
    // TODO(corysmith): Tune the size of the byte array.
    ByteArrayOutputStream valuesOutputStream = new ByteArrayOutputStream(2048);
    for (DataKey key : keys) {
      orderedValueSizes[valueSizeIndex++] = map.get(key).serializeTo(key, valuesOutputStream);
    }
    // Serialize all the keys in sorted order
    valueSizeIndex = 0;
    for (DataKey key : map.navigableKeySet()) {
      key.serializeTo(outStream, orderedValueSizes[valueSizeIndex]);
    }
    // write the values to the output stream.
    outStream.write(valuesOutputStream.toByteArray());
  }

  /**
   * Reads the serialized {@link DataKey} and {@link DataValue} to the {@link KeyValueConsumers}.
   *
   * @param inPath The path to the serialized protocol buffer.
   * @param primaryConsumers The {@link KeyValueConsumers} for the primary {@link DataKey} ->
   * {@link DataValue} entries.
   * @param transitiveConsumers The {@link KeyValueConsumers} for the transitive {@link DataKey} ->
   * {@link DataValue} entries.
   * @returns The manifest path for the serialized data.
   * @throws IOException when io operations fail.
   * @throws InvalidProtocolBufferException When the inPath is not a valid proto buffer -- e.g.
   * legacy R.txts generated from android_resources.
   */
  public Path read(
      Path inPath, KeyValueConsumers primaryConsumers, KeyValueConsumers transitiveConsumers)
      throws IOException, InvalidProtocolBufferException {
    try (InputStream in = Files.newInputStream(inPath, StandardOpenOption.READ)) {
      FileSystem currentFileSystem = inPath.getFileSystem();
      Header header = Header.parseDelimitedFrom(in);
      readEntriesSegment(primaryConsumers, in, currentFileSystem, header.getPrimaryEntries());
      readEntriesSegment(transitiveConsumers, in, currentFileSystem, header.getTransitiveEntries());
      // Assuming that the inPath will be the same filesystem as the manifest.
      // It seems highly unlikely they would be different.
      Path manifestPath = inPath.getFileSystem().getPath(header.getManifestPath().getFilename());
      Preconditions.checkArgument(
          Files.exists(manifestPath), "The manifest %s, does not exist.", manifestPath);
      return manifestPath;
    }
  }

  private void readEntriesSegment(
      KeyValueConsumers consumers,
      InputStream in,
      FileSystem currentFileSystem,
      int numberOfEntries)
      throws IOException, InvalidProtocolBufferException {
    Map<DataKey, KeyValueConsumer<DataKey, ? extends DataValue>> keys = new LinkedHashMap<>();
    for (int i = 0; i < numberOfEntries; i++) {
      SerializeFormat.DataKey protoKey = SerializeFormat.DataKey.parseDelimitedFrom(in);
      if (protoKey.hasResourceType()) {
        FullyQualifiedName resourceName = FullyQualifiedName.fromProto(protoKey);
        keys.put(
            resourceName,
            FullyQualifiedName.isOverwritable(resourceName)
                ? consumers.overwritingConsumer
                : consumers.nonOverwritingConsumer);
      } else {
        keys.put(RelativeAssetPath.fromProto(protoKey, currentFileSystem), consumers.assetConsumer);
      }
    }

    // TODO(corysmith): Make this a lazy read of the values.
    for (Entry<DataKey, KeyValueConsumer<DataKey, ?>> entry : keys.entrySet()) {
      SerializeFormat.DataValue protoValue = SerializeFormat.DataValue.parseDelimitedFrom(in);
      if (protoValue.hasXmlValue()) {
        // TODO(corysmith): Figure out why the generics are wrong.
        // If I use Map<DataKey, KeyValueConsumer<DataKey, ? extends DataValue>>, I can put
        // consumers into the map, but I can't call consume.
        // If I use Map<DataKey, KeyValueConsumer<DataKey, ? super DataValue>>, I can consume
        // but I can't put.
        // Same for below.
        @SuppressWarnings("unchecked")
        KeyValueConsumer<DataKey, DataValue> value =
            (KeyValueConsumer<DataKey, DataValue>) entry.getValue();
        value.consume(entry.getKey(), DataResourceXml.from(protoValue, currentFileSystem));
      } else {
        @SuppressWarnings("unchecked")
        KeyValueConsumer<DataKey, DataValue> value =
            (KeyValueConsumer<DataKey, DataValue>) entry.getValue();
        value.consume(entry.getKey(), DataValueFile.from(protoValue, currentFileSystem));
      }
    }
  }

  /** Queues the manifest for serialization. */
  public void serializeManifest(Path manifest) {
    this.manifest = manifest;
  }

  /** Queues the key and value for serialization as a primary entry. */
  public void serializeToPrimary(DataKey key, DataValue value) {
    primary.put(key, value);
  }

  /** Queues the key and value for serialization as a transitive entry. */
  public void serializeToTransitive(DataKey key, DataValue value) {
    deps.put(key, value);
  }
}
