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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValue.Builder;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import com.google.devtools.build.android.xml.Namespaces;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/** Serializes {@link DataKey},{@link DataValue} entries to a binary file. */
public class AndroidDataSerializer {

  /** A visitor to accumulate the necessary state to write a resource entry. */
  public interface SerializeEntryVisitor extends Writeable {
    SerializeEntryVisitor setSource(DataSource dataSource);

    SerializeEntryVisitor setKey(DataKey key);

    SerializeEntryVisitor overwrite(Set<DataSource> dataSource);

    SerializeEntryVisitor setXml(XmlResourceValue value);

    SerializeEntryVisitor setNamespaces(Namespaces namespaces);
  }

  private static class DataValueBuilder implements SerializeEntryVisitor {

    private final Builder builder;
    private final WritablePool<DataKey> keys;
    private final WritablePool<DataSource> sources;
    private final WritablePool<XmlResourceValue> xml;
    private final WritablePool<Namespaces> namespaces;

    public DataValueBuilder(
        Builder builder,
        WritablePool<DataKey> keys,
        WritablePool<DataSource> sources,
        WritablePool<XmlResourceValue> xml,
        WritablePool<Namespaces> namespaces) {
      this.builder = Preconditions.checkNotNull(builder);
      this.keys = Preconditions.checkNotNull(keys);
      this.sources = sources;
      this.xml = xml;
      this.namespaces = namespaces;
    }

    static DataValueBuilder create(
        WritablePool<DataKey> keys,
        WritablePool<DataSource> sources,
        WritablePool<XmlResourceValue> xml,
        WritablePool<Namespaces> namespaces) {
      return new DataValueBuilder(
          SerializeFormat.DataValue.newBuilder(), keys, sources, xml, namespaces);
    }

    @Override
    public SerializeEntryVisitor setSource(DataSource dataSource) {
      builder.setSourceId(sources.queue(dataSource));
      overwrite(dataSource.overrides());
      return this;
    }

    @Override
    public SerializeEntryVisitor setKey(DataKey key) {
      builder.setKeyId(keys.queue(key));
      return this;
    }

    @Override
    public SerializeEntryVisitor overwrite(Set<DataSource> dataSource) {
      for (DataSource source : dataSource) {
        builder.addOverwrittenSourceId(sources.queue(source));
      }
      return this;
    }

    @Override
    public SerializeEntryVisitor setXml(XmlResourceValue value) {
      builder.setXmlId(xml.queue(value));
      return this;
    }

    @Override
    public SerializeEntryVisitor setNamespaces(Namespaces namespaces) {
      builder.setNamespaceId(this.namespaces.queue(namespaces));
      return this;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      builder.build().writeDelimitedTo(out);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("builder", builder.build())
          .add("keys", keys)
          .add("sources", sources)
          .add("xml", xml)
          .add("namespaces", namespaces)
          .toString();
    }
  }

  private static class WritablePool<T extends Writeable> implements Writeable {
    BiMap<T, Integer> lookup = HashBiMap.create();
    Integer lastIndex = 0;

    Integer queue(T value) {
      if (!lookup.containsKey(value)) {
        lookup.put(value, lastIndex++);
      }
      return lookup.get(value);
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      final BiMap<Integer, T> indexed = lookup.inverse();
      for (int idx = 0; idx < lastIndex; idx++) {
        indexed.get(idx).writeTo(out);
      }
    }

    public int size() {
      return lookup.size();
    }

    public static <T extends Writeable> WritablePool<T> create() {
      return new WritablePool<>();
    }
  }

  private static final Logger logger = Logger.getLogger(AndroidDataSerializer.class.getName());

  private final NavigableMap<DataKey, DataValue> entries = new TreeMap<>();

  public static AndroidDataSerializer create() {
    return new AndroidDataSerializer();
  }

  private AndroidDataSerializer() {}

  /**
   * Writes all of the collected DataKey -> DataValue.
   *
   * <p>The binary format will be:
   *
   * <pre>
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
    WritablePool<DataKey> keys = WritablePool.create();
    WritablePool<DataSource> sources = WritablePool.create();
    WritablePool<XmlResourceValue> xml = WritablePool.create();
    WritablePool<Namespaces> namespaces = WritablePool.create();

    try (OutputStream outStream =
        new BufferedOutputStream(
            Files.newOutputStream(out, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE))) {

      // Set the header for the deserialization process.

      List<SerializeEntryVisitor> values = new ArrayList<>(entries.size());
      for (Entry<DataKey, DataValue> entry : entries.entrySet()) {
        values.add(
            entry
                .getValue()
                .serializeTo(
                    DataValueBuilder.create(keys, sources, xml, namespaces)
                        .setKey(entry.getKey())));
      }

      Header.newBuilder()
          .setKeyCount(keys.size())
          .setSourceCount(sources.size())
          .setValueCount(values.size())
          .setXmlCount(xml.size())
          .setNamespacesCount(namespaces.size())
          .build()
          .writeDelimitedTo(outStream);

      /*
      Serialization order:
        keys
        sources
        values
        xml
        namespaces

      This allows deserializing keys for R generation, as well as sources and entries for
      lightweight merging.
      */

      keys.writeTo(outStream);
      sources.writeTo(outStream);
      for (SerializeEntryVisitor value : values) {
        value.writeTo(outStream);
      }
      xml.writeTo(outStream);
      namespaces.writeTo(outStream);

      outStream.flush();
    }
    logger.fine(String.format("Serialized merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
  }

  /** Queues the key and value for serialization as a entries entry. */
  public void queueForSerialization(DataKey key, DataValue value) {
    entries.put(key, value);
  }
}
