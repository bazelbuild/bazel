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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import com.google.devtools.build.android.proto.SerializeFormat.ProtoSource;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.stream.Collectors;

/**
 * Tracks mappings from resource source paths (/foo/bar/res/values/colors.xml) to an ID for a more
 * compact serialization format.
 */
class DataSourceTable {

  private static final Function<DataValue, DataSource> VALUE_TO_SOURCE = DataValue::source;
  private final Map<DataSource, Integer> sourceTable = new LinkedHashMap<>();
  private DataSource[] idToSource;

  /**
   * Creates a DataSourceTable and serialize to the given outstream. Assigns each resource source
   * path a number to enable {@link #getSourceId(DataSource)} queries.
   *
   * @param map the final map of resources
   * @param outStream stream to serialize the source table
   * @param headerBuilder the header to serialize
   * @throws IOException if this fails to serialize the table to the outStream
   */
  public static DataSourceTable createAndWrite(
      NavigableMap<DataKey, DataValue> map, OutputStream outStream, Header.Builder headerBuilder)
      throws IOException {
    DataSourceTable sourceTable = new DataSourceTable();
    sourceTable.writeSourceInfo(map, outStream);
    sourceTable.setHeader(headerBuilder);
    return sourceTable;
  }

  /** Convert the absolute source path to the source table index */
  public int getSourceId(DataSource source) {
    return sourceTable.get(source);
  }

  private void writeSourceInfo(NavigableMap<DataKey, DataValue> map, OutputStream outStream)
      throws IOException {
    int sourceNumber = 0;
    LinkedList<DataSource> sourceQueue =
        map.values()
            .stream()
            .map(VALUE_TO_SOURCE)
            .collect(Collectors.toCollection(LinkedList::new));
    while (!sourceQueue.isEmpty()) {
      DataSource source = sourceQueue.pop();
      if (!sourceTable.containsKey(source)) {
        sourceTable.put(source, sourceNumber);
        ++sourceNumber;
        sourceQueue.addAll(source.overrides());
      }
    }
    for (DataSource dataSource : sourceTable.keySet()) {
      ProtoSource.newBuilder()
          .setFilename(dataSource.getPath().toString())
          .addAllOverwritten(sourcesToIds(dataSource.overrides()))
          .build()
          .writeDelimitedTo(outStream);
    }
  }

  private List<Integer> sourcesToIds(ImmutableSet<DataSource> overrides) {
    ImmutableList.Builder<Integer> idsBuilder = ImmutableList.builder();
    for (DataSource dataSource : overrides) {
      if (!sourceTable.containsKey(dataSource)) {
        throw new IllegalArgumentException(
            "Cannot find data source: " + dataSource.toString() + " in " + sourceTable.keySet());
      }
      idsBuilder.add(sourceTable.get(dataSource));
    }
    return idsBuilder.build();
  }

  /** Fill in the serialize format header information required to deserialize */
  private Header.Builder setHeader(Header.Builder headerBuilder) {
    return headerBuilder.setSourceCount(sourceTable.size());
  }

  /** Deserialize the source table and allow {@link #sourceFromId(int)} queries. */
  public static DataSourceTable read(InputStream in, FileSystem currentFileSystem, Header header)
      throws IOException {
    DataSourceTable sourceTable = new DataSourceTable();
    sourceTable.readSourceInfo(in, currentFileSystem, header);
    return sourceTable;
  }

  /** Convert the source ID to full Path */
  public DataSource sourceFromId(int sourceId) {
    return idToSource[sourceId];
  }

  private void readSourceInfo(InputStream in, FileSystem currentFileSystem, Header header)
      throws IOException {
    int numberOfSources = header.getSourceCount();
    // Read back the sources.
    idToSource = new DataSource[numberOfSources];
    for (int i = 0; i < numberOfSources; i++) {
      ProtoSource protoSource = SerializeFormat.ProtoSource.parseDelimitedFrom(in);
      idToSource[i] = DataSource.from(protoSource, currentFileSystem);
    }
  }
}
