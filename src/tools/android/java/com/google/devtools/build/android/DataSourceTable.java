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

import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import com.google.devtools.build.android.proto.SerializeFormat.ProtoSource;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.NavigableMap;

/**
 * Tracks mappings from resource source paths (/foo/bar/res/values/colors.xml) to an ID for a more
 * compact serialization format.
 */
class DataSourceTable {

  private final Map<Path, Integer> sourceTable = new HashMap<>();
  private Path[] idToSource;

  /**
   * Creates a DataSourceTable and serialize to the given outstream. Assigns each resource source
   * path a number to enable {@link #getSourceId(Path)} queries.
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
  public int getSourceId(Path source) {
    return sourceTable.get(source);
  }

  private void writeSourceInfo(NavigableMap<DataKey, DataValue> map, OutputStream outStream)
      throws IOException {
    int sourceNumber = 0;
    for (Map.Entry<DataKey, DataValue> entry : map.entrySet()) {
      Path source = entry.getValue().source();
      if (!sourceTable.containsKey(source)) {
        sourceTable.put(source, sourceNumber);
        ++sourceNumber;
        ProtoSource.newBuilder().setFilename(source.toString()).build().writeDelimitedTo(outStream);
      }
    }
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
  public Path sourceFromId(int sourceId) {
    return idToSource[sourceId];
  }

  private void readSourceInfo(InputStream in, FileSystem currentFileSystem, Header header)
      throws IOException {
    int numberOfSources = header.getSourceCount();
    // Read back the sources.
    idToSource = new Path[numberOfSources];
    for (int i = 0; i < numberOfSources; i++) {
      ProtoSource protoSource = SerializeFormat.ProtoSource.parseDelimitedFrom(in);
      Path source = currentFileSystem.getPath(protoSource.getFilename());
      idToSource[i] = source;
    }
  }
}
