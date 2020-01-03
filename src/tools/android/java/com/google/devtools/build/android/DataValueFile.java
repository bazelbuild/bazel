// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.hash.HashCode;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.resources.Visibility;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Represents a file based android resource or asset.
 *
 * <p>These include all resource types except those found in values, as well as all assets.
 */
public class DataValueFile implements DataResource, DataAsset {

  private final Visibility visibility;
  private final DataSource source;
  @Nullable private final HashCode fingerprint;

  private DataValueFile(Visibility visibility, DataSource source, @Nullable HashCode fingerprint) {
    this.visibility = visibility;
    this.source = source;
    this.fingerprint = fingerprint;
  }

  @Deprecated
  public static DataValueFile of(Path source) {
    return of(
        Visibility.UNKNOWN, DataSource.of(DependencyInfo.UNKNOWN, source), /*fingerprint=*/ null);
  }

  public static DataValueFile of(
      Visibility visibility, DataSource source, @Nullable HashCode fingerprint) {
    return new DataValueFile(visibility, source, fingerprint);
  }

  /** Creates a {@link DataValueFile} from a {@link SerializeFormat#DataValue}. */
  @Deprecated
  public static DataValueFile from(Path source) {
    return of(source);
  }

  @Override
  public int hashCode() {
    return source.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof DataValueFile)) {
      return false;
    }
    DataValueFile resource = (DataValueFile) obj;
    return Objects.equals(visibility, resource.visibility)
        && Objects.equals(source, resource.source)
        && Objects.equals(fingerprint, resource.fingerprint);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("source", source).toString();
  }

  @Override
  public DataSource source() {
    return source;
  }

  @Override
  public void writeAsset(RelativeAssetPath key, AndroidDataWritingVisitor mergedDataWriter)
      throws IOException {
    mergedDataWriter.copyAsset(source.getPath(), key.toPathString());
  }

  @Override
  public void writeResource(FullyQualifiedName key, AndroidDataWritingVisitor mergedDataWriter)
      throws MergingException {
    mergedDataWriter.copyResource(source.getPath(), key.toPathString(source.getPath()));
  }

  @Override
  public int serializeTo(DataSourceTable sourceTable, OutputStream output)
      throws IOException {
    SerializeFormat.DataValue.Builder builder = SerializeFormat.DataValue.newBuilder();
    SerializeFormat.DataValue value = builder.setSourceId(sourceTable.getSourceId(source)).build();
    value.writeDelimitedTo(output);
    return CodedOutputStream.computeUInt32SizeNoTag(value.getSerializedSize())
        + value.getSerializedSize();
  }

  @Override
  public DataResource combineWith(DataResource resource) {
    throw new IllegalArgumentException(getClass() + " does not combine.");
  }

  @Override
  public DataResource overwrite(DataResource resource) {
    if (equals(resource)) {
      return this;
    }
    return new DataValueFile(visibility, source.overwrite(resource.source()), fingerprint);
  }

  @Override
  public DataAsset overwrite(DataAsset asset) {
    if (equals(asset)) {
      return this;
    }
    return new DataValueFile(visibility, source.overwrite(asset.source()), fingerprint);
  }

  @Override
  public void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(source().getDependencyInfo(), visibility, key.type(), key.name());
  }

  @Override
  public DataValue update(DataSource source) {
    return new DataValueFile(visibility, source, fingerprint);
  }

  @Override
  public String asConflictString() {
    return source.asConflictString();
  }

  @Override
  public boolean valueEquals(DataValue value) {
    if (!(value instanceof DataValueFile)) {
      return false;
    }
    DataValueFile other = (DataValueFile) value;
    if (!Objects.equals(visibility, other.visibility)) {
      return false;
    }
    // Just check the path, ignoring other components of DataSource.  Build label is not
    // relevant to whether something is a conflict.
    if (Objects.equals(source.getPath(), other.source.getPath())) {
      return true;
    }
    // fingerprint==null && other.fingerprint==null shouldn't count as equality.
    if (fingerprint != null && Objects.equals(fingerprint, other.fingerprint)) {
      return true;
    }
    return false;
  }

  @Override
  public int compareMergePriorityTo(DataValue value) {
    return 0;
  }
}
