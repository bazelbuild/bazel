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

import com.android.SdkConstants;
import com.google.common.base.Objects;
import com.google.devtools.build.android.proto.SerializeFormat.ProtoSource;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributeView;

/** Represents where the DataValue was derived from. */
public class DataSource implements Comparable<DataSource> {

  public static DataSource from(ProtoSource protoSource, FileSystem currentFileSystem) {
    Path path = currentFileSystem.getPath(protoSource.getFilename());
    return of(path);
  }

  public static DataSource of(Path source) {
    return new DataSource(source);
  }

  private final Path path;

  private DataSource(Path path) {
    this.path = path;
  }

  public Path getPath() {
    return path;
  }

  public long getFileSize() throws IOException {
    return Files.getFileAttributeView(path, BasicFileAttributeView.class).readAttributes().size();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    return Objects.equal(path, ((DataSource) o).path);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(path);
  }

  @Override
  public int compareTo(DataSource o) {
    return path.compareTo(o.path);
  }

  public InputStream newBufferedInputStream() throws IOException {
    return new BufferedInputStream(Files.newInputStream(path));
  }

  /** Selects which DataSource should be considered the authoritative source of a value. */
  // TODO(corysmith): Combine the sources so that we know both of the originating libraries.
  // For now, prefer sources that have explicit definitions (values/ and not layout/), since the
  // values are ultimately written out to a merged values.xml. Sources from layout/menu, etc.
  // can come from "@+id" definitions.
  public DataSource combine(DataSource otherSource) {
    boolean thisInValuesFolder = isInValuesFolder();
    boolean otherInValuesFolder = otherSource.isInValuesFolder();
    if (thisInValuesFolder && !otherInValuesFolder) {
      return this;
    }
    if (!thisInValuesFolder && otherInValuesFolder) {
      return otherSource;
    }
    return this;
  }

  public boolean isInValuesFolder() {
    return path.getParent().getFileName().toString().startsWith(SdkConstants.FD_RES_VALUES);
  }
}
