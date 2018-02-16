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
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableSet;
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

  public static DataSource of(Path sourcePath) {
    return new DataSource(sourcePath, ImmutableSet.<DataSource>of());
  }

  private final Path path;
  private final ImmutableSet<DataSource> overrides;

  private DataSource(Path path, ImmutableSet<DataSource> overrides) {
    this.path = path;
    this.overrides = overrides;
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
    return Objects.equal(path, ((DataSource) o).path)
        && Objects.equal(overrides, ((DataSource) o).overrides);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(path, overrides);
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

  public DataSource overwrite(DataSource... sources) {
    ImmutableSet.Builder<DataSource> overridesBuilder =
        ImmutableSet.<DataSource>builder().addAll(this.overrides);
    for (DataSource dataSource : sources) {
      // A DataSource cannot overwrite itself.
      // This will be an error once the depot can be assured not have source files.
      if (!dataSource.path.equals(path)) {
        // Flatten the DataSource to a placeholder to avoid building trees, which end up being
        // expensive, slow, and hard to reason about.
        overridesBuilder.add(of(dataSource.path));
      }
      overridesBuilder.addAll(dataSource.overrides);
    }
    return new DataSource(path, overridesBuilder.build());
  }

  public ImmutableSet<DataSource> overrides() {
    return overrides;
  }

  public boolean isInValuesFolder() {
    return path.getParent().getFileName().toString().startsWith(SdkConstants.FD_RES_VALUES);
  }

  public boolean hasOveridden(DataSource source) {
    return overrides.contains(source);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("path", path)
        .add("overrides", overrides)
        .toString();
  }

  /** Returns a representation suitible for a conflict message. */
  public String asConflictString() {
    return path.toString();
  }
}
