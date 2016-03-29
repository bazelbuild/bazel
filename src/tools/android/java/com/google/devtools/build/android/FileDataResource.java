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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Represents a file based android resource or asset.
 *
 * These include all resource types except those found in values, as well as all assets.
 */
public class FileDataResource implements DataResource, DataAsset {

  private final Path source;

  private FileDataResource(Path source) {
    this.source = source;
  }

  public static FileDataResource of(Path source) {
    return new FileDataResource(source);
  }

  @Override
  public int hashCode() {
    return source.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FileDataResource)) {
      return false;
    }
    FileDataResource resource = (FileDataResource) obj;
    return Objects.equals(source, resource.source);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("source", source)
        .toString();
  }

  @Override
  public Path source() {
    return source;
  }

  @Override
  public void write(Path newResourceDirectory) throws IOException {
    // TODO(corysmith): Implement the copy semantics.
    throw new UnsupportedOperationException();
  }
}
