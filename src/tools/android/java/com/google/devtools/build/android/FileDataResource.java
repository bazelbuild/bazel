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
import com.google.devtools.build.android.FullyQualifiedName.Factory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Represents a file based android resource.
 *
 * These include all resource types except those found in values.
 */
public class FileDataResource implements DataResource {

  private FullyQualifiedName qn;
  private Path source;

  public FileDataResource(FullyQualifiedName qn, Path source) {
    this.qn = qn;
    this.source = source;
  }

  public static DataResource of(FullyQualifiedName qn, Path source) {
    return new FileDataResource(qn, source);
  }

  @Override
  public int hashCode() {
    return Objects.hash(qn, source);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FileDataResource)) {
      return false;
    }
    FileDataResource resource = (FileDataResource) obj;
    return Objects.equals(qn, resource.qn) && Objects.equals(source, resource.source);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("qn", qn).add("source", source).toString();
  }

  static DataResource fromPath(Path path, Factory fqnFactory) {
    if (path.getNameCount() < 2) {
      throw new IllegalArgumentException(
          String.format(
              "The resource path %s is too short. "
                  + "The path is expected to be <resource type>/<file name>.",
              path));
    }
    String rawFqn =
        removeFileExtension(path.subpath(path.getNameCount() - 2, path.getNameCount()).toString());
    return of(fqnFactory.parse(rawFqn), path);
  }

  private static String removeFileExtension(String pathWithExtension) {
    int extensionStart = pathWithExtension.lastIndexOf('.');
    if (extensionStart > 0) {
      return pathWithExtension.substring(0, extensionStart);
    }
    return pathWithExtension;
  }

  @Override
  public void write(Path newResourceDirectory) throws IOException {
    // TODO(corysmith): Implement the copy semantics.
    throw new UnsupportedOperationException();
  }
}
