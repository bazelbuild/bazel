// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;

/**
 * A root path used in {@link RootedPath} and in artifact roots.
 *
 * <p>A typical root could be the exec path, a package root, or an output root specific to some
 * configuration.
 */
public interface Root extends Comparable<Root>, Serializable {

  static ObjectCodec<Root> getCodec(PathCodec pathCodec) {
    return new RootCodec(pathCodec);
  }

  /** Constructs a root from a path. */
  static Root fromPath(Path path) {
    return new PathRoot(path);
  }

  /** Returns a root from the file system's root directory. */
  static Root fromFileSystemRoot(FileSystem fileSystem) {
    return fileSystem.getRoot();
  }

  /** Returns a path by concatenating the root and the root-relative path. */
  Path getRelative(PathFragment relativePath);

  /** Returns a path by concatenating the root and the root-relative path. */
  Path getRelative(String relativePath);

  /** Returns the relative path between the root and the given path. */
  PathFragment relativize(Path path);

  @Deprecated
  PathFragment relativize(PathFragment relativePath);

  /** Returns whether the given path is under this root. */
  boolean contains(Path path);

  @Deprecated
  boolean contains(PathFragment relativePath);

  /**
   * Returns the underlying path. Please avoid using this method. Roots may eventually not be
   * directly backed by paths.
   */
  Path asPath();

  @Deprecated
  boolean isRootDirectory();

  /** Implementation of Root that is backed by a {@link Path}. */
  final class PathRoot implements Root {
    private final Path path;

    PathRoot(Path path) {
      this.path = path;
    }

    @Override
    public Path getRelative(PathFragment relativePath) {
      return path.getRelative(relativePath);
    }

    @Override
    public Path getRelative(String relativePath) {
      return path.getRelative(relativePath);
    }

    @Override
    public PathFragment relativize(Path path) {
      return path.relativeTo(this.path);
    }

    @Override
    public PathFragment relativize(PathFragment relativePath) {
      Preconditions.checkArgument(relativePath.isAbsolute());
      return relativePath.relativeTo(path.asFragment());
    }

    @Override
    public boolean contains(Path path) {
      return path.startsWith(this.path);
    }

    @Override
    public boolean contains(PathFragment relativePath) {
      return relativePath.isAbsolute() && relativePath.startsWith(path.asFragment());
    }

    @Override
    public Path asPath() {
      return path;
    }

    @Override
    public boolean isRootDirectory() {
      return path.isRootDirectory();
    }

    @Override
    public String toString() {
      return path.toString();
    }

    @Override
    public int compareTo(Root o) {
      return path.compareTo(((PathRoot) o).path);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      PathRoot pathRoot = (PathRoot) o;
      return path.equals(pathRoot.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }
  }

  /** Codec to serialize {@link Root}s. */
  class RootCodec implements ObjectCodec<Root> {
    private final PathCodec pathCodec;

    private RootCodec(PathCodec pathCodec) {
      this.pathCodec = pathCodec;
    }

    @Override
    public Class<Root> getEncodedClass() {
      return Root.class;
    }

    @Override
    public void serialize(Root obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      if (obj instanceof PathRoot) {
        pathCodec.serialize(((PathRoot) obj).path, codedOut);
      } else {
        throw new AssertionError("Unknown Root subclass: " + obj.getClass().getName());
      }
    }

    @Override
    public Root deserialize(CodedInputStream codedIn) throws SerializationException, IOException {
      Path path = pathCodec.deserialize(codedIn);
      return new PathRoot(path);
    }
  }
}
