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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.AsyncObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.DynamicCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A root path used in {@link RootedPath} and in artifact roots.
 *
 * <p>A typical root could be the exec path, a package root, or an output root specific to some
 * configuration. We also support absolute roots for non-hermetic paths outside the user workspace.
 */
public abstract class Root implements Comparable<Root> {

  /** Constructs a root from a path. */
  public static Root fromPath(Path path) {
    return new PathRoot(path);
  }

  /** Returns an absolute root. Can only be used with absolute path fragments. */
  public static Root absoluteRoot(FileSystem fileSystem) {
    return fileSystem.getAbsoluteRoot();
  }

  public static Root toFileSystem(Root root, FileSystem fileSystem) {
    return root.isAbsolute()
      ? new AbsoluteRoot(fileSystem)
      : new PathRoot(fileSystem.getPath(root.asPath().asFragment()));
  }

  /** Returns a path by concatenating the root and the root-relative path. */
  public abstract Path getRelative(PathFragment rootRelativePath);

  /** Returns a path by concatenating the root and the root-relative path. */
  public abstract Path getRelative(String rootRelativePath);

  /** Returns the relative path between the root and the given path. */
  public abstract PathFragment relativize(Path path);

  /** Returns the relative path between the root and the given absolute path fragment. */
  public abstract PathFragment relativize(PathFragment absolutePathFragment);

  /** Returns whether the given path is under this root. */
  public abstract boolean contains(Path path);

  /** Returns whether the given absolute path fragment is under this root. */
  public abstract boolean contains(PathFragment absolutePathFragment);

  /**
   * Returns the underlying path. Please avoid using this method.
   *
   * <p>Not all roots are backed by paths, so this may return null.
   */
  @Nullable
  public abstract Path asPath();

  /** Returns the underlying FileSystem this Root is on. */
  public abstract FileSystem getFileSystem();

  public abstract boolean isAbsolute();

  /** Implementation of Root that is backed by a {@link Path}. */
  public static final class PathRoot extends Root {
    private final Path path;

    private PathRoot(Path path) {
      this.path = path;
    }

    @Override
    public Path getRelative(PathFragment rootRelativePath) {
      return path.getRelative(rootRelativePath);
    }

    @Override
    public Path getRelative(String rootRelativePath) {
      return path.getRelative(rootRelativePath);
    }

    @Override
    public PathFragment relativize(Path path) {
      return path.relativeTo(this.path);
    }

    @Override
    public PathFragment relativize(PathFragment absolutePathFragment) {
      Preconditions.checkArgument(absolutePathFragment.isAbsolute());
      return absolutePathFragment.relativeTo(path.asFragment());
    }

    @Override
    public boolean contains(Path path) {
      return path.startsWith(this.path);
    }

    @Override
    public boolean contains(PathFragment absolutePathFragment) {
      return absolutePathFragment.isAbsolute()
          && absolutePathFragment.startsWith(path.asFragment());
    }

    @Override
    public Path asPath() {
      return path;
    }

    @Override
    public FileSystem getFileSystem() {
      return path.getFileSystem();
    }

    @Override
    public boolean isAbsolute() {
      return false;
    }

    @Override
    public String toString() {
      return path.toString();
    }

    @Override
    public int compareTo(Root o) {
      if (o instanceof AbsoluteRoot) {
        return 1;
      } else if (o instanceof PathRoot pathRoot) {
        return path.compareTo(pathRoot.path);
      } else {
        throw new AssertionError("Unknown Root subclass: " + o.getClass().getName());
      }
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

  /** An absolute root of a file system. Can only resolve absolute path fragments. */
  public static final class AbsoluteRoot extends Root {
    private final FileSystem fileSystem;

    AbsoluteRoot(FileSystem fileSystem) {
      this.fileSystem = Preconditions.checkNotNull(fileSystem);
    }

    @Override
    public Path getRelative(PathFragment rootRelativePath) {
      Preconditions.checkArgument(rootRelativePath.isAbsolute());
      return fileSystem.getPath(rootRelativePath);
    }

    @Override
    public Path getRelative(String rootRelativePath) {
      return getRelative(PathFragment.create(rootRelativePath));
    }

    @Override
    public PathFragment relativize(Path path) {
      return path.asFragment();
    }

    @Override
    public PathFragment relativize(PathFragment absolutePathFragment) {
      Preconditions.checkArgument(absolutePathFragment.isAbsolute());
      return absolutePathFragment;
    }

    @Override
    public boolean contains(Path path) {
      return true;
    }

    @Override
    public boolean contains(PathFragment absolutePathFragment) {
      return absolutePathFragment.isAbsolute();
    }

    @Override
    public boolean isAbsolute() {
      return true;
    }

    @Nullable
    @Override
    public Path asPath() {
      return null;
    }

    @Override
    public FileSystem getFileSystem() {
      return fileSystem;
    }

    @Override
    public String toString() {
      return "<absolute root>";
    }

    @Override
    public int compareTo(Root o) {
      if (o instanceof AbsoluteRoot) {
        return Integer.compare(hashCode(), o.hashCode());
      } else if (o instanceof PathRoot) {
        return -1;
      } else {
        throw new AssertionError("Unknown Root subclass: " + o.getClass().getName());
      }
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof AbsoluteRoot)) {
        return false;
      }
      AbsoluteRoot that = (AbsoluteRoot) o;
      return fileSystem.equals(that.fileSystem);
    }

    @Override
    public int hashCode() {
      return 31 + fileSystem.hashCode();
    }
  }

  /** Serialization dependencies for {@link RootCodec}. */
  public static class RootCodecDependencies {
    private final ImmutableList<Root> likelyPopularRoots;

    /** Convenience constructor for an instance with no likely roots. */
    public RootCodecDependencies() {
      this(ImmutableList.of());
    }

    /** Convenience constructor for an instance with one likely root. */
    public RootCodecDependencies(Root likelyPopularRoot) {
      this(ImmutableList.of(likelyPopularRoot));
    }

    /**
     * Creates an instance with the given likely roots.
     *
     * <p>When the RootCodec serializes any Root that compares equal to one of the likely roots, it
     * will be emitted as a single byte. Upon deserializing, that exact Root will be returned
     * (thereby canonicalizing to that Root instance).
     *
     * <p>Up to 255 likely roots may be specified. In practice, there should only be very few of
     * them; each serialization event may incur an equality comparison with all the likely roots.
     * Since the likely roots are checked in order, they should be ordered with the most likely ones
     * coming first.
     */
    public RootCodecDependencies(Iterable<Root> likelyPopularRoots) {
      this.likelyPopularRoots = ImmutableList.copyOf(likelyPopularRoots);
      // max length 255; value at index i encoded as number i + 1; value 0 means "not one of these".
      Preconditions.checkArgument(this.likelyPopularRoots.size() < 256);
    }
  }

  @SuppressWarnings("unused") // Used at run-time via classpath scanning + reflection.
  private static class RootCodec extends AsyncObjectCodec<Root> {
    private static final DynamicCodec PATH_ROOT_CODEC = new DynamicCodec(PathRoot.class);
    private static final DynamicCodec ABSOLUTE_ROOT_CODEC = new DynamicCodec(AbsoluteRoot.class);

    @Override
    public Class<? extends Root> getEncodedClass() {
      return Root.class;
    }

    @Override
    public void serialize(SerializationContext context, Root root, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      // Common case of a common root.
      RootCodecDependencies codecDeps = context.getDependency(RootCodecDependencies.class);
      for (int i = 0; i < codecDeps.likelyPopularRoots.size(); i++) {
        Root likely = codecDeps.likelyPopularRoots.get(i);
        if (root.equals(likely)) {
          codedOut.write((byte) (i + 1));
          return;
        }
      }

      // Everything else.
      codedOut.write((byte) 0);

      if (root instanceof PathRoot) {
        codedOut.writeBoolNoTag(true);
        PATH_ROOT_CODEC.serialize(context, root, codedOut);
      } else if (root instanceof AbsoluteRoot) {
        codedOut.writeBoolNoTag(false);
        ABSOLUTE_ROOT_CODEC.serialize(context, root, codedOut);
      } else {
        throw new IllegalStateException("Unexpected Root: " + root);
      }
    }

    @Override
    public Root deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int likelyIndicator = codedIn.readRawByte();
      if (likelyIndicator != 0) {
        RootCodecDependencies codecDeps = context.getDependency(RootCodecDependencies.class);
        Root popularRoot = codecDeps.likelyPopularRoots.get(likelyIndicator - 1);
        context.registerInitialValue(popularRoot);
        return popularRoot;
      }

      return (Root)
          (codedIn.readBool() ? PATH_ROOT_CODEC : ABSOLUTE_ROOT_CODEC)
              .deserializeAsync(context, codedIn);
    }
  }
}
