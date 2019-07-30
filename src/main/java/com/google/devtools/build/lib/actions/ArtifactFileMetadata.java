// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.BigIntegerFingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.math.BigInteger;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value that represents a file for the purposes of up-to-dateness checks of actions.
 *
 * <p>It always stands for an actual file. In particular, tree artifacts and middlemen do not have a
 * corresponding {@link ArtifactFileMetadata}. However, the file is not necessarily present in the
 * file system; this happens when intermediate build outputs are not downloaded (and maybe when an
 * input artifact of an action is missing?)
 *
 * <p>It makes its main appearance in {@code ActionExecutionValue.artifactData}. It has two main
 * uses:
 *
 * <ul>
 *   <li>This is how dependent actions get hold of the output metadata of their generated inputs. In
 *       this case, it will be transformed into a {@link FileArtifactValue} by {@code
 *       ArtifactFunction}.
 *   <li>This is how {@code FileSystemValueChecker} figures out which actions need to be invalidated
 *       (just propagating the invalidation up from leaf nodes is not enough, because the output
 *       tree may have been changed while Blaze was not looking)
 * </ul>
 *
 * <p>It would probably be possible to unify this and {@link FileArtifactValue} since they contain
 * much the same data. However, {@link FileArtifactValue} has a few other uses that are do not map
 * easily to {@link ArtifactFileMetadata}, mostly relating to ActionFS.
 */
@Immutable
@ThreadSafe
public abstract class ArtifactFileMetadata {
  /**
   * Used as as placeholder in {@code OutputStore.artifactData} for artifacts that have entries in
   * {@code OutputStore.additionalOutputData}.
   */
  @AutoCodec public static final ArtifactFileMetadata PLACEHOLDER = new PlaceholderFileValue();

  // No implementations outside this class.
  private ArtifactFileMetadata() {}

  public boolean exists() {
    return realFileStateValue().getType() != FileStateType.NONEXISTENT;
  }

  /** Returns true if the original path is a symlink; the target path can never be a symlink. */
  public boolean isSymlink() {
    return false;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing regular or special
   * file. If so, its parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return realFileStateValue().getType() == FileStateType.REGULAR_FILE;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing special file. If so,
   * its parent directory is guaranteed to exist.
   */
  public boolean isSpecialFile() {
    return realFileStateValue().getType() == FileStateType.SPECIAL_FILE;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isDirectory() {
    return realFileStateValue().getType() == FileStateType.DIRECTORY;
  }

  protected abstract FileStateValue realFileStateValue();

  public FileContentsProxy getContentsProxy() {
    return realFileStateValue().getContentsProxy();
  }

  public long getSize() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getSize();
  }

  @Nullable
  public byte[] getDigest() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getDigest();
  }

  /** Returns a quick fingerprint via a BigInteger */
  public BigInteger getFingerprint() {
    BigIntegerFingerprint fp = new BigIntegerFingerprint();
    fp.addBoolean(exists());
    fp.addBoolean(isSpecialFile());
    fp.addBoolean(isDirectory());
    fp.addBoolean(isFile());
    if (isFile()) {
      fp.addLong(getSize());
      fp.addDigestedBytes(getDigest());
    }
    return fp.getFingerprint();
  }

  public static ArtifactFileMetadata forRegularFile(
      PathFragment pathFragment, FileStateValue fileStateValue) {
    return new Regular(pathFragment, fileStateValue);
  }

  public static ArtifactFileMetadata forSymlink(
      PathFragment resolvedPath, FileStateValue resolvedFileStateValue, PathFragment linkTarget) {
    return new Symlink(resolvedPath, resolvedFileStateValue, linkTarget);
  }

  /**
   * Implementation of {@link ArtifactFileMetadata} for files whose fully resolved path is the same
   * as the requested path. For example, this is the case for the path "foo/bar/baz" if neither
   * 'foo' nor 'foo/bar' nor 'foo/bar/baz' are symlinks.
   */
  private static final class Regular extends ArtifactFileMetadata {
    private final PathFragment realPath;
    private final FileStateValue fileStateValue;

    Regular(PathFragment realPath, FileStateValue fileStateValue) {
      this.realPath = Preconditions.checkNotNull(realPath);
      this.fileStateValue = Preconditions.checkNotNull(fileStateValue);
    }

    @Override
    public FileStateValue realFileStateValue() {
      return fileStateValue;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      return fileStateValue.getContentsProxy();
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != Regular.class) {
        return false;
      }
      Regular other = (Regular) obj;
      return realPath.equals(other.realPath) && fileStateValue.equals(other.fileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realPath, fileStateValue);
    }

    @Override
    public String toString() {
      return realPath + ", " + fileStateValue;
    }

    @Override
    public BigInteger getFingerprint() {
      BigInteger original = super.getFingerprint();
      BigIntegerFingerprint fp = new BigIntegerFingerprint();
      fp.addBigIntegerOrdered(original);
      fp.addString(getClass().getCanonicalName());
      fp.addPath(realPath);
      fp.addBigIntegerOrdered(fileStateValue.getValueFingerprint());
      return fp.getFingerprint();
    }
  }

  /** Implementation of {@link ArtifactFileMetadata} for files that are symlinks. */
  private static final class Symlink extends ArtifactFileMetadata {
    private final PathFragment linkTarget;
    private final PathFragment realPath;
    private final FileStateValue realFileStateValue;

    private Symlink(
        PathFragment realPath, FileStateValue realFileStateValue, PathFragment linkTarget) {
      this.realPath = realPath;
      this.realFileStateValue = realFileStateValue;
      this.linkTarget = linkTarget;
    }

    @Override
    public FileStateValue realFileStateValue() {
      return realFileStateValue;
    }

    @Override
    public BigInteger getFingerprint() {
      BigInteger original = super.getFingerprint();
      BigIntegerFingerprint fp = new BigIntegerFingerprint();
      fp.addBigIntegerOrdered(original);
      fp.addString(getClass().getCanonicalName());
      fp.addPath(linkTarget);
      return fp.getFingerprint();
    }

    @Override
    public boolean isSymlink() {
      return true;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != Symlink.class) {
        return false;
      }
      Symlink other = (Symlink) obj;
      return realPath.equals(other.realPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && linkTarget.equals(other.linkTarget);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realPath, realFileStateValue, linkTarget, Boolean.TRUE);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink (real_path=%s, real_state=%s, link_value=%s)",
          realPath, realFileStateValue, linkTarget);
    }
  }

  private static final class PlaceholderFileValue extends ArtifactFileMetadata {
    private static final BigInteger FINGERPRINT =
        new BigIntegerFingerprint().addString("PlaceholderFileValue").getFingerprint();

    private PlaceholderFileValue() {}

    @Override
    public FileStateValue realFileStateValue() {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public BigInteger getFingerprint() {
      return FINGERPRINT;
    }

    @Override
    public String toString() {
      return "PlaceholderFileValue:Singleton";
    }
  }
}
