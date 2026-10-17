// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.hash;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.RepoDefinition;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Computes a single digest of the contents of an external repository, for {@code
 * --generate_hashes}.
 *
 * <p>The contents of a repo fetched by a repo rule that is neither {@code local} nor {@code
 * configure} only change when Bazel refetches it, and a refetch invalidates the repo's {@link
 * RepositoryDirectoryValue}. Such a repo can therefore be hashed as one unit, read once per fetch
 * with direct filesystem calls, instead of tracking each of its files in Skyframe. That matters for
 * toolchain repos with hundreds of thousands of files, which Bazel would otherwise check for
 * changes on every command. Edits made by hand inside such a repo are not seen.
 *
 * <p>The executable bit of files is not part of the digest: the same repo reports different
 * executable bits depending on whether it was fetched locally or is served lazily from the remote
 * repo contents cache.
 *
 * <p>Repo rules often create absolute symlinks into the output base or the workspace, and Bazel
 * only makes them relative for reproducible repos. To keep the digest independent of where the
 * output base and the workspace are, absolute symlink targets under them are hashed relative to
 * them. Symlinks into the workspace are also hashed by the contents of the files they point to:
 * Bazel refetches the repo when such a file changes (if the repo rule watches it), but the symlink
 * itself stays the same.
 *
 * <p>For other repos (the main repo, {@code local} and {@code configure} repos, and overridden
 * repos), {@link Value#usesFileDigests} is true and source files are hashed individually by {@link
 * SourceFileDigestFunction}.
 */
public final class RepoContentDigestFunction implements SkyFunction {
  // Not hermetic: the repo's files are read directly instead of through Skyframe.
  public static final SkyFunctionName NAME =
      SkyFunctionName.createNonHermetic("QUERY_REPO_CONTENT_DIGEST");

  private final BlazeDirectories directories;

  public RepoContentDigestFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  public static Key key(RepositoryName repo) {
    return Key.create(repo);
  }

  /** Key for {@link RepoContentDigestFunction}. */
  public static final class Key extends AbstractSkyKey<RepositoryName> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    private static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return NAME;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /** The digest of a repo's contents, or null if its files must be hashed individually. */
  public static final class Value implements SkyValue {
    static final Value USE_FILE_DIGESTS = new Value(null);

    @Nullable private final byte[] digest;

    private Value(@Nullable byte[] digest) {
      this.digest = digest;
    }

    public boolean usesFileDigests() {
      return digest == null;
    }

    /** The digest of the repo's contents, or null if {@link #usesFileDigests} is true. */
    @Nullable
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof Value other && Arrays.equals(digest, other.digest);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(digest);
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws RepoContentDigestFunctionException, InterruptedException {
    RepositoryName repo = (RepositoryName) skyKey.argument();
    if (repo.isMain()) {
      return Value.USE_FILE_DIGESTS;
    }
    var definition = (RepoDefinitionValue) env.getValue(RepoDefinitionValue.key(repo));
    if (definition == null) {
      return null;
    }
    if (!(definition instanceof RepoDefinitionValue.Found(RepoDefinition repoDefinition))
        || repoDefinition.repoRule().local()
        || repoDefinition.repoRule().configure()) {
      // Overridden, local and configure repos can change without a refetch.
      return Value.USE_FILE_DIGESTS;
    }
    var directory = (RepositoryDirectoryValue) env.getValue(RepositoryDirectoryValue.key(repo));
    if (directory == null) {
      return null;
    }
    if (!(directory instanceof RepositoryDirectoryValue.Success success)) {
      return Value.USE_FILE_DIGESTS;
    }
    Fingerprint fp = new Fingerprint();
    try {
      Path root = success.root().asPath();
      Path externalRoot =
          directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
      // Most specific first. The repo root may itself be a symlink into the repo contents cache.
      ImmutableList<SymlinkBase> bases =
          ImmutableList.of(
              new SymlinkBase(root.asFragment(), "<repo>"),
              new SymlinkBase(root.resolveSymbolicLinks().asFragment(), "<repo>"),
              new SymlinkBase(externalRoot.asFragment(), "<external>"),
              new SymlinkBase(directories.getWorkspace().asFragment(), "<workspace>"),
              new SymlinkBase(directories.getOutputBase().asFragment(), "<output_base>"));
      addDirectory(fp, root, "", bases);
    } catch (IOException e) {
      throw new RepoContentDigestFunctionException(e);
    }
    return new Value(fp.digestAndReset());
  }

  /** A directory that absolute symlink targets under it are hashed relative to. */
  private record SymlinkBase(PathFragment path, String name) {}

  /** Adds the sorted entries of {@code dir}, recursively, without following symlinks. */
  private void addDirectory(
      Fingerprint fp, Path dir, String relativePath, ImmutableList<SymlinkBase> bases)
      throws IOException, InterruptedException {
    List<Dirent> entries = new ArrayList<>(dir.readdir(Symlinks.NOFOLLOW));
    entries.sort(Comparator.comparing(Dirent::getName));
    for (Dirent entry : entries) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path path = dir.getChild(entry.getName());
      String childPath = relativePath + "/" + entry.getName();
      fp.addString(childPath);
      switch (entry.getType()) {
        case DIRECTORY -> {
          fp.addString("<directory>");
          addDirectory(fp, path, childPath, bases);
        }
        case SYMLINK -> {
          fp.addString("<symlink>");
          addSymlinkTarget(fp, path, bases);
        }
        case FILE -> {
          fp.addString("<file>");
          fp.addBytes(path.getDigest());
        }
        default -> fp.addString("<special>");
      }
    }
  }

  private void addSymlinkTarget(Fingerprint fp, Path link, ImmutableList<SymlinkBase> bases)
      throws IOException {
    PathFragment target = link.readSymbolicLink();
    if (!target.isAbsolute()) {
      fp.addString(target.getPathString());
      return;
    }
    for (SymlinkBase base : bases) {
      if (target.startsWith(base.path())) {
        fp.addString(base.name());
        fp.addString(target.relativeTo(base.path()).getPathString());
        if (base.name().equals("<workspace>")) {
          Path file = link.getFileSystem().getPath(target);
          FileStatus status = file.statIfFound(Symlinks.FOLLOW);
          if (status != null && status.isFile()) {
            fp.addBytes(file.getDigest());
          } else {
            fp.addString(status == null ? "<missing>" : "<not a file>");
          }
        }
        return;
      }
    }
    fp.addString(target.getPathString());
  }

  private static final class RepoContentDigestFunctionException extends SkyFunctionException {
    RepoContentDigestFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
