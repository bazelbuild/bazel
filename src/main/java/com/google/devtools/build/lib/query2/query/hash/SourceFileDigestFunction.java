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

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import javax.annotation.Nullable;

/**
 * Computes the content digest and executable bit of a source file for {@code query
 * --generate_hashes}.
 *
 * <p>The value depends on the file's {@link FileValue}, so on a warm server the file is only
 * re-read when Skyframe observes that its state changed. A mode change alone is observed when the
 * file state is tracked by its ctime, which is the default for local files; it is not observed for
 * files whose state is a digest provided by the filesystem.
 */
public final class SourceFileDigestFunction implements SkyFunction {
  public static final SkyFunctionName NAME =
      SkyFunctionName.createHermetic("QUERY_SOURCE_FILE_DIGEST");

  public static Key key(RootedPath rootedPath) {
    return Key.create(rootedPath);
  }

  /** Key for {@link SourceFileDigestFunction}. */
  public static final class Key extends AbstractSkyKey<RootedPath> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RootedPath arg) {
      super(arg);
    }

    private static Key create(RootedPath arg) {
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

  /**
   * The digest and executable bit of a source file, or a marker for missing files and directories.
   */
  public static final class Value implements SkyValue {
    static final Value MISSING =
        new Value("<missing>".getBytes(StandardCharsets.UTF_8), /* executable= */ false);
    // Directories used as source labels are not hashed recursively.
    static final Value DIRECTORY =
        new Value("<directory>".getBytes(StandardCharsets.UTF_8), /* executable= */ false);
    static final Value SPECIAL =
        new Value("<special>".getBytes(StandardCharsets.UTF_8), /* executable= */ false);

    private final byte[] digest;
    private final boolean executable;

    private Value(byte[] digest, boolean executable) {
      this.digest = digest;
      this.executable = executable;
    }

    public byte[] getDigest() {
      return digest;
    }

    /** Whether the file is executable. It changes a target's outputs when the file is a runfile. */
    public boolean isExecutable() {
      return executable;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof Value other
          && Arrays.equals(digest, other.digest)
          && executable == other.executable;
    }

    @Override
    public int hashCode() {
      return 31 * Arrays.hashCode(digest) + Boolean.hashCode(executable);
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SourceFileDigestFunctionException, InterruptedException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(FileValue.key(rootedPath), IOException.class);
    } catch (IOException e) {
      throw new SourceFileDigestFunctionException(e);
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      return Value.MISSING;
    }
    if (fileValue.isDirectory()) {
      return Value.DIRECTORY;
    }
    if (fileValue.isSpecialFile()) {
      return Value.SPECIAL;
    }
    // Prefer the digest already recorded in the file state (available when the filesystem
    // provides fast digests); otherwise read the file, going through the computed digest cache.
    Path realPath = fileValue.realRootedPath(rootedPath).asPath();
    byte[] digest = fileValue.getDigest();
    boolean executable;
    try {
      if (digest == null) {
        digest = DigestUtils.manuallyComputeDigest(realPath, /* status= */ null);
      }
      executable = realPath.isExecutable();
    } catch (IOException e) {
      throw new SourceFileDigestFunctionException(e);
    }
    return new Value(digest, executable);
  }

  private static final class SourceFileDigestFunctionException extends SkyFunctionException {
    SourceFileDigestFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
