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

package com.google.devtools.build.lib.actions.cache;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Arrays;
import java.util.Date;

/**
 * A class to represent file metadata.
 * ActionCacheChecker may assume that, for a given file, equal
 * metadata at different moments implies equal file-contents,
 * where metadata equality is computed using Metadata.equals().
 * <p>
 * NB! Several other parts of Blaze are relying on the fact that metadata
 * uses mtime and not ctime. If metadata is ever changed
 * to use ctime, all uses of Metadata must be carefully examined.
 */
@Immutable @ThreadSafe
public final class Metadata {
  private final long mtime;
  private final byte[] digest;

  // Convenience object for use with volatile files that we do not want checked
  // (e.g. the build-changelist.txt)
  public static final Metadata CONSTANT_METADATA = new Metadata(-1);

  /**
   * Construct an instance for a directory with the specified mtime. The {@link #isFile} method
   * returns true if and only if a digest is set.
   */
  public Metadata(long mtime) {
    this.mtime = mtime;
    this.digest = null;
  }

  /**
   * Construct an instance for a file with the specified digest. The {@link #isFile} method returns
   * true if and only if a digest is set.
   */
  public Metadata(byte[] digest) {
    this.mtime = 0L;
    this.digest = Preconditions.checkNotNull(digest);
  }

  public boolean isFile() {
    return digest != null;
  }

  /**
   * Returns the digest for the underlying file system object.
   *
   * <p>The return value is owned by the cache and must not be modified.
   */
  public byte[] getDigest() {
    Preconditions.checkState(digest != null);
    return digest;
  }

  public long getModifiedTime() {
    return mtime;
  }

  @Override
  public int hashCode() {
    int hash = 0;
    if (digest != null) {
      // We are already dealing with the digest so we can just use portion of it
      // as a hash code.
      hash += digest[0] + (digest[1] << 8) + (digest[2] << 16) + (digest[3] << 24);
    } else {
      // Inlined hashCode for Long, so we don't
      // have to construct an Object, just to compute
      // a 32-bit hash out of a 64 bit value.
      hash = (int) (mtime ^ (mtime >>> 32));
    }
    return hash;
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }
    if (!(that instanceof Metadata)) {
      return false;
    }
    // Do a strict comparison - both digest and mtime should match
    return Arrays.equals(this.digest, ((Metadata) that).digest)
        && this.mtime == ((Metadata) that).mtime;
  }

  @Override
  public String toString() {
    if (digest != null) {
      return "MD5 " + BaseEncoding.base16().lowerCase().encode(digest);
    } else if (mtime > 0) {
      return "timestamp " + new Date(mtime);
    }
    return "no metadata";
  }
}
