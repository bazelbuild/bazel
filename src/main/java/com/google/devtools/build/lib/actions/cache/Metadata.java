// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Date;

/**
 * A class to represent file metadata.
 * DatabaseDependencyChecker may assume that, for a given file, equal
 * metadata at different moments implies equal file-contents,
 * where metadata equality is computed using Metadata.equals().
 * <p>
 * NB! Several other parts of Blaze are relying on the fact that metadata
 * uses mtime and not ctime. E.g. AbstractBuilder will cache artifact
 * stats (through the use of artifactExists() method) before setting
 * read-only permissions.  If metadata is ever changed
 * to use ctime, all uses of ArtifactMetadataCache must be carefully examined.
 */
@Immutable @ThreadSafe
public final class Metadata {

  /** Indicates what kind of information is stored in this object; used by tests only. */
  @VisibleForTesting
  public enum Type {
    EMPTY,
    TIMESTAMP,
    MD5
  }

  public final long mtime;
  public final byte[] digest;

  // Convenience object (e.g. for use with ConcurrentHashMap)
  public static final Metadata NO_METADATA = new Metadata(0, null);

  // Convenience object for use with volatile files that we do not want checked
  // (e.g. the build-changelist.txt)
  public static final Metadata CONSTANT_METADATA = new Metadata(-1, null);

  public Metadata(long mtime, byte[] digest) {
    this.mtime = mtime;
    this.digest = digest;
  }

  // Should only be used by MetadataCache.
  void writeValue(DataOutputStream out) throws IOException {
    if (digest != null) {
      out.writeBoolean(true);
      out.write(digest);
    } else {
      out.writeBoolean(false);
    }
    out.writeLong(mtime);
  }

  // Should only be used by MetadataCache.
  static Metadata readValue(DataInputStream in) throws IOException {
    if (in.readBoolean()) {
      byte[] digestValue = new byte[16];
      in.readFully(digestValue);
      return new Metadata(in.readLong(), digestValue);
    } else {
      return new Metadata(in.readLong(), null);
    }
  }

  @Override
  public int hashCode() {
    // Inlined hashCode for Long, so we don't
    // have to construct an Object, just to compute
    // a 32-bit hash out of a 64 bit value.
    int hash = (int)(mtime ^ (mtime >>> 32));
    if (digest != null) {
      // We are already dealing with the digest so we can just use portion of it
      // as a hash code.
      hash += digest[0] + (digest[1] << 8) + (digest[2] << 16) + (digest[3] << 24);
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

  @VisibleForTesting
  public Type getTypeForTesting() {
    if (digest != null) {
      return Type.MD5;
    } else if (mtime > 0) {
      return Type.TIMESTAMP;
    } else {
      return Type.EMPTY;
    }
  }

  @Override
  public String toString() {
    if (digest != null) {
      return "MD5 " + BaseEncoding.base16().lowerCase().encode(digest) + " at " + new Date(mtime);
    } else if (mtime > 0) {
      return "timestamp " + new Date(mtime);
    }
    return "no metadata";
  }
}
