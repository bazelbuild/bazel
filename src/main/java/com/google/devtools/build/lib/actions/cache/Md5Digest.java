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

import com.google.common.base.Preconditions;
import com.google.common.hash.HashCode;
import java.util.Arrays;

/** A value class for capturing and comparing MD5-based digests. */
public class Md5Digest {

  static final int MD5_SIZE = 16;
  private final byte[] digest;

  /**
   * Construct the digest from the given bytes.
   *
   * @param digest an MD5 digest. Must be sized properly.
   */
  public Md5Digest(byte[] digest) {
    Preconditions.checkState(digest.length == MD5_SIZE);
    this.digest = digest;
  }

  @Override
  public int hashCode() {
    // We are already dealing with the digest so we can just use portion of it as a hash code.
    return digest[0] + (digest[1] << 8) + (digest[2] << 16) + (digest[3] << 24);
  }

  @Override
  public boolean equals(Object obj) {
    return (obj instanceof Md5Digest) && Arrays.equals(digest, ((Md5Digest) obj).digest);
  }

  @Override
  public String toString() {
    return HashCode.fromBytes(digest).toString();
  }

  /**
   * Warning: This is a mutable wrapper and does not necessarily own the underlying byte[]. Don't
   * modify it unless you are sure about it.
   */
  public byte[] getDigestBytesUnsafe() {
    return digest;
  }
}
