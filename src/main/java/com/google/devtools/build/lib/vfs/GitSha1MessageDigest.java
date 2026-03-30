// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.ByteArrayOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** A {@link MessageDigest} for GitSha1. */
public final class GitSha1MessageDigest extends MessageDigest {
  private static final byte[] header = {'b', 'l', 'o', 'b', ' '};
  private final MessageDigest sha1;
  private final ByteArrayOutputStream stream;

  public GitSha1MessageDigest() throws NoSuchAlgorithmException {
    super("GITSHA1");
    sha1 = MessageDigest.getInstance("SHA-1");
    stream = new ByteArrayOutputStream();
  }

  @Override
  public void engineUpdate(byte[] data, int offset, int length) {
    stream.write(data, offset, length);
  }

  @Override
  public void engineUpdate(byte b) {
    stream.write(b);
  }

  @Override
  protected void engineReset() {
    internalReset();
  }

  @Override
  protected byte[] engineDigest() {
    int size = stream.size();
    sha1.update(header);
    sha1.update(Integer.toString(size).getBytes(UTF_8));
    sha1.update((byte) 0);
    sha1.update(stream.toByteArray());
    byte[] digest = sha1.digest();
    internalReset();
    return digest;
  }

  private void internalReset() {
    sha1.reset();
    stream.reset();
  }
}
