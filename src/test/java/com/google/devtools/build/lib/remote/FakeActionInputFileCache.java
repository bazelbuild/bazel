// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import javax.annotation.Nullable;

/** A fake implementation of the {@link ActionInputFileCache} interface. */
final class FakeActionInputFileCache implements ActionInputFileCache {
  private final Path execRoot;
  private final BiMap<ActionInput, ByteString> cas = HashBiMap.create();

  FakeActionInputFileCache(Path execRoot) {
    this.execRoot = execRoot;
  }

  void setDigest(ActionInput input, ByteString digest) {
    cas.put(input, digest);
  }

  @Override
  @Nullable
  public byte[] getDigest(ActionInput input) throws IOException {
    return Preconditions.checkNotNull(cas.get(input), input).toByteArray();
  }

  @Override
  public boolean isFile(Artifact input) {
    return execRoot.getRelative(input.getExecPath()).isFile();
  }

  @Override
  public long getSizeInBytes(ActionInput input) throws IOException {
    return execRoot.getRelative(input.getExecPath()).getFileSize();
  }

  @Override
  public boolean contentsAvailableLocally(ByteString digest) {
    throw new UnsupportedOperationException();
  }

  @Override
  @Nullable
  public ActionInput getInputFromDigest(ByteString hexDigest) {
    HashCode code = HashCode.fromString(hexDigest.toStringUtf8());
    ByteString digest = ByteString.copyFrom(code.asBytes());
    return Preconditions.checkNotNull(cas.inverse().get(digest));
  }

  @Override
  public Path getInputPath(ActionInput input) {
    throw new UnsupportedOperationException();
  }
}