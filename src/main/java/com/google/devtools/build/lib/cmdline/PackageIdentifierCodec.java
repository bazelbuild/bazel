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

package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Custom serialization logic for {@link PackageIdentifier}s. */
public class PackageIdentifierCodec implements ObjectCodec<PackageIdentifier> {

  private final RepositoryNameCodec repoNameCodec = new RepositoryNameCodec();

  @Override
  public Class<PackageIdentifier> getEncodedClass() {
    return PackageIdentifier.class;
  }

  @Override
  public void serialize(PackageIdentifier pkgId, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    repoNameCodec.serialize(pkgId.getRepository(), codedOut);
    PathFragment.CODEC.serialize(pkgId.getPackageFragment(), codedOut);
  }

  @Override
  public PackageIdentifier deserialize(CodedInputStream codedIn)
      throws IOException, SerializationException {
    RepositoryName repoName = repoNameCodec.deserialize(codedIn);
    PathFragment pathFragment = PathFragment.CODEC.deserialize(codedIn);
    return PackageIdentifier.create(repoName, pathFragment);
  }
}
