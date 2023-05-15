// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.eval.StarlarkValue;

/** Represents a single {@code .proto} source file. Deserializer only used in tests */
@AutoValue
abstract class ProtoSource implements StarlarkValue {
  @VisibleForTesting
  abstract Artifact getSourceFile();

  @VisibleForTesting
  abstract PathFragment getSourceRoot();

  @VisibleForTesting
  PathFragment getImportPath() {
    return getSourceFile().getRepositoryRelativePath().relativeTo(getSourceRoot());
  }

  @VisibleForTesting
  static ProtoSource create(StarlarkInfo protoSourceStruct) throws Exception {
    return new AutoValue_ProtoSource(
        protoSourceStruct.getValue("_source_file", Artifact.class),
        PathFragment.create(protoSourceStruct.getValue("_proto_path", String.class)));
  }
}
