// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import javax.annotation.Nullable;

// Note: AutoValue v1.4-rc1 has AutoValue.CopyAnnotations which makes it work with Skylark. No need
// to un-AutoValue this class to expose it to Skylark.
/**
 * Specifies how to generate language-specific code from .proto files. Used by LANG_proto_library
 * rules.
 */
@AutoValue
public abstract class ProtoLangToolchainProvider implements TransitiveInfoProvider {
  public abstract String commandLine();

  @Nullable
  public abstract FilesToRunProvider pluginExecutable();

  @Nullable
  public abstract TransitiveInfoCollection runtime();

  public abstract NestedSet<Artifact> blacklistedProtos();

  public static ProtoLangToolchainProvider create(
      String commandLine,
      FilesToRunProvider pluginExecutable,
      TransitiveInfoCollection runtime,
      NestedSet<Artifact> blacklistedProtos) {
    return new AutoValue_ProtoLangToolchainProvider(
        commandLine, pluginExecutable, runtime, blacklistedProtos);
  }
}
