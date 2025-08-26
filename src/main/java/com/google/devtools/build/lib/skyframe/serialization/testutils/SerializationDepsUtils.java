// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.Artifact.ArtifactSerializationContext;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;

/** Utilities for testing with serialization dependencies. */
public final class SerializationDepsUtils {

  /** Default serialization dependencies for testing. */
  public static final ImmutableClassToInstanceMap<?> SERIALIZATION_DEPS_FOR_TEST =
      ImmutableClassToInstanceMap.builder()
          .put(
              ArtifactSerializationContext.class,
              (execPath, root, owner) -> new SourceArtifact(root, execPath, owner))
          .put(OptionsChecksumCache.class, new MapBackedChecksumCache())
          .build();

  private SerializationDepsUtils() {}
}
