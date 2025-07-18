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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.FrontierNodeVersion;
import com.google.devtools.build.skyframe.SkyKey;

/** State to track the {@link SkyKey}s that were deserialized from the remote analysis cache. */
public record RemoteAnalysisCachingState(
    FrontierNodeVersion version, ImmutableSet<SkyKey> deserializedKeys) {
  public static final RemoteAnalysisCachingState EMPTY =
      new RemoteAnalysisCachingState(null, ImmutableSet.of());
}
