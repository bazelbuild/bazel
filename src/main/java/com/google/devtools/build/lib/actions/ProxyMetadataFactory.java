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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.FileArtifactValue.ProxyFileArtifactValue;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Factory for {@link ProxyFileArtifactValue}.
 *
 * <p>Used by {@link ActionCacheChecker} to re-create proxy metadata for cached actions.
 */
public interface ProxyMetadataFactory {

  @Nullable
  ProxyFileArtifactValue createProxyMetadata(Artifact artifact) throws IOException;

  /** Factory suitable for builds that never expect to use {@link ProxyFileArtifactValue}. */
  ProxyMetadataFactory NO_PROXIES = artifact -> null;
}
