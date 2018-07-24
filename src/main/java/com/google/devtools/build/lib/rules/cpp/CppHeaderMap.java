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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.collect.ImmutableList.builder;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.ImmutableListCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/**
 * Structure for Clang header maps. Stores the .hmap and -internal.hmap artifacts as well
 * as the actual header maps that should be written to disk.
 */
@Immutable
@AutoCodec
public final class CppHeaderMap {
  // NOTE: If you add a field here, you'll likely need to update CppHeaderMapAction.computeKey().
  private final Artifact artifact;
  private final String name;
  private final String includePrefix;
  private final boolean flattenVirtualHeaders;
  private final ImmutableList<Artifact> headers;

  CppHeaderMap(
      Artifact artifact,
      String name,
      String includePrefix,
      boolean flattenVirtualHeaders,
      Iterable<Artifact> headers) {
    this.artifact = artifact;
    this.name = name;
    this.includePrefix = includePrefix;
    this.flattenVirtualHeaders = flattenVirtualHeaders;
    this.headers = ImmutableList.<Artifact>builder()
        .addAll(headers)
        .build();
  }

  public Artifact getArtifact() {
    return artifact;
  }

  public String getName() {
    return name;
  }

  public String getIncludePrefix() {
    return includePrefix;
  }

  public ImmutableList<Artifact> getHeaders() {
    return headers;
  }

  public ImmutableMap<String, PathFragment> getMapping() {
    ImmutableMap.Builder builder = ImmutableMap.builder();
    for (Artifact header : headers) {
      String key;
      if (flattenVirtualHeaders) {
        String basename = header.getExecPath().getBaseName();
        String prefix;
        if (includePrefix.equals("")) {
          prefix = includePrefix;
        } else {
          prefix = includePrefix + "/";
        }
        key = prefix + basename;
      } else {
        key = includePrefix + "/" + header.getExecPathString();
      }
      builder.put(key, header.getExecPath());
    }
    return builder.build();
  }

  @Override
  public String toString() {
    return name + "@" + artifact;
  }
}
