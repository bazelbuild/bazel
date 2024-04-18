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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** The value of the binding of "resolved" in a given file. */
public class ResolvedFileValue implements SkyValue {

  public static final String ORIGINAL_RULE_CLASS = "original_rule_class";
  public static final String ORIGINAL_ATTRIBUTES = "original_attributes";
  public static final String DEFINITION_INFORMATION = "definition_information";
  public static final String RULE_CLASS = "rule_class";
  public static final String ATTRIBUTES = "attributes";
  public static final String OUTPUT_TREE_HASH = "output_tree_hash";
  public static final String REPOSITORIES = "repositories";
  public static final String NATIVE = "native";

  /** Argument for the SkyKey to request the resolved value of a file. */
  @Immutable
  @AutoCodec
  public static class ResolvedFileKey implements SkyKey {
    private static final SkyKeyInterner<ResolvedFileKey> interner = SkyKey.newInterner();

    private final RootedPath path;

    private ResolvedFileKey(RootedPath path) {
      this.path = path;
    }

    private static ResolvedFileKey create(RootedPath path) {
      return interner.intern(new ResolvedFileKey(path));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static ResolvedFileKey intern(ResolvedFileKey resolvedFileKey) {
      return interner.intern(resolvedFileKey);
    }

    public RootedPath getPath() {
      return path;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.RESOLVED_FILE;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof ResolvedFileKey other)) {
        return false;
      }
      return Objects.equals(path, other.path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public SkyKeyInterner<ResolvedFileKey> getSkyKeyInterner() {
      return interner;
    }
  }

  public static SkyKey key(RootedPath path) {
    return ResolvedFileKey.create(path);
  }

  private final List<Map<String, Object>> resolvedValue;

  ResolvedFileValue(List<Map<String, Object>> resolvedValue) {
    this.resolvedValue = resolvedValue;
  }

  public List<Map<String, Object>> getResolvedValue() {
    return resolvedValue;
  }

  @Override
  public int hashCode() {
    return resolvedValue.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ResolvedFileValue)) {
      return false;
    }
    return this.resolvedValue.equals(((ResolvedFileValue) other).getResolvedValue());
  }
}
