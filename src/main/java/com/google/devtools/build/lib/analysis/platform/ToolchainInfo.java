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

package com.google.devtools.build.lib.analysis.platform;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainInfoApi;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * A provider that supplies information about a specific language toolchain, including what platform
 * constraints are required for execution and for the target platform.
 *
 * <p>Unusually, ToolchainInfo exposes both its StarlarkCallable-annotated fields and a Map of
 * additional fields to Starlark code. Also, these are not disjoint.
 */
@Immutable
public class ToolchainInfo extends NativeInfo implements ToolchainInfoApi {

  /** Name used in Starlark for accessing this provider. */
  public static final String STARLARK_NAME = "ToolchainInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<ToolchainInfo> PROVIDER = new Provider();

  /** Provider for {@link ToolchainInfo} objects. */
  private static class Provider extends BuiltinProvider<ToolchainInfo>
      implements ToolchainInfoApi.Provider {
    private Provider() {
      super(STARLARK_NAME, ToolchainInfo.class);
    }

    @Override
    public ToolchainInfo toolchainInfo(Dict<String, Object> kwargs, StarlarkThread thread) {
      return new ToolchainInfo(kwargs, thread.getCallerLocation());
    }
  }

  @AutoCodec.VisibleForSerialization final ImmutableSortedMap<String, Object> values;
  private ImmutableSet<String> fieldNames; // initialized lazily (with monitor synchronization)

  /** Constructs a ToolchainInfo. The {@code values} map itself is not retained. */
  protected ToolchainInfo(Map<String, Object> values, Location location) {
    super(location);
    this.values = copyValues(values);
  }

  public ToolchainInfo(Map<String, Object> values) {
    this.values = copyValues(values);
  }

  @Override
  public BuiltinProvider<ToolchainInfo> getProvider() {
    return PROVIDER;
  }

  /**
   * Preprocesses a map of field values to convert the field names and field values to
   * Starlark-acceptable names and types.
   *
   * <p>Entries are ordered by key.
   */
  private static ImmutableSortedMap<String, Object> copyValues(Map<String, Object> values) {
    ImmutableSortedMap.Builder<String, Object> builder = ImmutableSortedMap.naturalOrder();
    for (Map.Entry<String, Object> e : values.entrySet()) {
      builder.put(Attribute.getStarlarkName(e.getKey()), Starlark.fromJava(e.getValue(), null));
    }
    return builder.build();
  }

  @Override
  public Object getValue(String name) throws EvalException {
    Object x = values.get(name);
    return x != null ? x : super.getValue(name);
  }

  @Override
  public synchronized ImmutableCollection<String> getFieldNames() {
    if (fieldNames == null) {
      fieldNames =
          ImmutableSet.<String>builder()
              .addAll(values.keySet())
              .addAll(super.getFieldNames())
              .build();
    }
    return fieldNames;
  }
}
