// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label.PackageContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/** Stores the {@link NativeAndStarlarkFlags} that are the result of {@link ParsedFlagsFunction}. */
@AutoCodec
public final class ParsedFlagsValue implements SkyValue {
  /** Key for {@link ParsedFlagsValue} based on the raw flags. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    /**
     * Returns a new {@link Key} for the given command-line flags, such as {@code
     * --compilation_mode=bdg} or {@code --//custom/starlark:flag=23}.
     */
    public static Key create(ImmutableList<String> rawFlags, PackageContext packageContext) {
      return create(rawFlags, packageContext, /* includeDefaultValues= */ false);
    }

    /**
     * Returns a new {@link Key} for the given command-line flags, such as {@code
     * --compilation_mode=bdg} or {@code --//custom/starlark:flag=23}.
     */
    @AutoCodec.Instantiator
    @VisibleForSerialization
    public static Key create(
        ImmutableList<String> rawFlags,
        PackageContext packageContext,
        boolean includeDefaultValues) {
      return interner.intern(new Key(rawFlags, packageContext, includeDefaultValues));
    }

    private final ImmutableList<String> rawFlags;
    private final PackageContext packageContext;
    private final boolean includeDefaultValues;

    private Key(
        ImmutableList<String> rawFlags,
        PackageContext packageContext,
        boolean includeDefaultValues) {
      this.rawFlags = rawFlags;
      this.packageContext = packageContext;
      this.includeDefaultValues = includeDefaultValues;
    }

    public ImmutableList<String> rawFlags() {
      return rawFlags;
    }

    public PackageContext packageContext() {
      return packageContext;
    }

    public boolean includeDefaultValues() {
      return includeDefaultValues;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PARSED_FLAGS;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(rawFlags, key.rawFlags)
          && Objects.equals(packageContext, key.packageContext);
    }

    @Override
    public int hashCode() {
      return Objects.hash(rawFlags, packageContext);
    }

    @Override
    public String toString() {
      return "ParsedFlagsValue.Key{rawFlags="
          + rawFlags
          + ", packageContext="
          + packageContext
          + "}";
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  public static ParsedFlagsValue create(NativeAndStarlarkFlags flags) {
    return new ParsedFlagsValue(flags);
  }

  private final NativeAndStarlarkFlags flags;

  ParsedFlagsValue(NativeAndStarlarkFlags flags) {
    this.flags = flags;
  }

  public NativeAndStarlarkFlags flags() {
    return flags;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ParsedFlagsValue that)) {
      return false;
    }
    return this.flags.equals(that.flags);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(flags);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("flags", flags).toString();
  }
}
