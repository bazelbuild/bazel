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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import javax.annotation.Nullable;

/**
 * Container for storing a set of native and Starlark flag settings in separate buckets.
 *
 * <p>This is necessary because native and Starlark flags are parsed with different logic.
 */
@AutoValue
public abstract class NativeAndStarlarkFlags {

  /** Builder for new {@link NativeAndStarlarkFlags} instances. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder nativeFlags(ImmutableList<String> nativeFlags);

    public abstract Builder starlarkFlags(ImmutableMap<String, Object> starlarkFlags);

    public abstract Builder optionsClasses(
        ImmutableSet<Class<? extends FragmentOptions>> optionsClasses);

    public abstract Builder repoMapping(RepositoryMapping repoMapping);

    public abstract NativeAndStarlarkFlags build();
  }

  /** Returns a new {@link Builder}. */
  public static Builder builder() {
    return new AutoValue_NativeAndStarlarkFlags.Builder()
        .nativeFlags(ImmutableList.of())
        .starlarkFlags(ImmutableMap.of())
        .optionsClasses(ImmutableSet.of());
  }

  /**
   * The native flags from a given set of flags, in the format <code>[--flag=value]</code> or <code>
   * ["--flag", "value"]</code>.
   */
  public abstract ImmutableList<String> nativeFlags();

  /**
   * The Starlark flags from a given set of flags, mapped to the correct converted data type. If a
   * Starlark flag is explicitly set to the default value it should still appear in this map so that
   * consumers can properly handle the flag.
   */
  public abstract ImmutableMap<String, Object> starlarkFlags();

  abstract ImmutableSet<Class<? extends FragmentOptions>> optionsClasses();

  @Nullable
  abstract RepositoryMapping repoMapping();

  public boolean isEmpty() {
    return nativeFlags().isEmpty() && starlarkFlags().isEmpty();
  }

  public OptionsParsingResult parse() throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(this.optionsClasses())
            // We need the ability to re-map internal options in the mappings file.
            .ignoreInternalOptions(false)
            .withConversionContext(this.repoMapping())
            .build();
    parser.parse(this.nativeFlags().asList());
    parser.setStarlarkOptions(this.starlarkFlags());
    return parser;
  }

  /**
   * Returns a new {@link BuildOptions} instance, which contains all flags from the given {@link
   * BuildOptions} with the flags in this {@link NativeAndStarlarkFlags} merged in.
   *
   * <p>The merging logic is as follows:
   * <li>For native flags, only the fragments in the original {@link BuildOptions} are kept.
   * <li>Any native flags in this instance, for fragments that are kept, are set to the value from
   *     this instance.
   * <li>All Starlark flags from the original {@link BuildOptions} are kept, then all Starlark
   *     options from this instance are added.
   * <li>Any Starlark flags which are present in both, the value from this instance is kept.
   *
   *     <p>To preserve fragment trimming, this method will not expand the set of included native
   *     fragments from the original {@link BuildOptions}. If the parsing result contains native
   *     options whose owning fragment is not part of the original {@link BuildOptions} they will be
   *     ignored (i.e. not set on the resulting options). Starlark options are not affected by this
   *     restriction.
   *
   * @param source the base options to modify
   * @return the new options after applying this object to the original options
   */
  public BuildOptions mergeWith(BuildOptions source) throws OptionsParsingException {
    // TODO: https://github.com/bazelbuild/bazel/issues/23147 - Inline applyParsingResult and remove
    // it from BuildOptions.
    return source.applyParsingResult(this.parse());
  }
}
