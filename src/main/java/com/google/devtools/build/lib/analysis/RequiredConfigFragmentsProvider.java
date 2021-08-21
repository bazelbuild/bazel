// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.MoreObjects;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.ClassName;
import java.util.List;

/**
 * Provides a user-friendly list of the {@link Fragment}s and {@link
 * com.google.devtools.build.lib.analysis.config.FragmentOptions} required by this target and its
 * transitive dependencies.
 *
 * <p>See {@link com.google.devtools.build.lib.analysis.config.RequiredFragmentsUtil} for details.
 */
@Immutable
public final class RequiredConfigFragmentsProvider implements TransitiveInfoProvider {

  @SerializationConstant
  public static final RequiredConfigFragmentsProvider EMPTY =
      new RequiredConfigFragmentsProvider(
          ImmutableSet.of(),
          FragmentClassSet.of(ImmutableSet.of()),
          ImmutableSet.of(),
          ImmutableSet.of());

  private final ImmutableSet<Class<? extends FragmentOptions>> optionsClasses;
  private final FragmentClassSet fragmentClasses;
  private final ImmutableSet<String> defines;
  private final ImmutableSet<Label> starlarkOptions;

  private RequiredConfigFragmentsProvider(
      ImmutableSet<Class<? extends FragmentOptions>> optionsClasses,
      FragmentClassSet fragmentClasses,
      ImmutableSet<String> defines,
      ImmutableSet<Label> starlarkOptions) {
    this.optionsClasses = optionsClasses;
    this.fragmentClasses = fragmentClasses;
    this.defines = defines;
    this.starlarkOptions = starlarkOptions;
  }

  public ImmutableSet<Class<? extends FragmentOptions>> getOptionsClasses() {
    return optionsClasses;
  }

  public FragmentClassSet getFragmentClasses() {
    return fragmentClasses;
  }

  public ImmutableSet<String> getDefines() {
    return defines;
  }

  public ImmutableSet<Label> getStarlarkOptions() {
    return starlarkOptions;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add(
            "optionsClasses",
            Collections2.transform(optionsClasses, ClassName::getSimpleNameWithOuter))
        .add("fragmentClasses", fragmentClasses)
        .add("defines", defines)
        .add("starlarkOptions", starlarkOptions)
        .toString();
  }

  /** Merges the values of two {@link RequiredConfigFragmentsProvider} instances. */
  public static RequiredConfigFragmentsProvider merge(
      RequiredConfigFragmentsProvider a, RequiredConfigFragmentsProvider b) {
    if (a == EMPTY) {
      return b;
    }
    if (b == EMPTY) {
      return a;
    }
    return builder().merge(a).merge(b).build();
  }

  /** Merges the values of one or more {@link RequiredConfigFragmentsProvider} instances. */
  public static RequiredConfigFragmentsProvider merge(
      List<RequiredConfigFragmentsProvider> providers) {
    checkArgument(!providers.isEmpty());
    RequiredConfigFragmentsProvider.Builder merged = null;
    RequiredConfigFragmentsProvider candidate = EMPTY;
    for (RequiredConfigFragmentsProvider provider : providers) {
      if (provider == EMPTY) {
        continue;
      }
      if (merged != null) {
        merged.merge(provider);
      } else if (candidate == EMPTY) {
        candidate = provider;
      } else {
        merged = builder().merge(candidate).merge(provider);
      }
    }
    return merged == null ? candidate : merged.build();
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for required config fragments. */
  public static final class Builder {
    private final ImmutableSet.Builder<Class<? extends FragmentOptions>> optionsClasses =
        ImmutableSet.builder();
    private final ImmutableSet.Builder<Class<? extends Fragment>> fragmentClasses =
        ImmutableSortedSet.orderedBy(FragmentClassSet.LEXICAL_FRAGMENT_SORTER);
    private final ImmutableSet.Builder<String> defines = ImmutableSet.builder();
    private final ImmutableSet.Builder<Label> starlarkOptions = ImmutableSet.builder();

    private Builder() {}

    public Builder addOptionsClass(Class<? extends FragmentOptions> optionsClass) {
      optionsClasses.add(optionsClass);
      return this;
    }

    public Builder addOptionsClasses(Iterable<Class<? extends FragmentOptions>> optionsClasses) {
      this.optionsClasses.addAll(optionsClasses);
      return this;
    }

    public Builder addFragmentClasses(Iterable<Class<? extends Fragment>> fragmentClasses) {
      this.fragmentClasses.addAll(fragmentClasses);
      return this;
    }

    public Builder addDefine(String define) {
      defines.add(define);
      return this;
    }

    public Builder addStarlarkOption(Label starlarkOption) {
      starlarkOptions.add(starlarkOption);
      return this;
    }

    public Builder addStarlarkOptions(Iterable<Label> starlarkOptions) {
      this.starlarkOptions.addAll(starlarkOptions);
      return this;
    }

    public Builder merge(RequiredConfigFragmentsProvider provider) {
      optionsClasses.addAll(provider.optionsClasses);
      fragmentClasses.addAll(provider.fragmentClasses);
      defines.addAll(provider.defines);
      starlarkOptions.addAll(provider.starlarkOptions);
      return this;
    }

    public RequiredConfigFragmentsProvider build() {
      ImmutableSet<Class<? extends FragmentOptions>> optionsClasses = this.optionsClasses.build();
      ImmutableSet<Class<? extends Fragment>> fragmentClasses = this.fragmentClasses.build();
      ImmutableSet<String> defines = this.defines.build();
      ImmutableSet<Label> starlarkOptions = this.starlarkOptions.build();
      if (optionsClasses.isEmpty()
          && fragmentClasses.isEmpty()
          && defines.isEmpty()
          && starlarkOptions.isEmpty()) {
        return EMPTY;
      }
      return new RequiredConfigFragmentsProvider(
          optionsClasses, FragmentClassSet.of(fragmentClasses), defines, starlarkOptions);
    }
  }
}
