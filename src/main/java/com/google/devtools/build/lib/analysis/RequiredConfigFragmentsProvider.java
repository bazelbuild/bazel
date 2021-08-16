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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
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
  private final ImmutableSet<String> requiredConfigFragments;

  public RequiredConfigFragmentsProvider(ImmutableSet<String> requiredConfigFragments) {
    this.requiredConfigFragments = requiredConfigFragments;
  }

  public ImmutableSet<String> getRequiredConfigFragments() {
    return requiredConfigFragments;
  }

  /** Merges the values of multiple {@link RequiredConfigFragmentsProvider}s. */
  public static RequiredConfigFragmentsProvider merge(
      List<RequiredConfigFragmentsProvider> providers) {
    checkArgument(!providers.isEmpty());
    if (providers.size() == 1) {
      return providers.get(0);
    }
    RequiredConfigFragmentsProvider.Builder merged = builder();
    for (RequiredConfigFragmentsProvider provider : providers) {
      merged.merge(provider);
    }
    return new RequiredConfigFragmentsProvider(merged.build());
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for required config fragments. */
  // TODO(b/149094955): Have this build a structured version of RequiredConfigFragmentsProvider.
  public static final class Builder {
    private final ImmutableSortedSet.Builder<String> strings = ImmutableSortedSet.naturalOrder();

    private Builder() {}

    // TODO(b/149094955): This should never accept arbitrary strings.
    public Builder addAll(Iterable<String> strings) {
      this.strings.addAll(strings);
      return this;
    }

    public Builder addOptionsClass(Class<? extends FragmentOptions> optionsClass) {
      strings.add(ClassName.getSimpleNameWithOuter(optionsClass));
      return this;
    }

    public Builder addOptionsClasses(Iterable<Class<? extends FragmentOptions>> optionsClasses) {
      return addClasses(optionsClasses);
    }

    public Builder addFragmentClasses(Iterable<Class<? extends Fragment>> fragmentClasses) {
      return addClasses(fragmentClasses);
    }

    private Builder addClasses(Iterable<? extends Class<?>> classes) {
      return addAll(Iterables.transform(classes, ClassName::getSimpleNameWithOuter));
    }

    public Builder addDefine(String define) {
      strings.add("--define:" + define);
      return this;
    }

    public Builder addStarlarkOption(Label starlarkOption) {
      return addStarlarkOption(starlarkOption.toString());
    }

    public Builder addStarlarkOption(String starlarkOption) {
      strings.add(starlarkOption);
      return this;
    }

    public Builder merge(RequiredConfigFragmentsProvider provider) {
      return addAll(provider.requiredConfigFragments);
    }

    public ImmutableSortedSet<String> build() {
      return strings.build();
    }
  }
}
