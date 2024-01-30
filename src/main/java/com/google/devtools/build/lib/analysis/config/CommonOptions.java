// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.common.options.Options;

/** Common sets of option objects for use in core processing. */
public final class CommonOptions {

  // Ideally the empty build options should be actually empty: no fragment options and no flags. But
  // core Bazel
  // code assumes CoreOptions exists. For example CoreOptions.check_visibility is required for
  // basic configured target graph evaluation. So we provide CoreOptions with default values
  // (not inherited from parent configuration). This means flags like --check_visibility may not
  // be consistently applied. If this becomes a problem in practice we can carve out exceptions
  // to flags like that to propagate.
  // TODO(bazel-team): break out flags that configure Bazel's analysis phase into their own
  // FragmentOptions and propagate them to this configuration. Those flags should also be
  // ineligible outputs for other transitions because they're not meant for rule logic.  That
  // would guarantee consistency of flags like --check_visibility while still preventing forking.
  public static final BuildOptions EMPTY_OPTIONS = createEmptyOptions();

  private static BuildOptions createEmptyOptions() {
    BuildOptions options =
        BuildOptions.builder().addFragmentOptions(Options.getDefaults(CoreOptions.class)).build();
    // Disable the exec transition. Since this config is empty it shouldn't trigger any exec
    // transitions. More important, the default value this would otherwise propagate may not exist
    // in the repo (if the repo remaps with a repo-wide bazelrc).
    options.get(CoreOptions.class).starlarkExecConfig = null;
    return options;
  }

  private CommonOptions() {}
}
