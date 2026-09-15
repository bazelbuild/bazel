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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.ProviderCollection;

/**
 * An abstract class for adding a Starlark API for the built-in providers. Derived classes should
 * declare functions to be used from Starlark.
 */
public abstract class StarlarkApiProvider {
  private ProviderCollection info;

  protected ProviderCollection getInfo() {
    return info;
  }

  public final void init(ProviderCollection info) {
    if (this.info != null) {
      // todo(dslomov): nuke this weird initialization mechanism.

      // Allow multiple calls.
      // It is possible for the Starlark rule to get a StarlarkApiProvider such as `target.java`
      // from its dependency and pass it on. It does not make a whole lot of sense, but we
      // shouldn't crash.
      return;
    }
    this.info = Preconditions.checkNotNull(info);
  }
}
