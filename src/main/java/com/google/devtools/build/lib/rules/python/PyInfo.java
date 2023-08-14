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

package com.google.devtools.build.lib.rules.python;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import net.starlark.java.eval.EvalException;

/** Instance of the provider type for the Python rules. */
@VisibleForTesting
public final class PyInfo {
  public static final PyInfoProvider PROVIDER = new PyInfoProvider();

  private final StarlarkInfo info;

  private PyInfo(StarlarkInfo info) {
    this.info = info;
  }

  public PyInfoProvider getProvider() {
    return PROVIDER;
  }

  public NestedSet<Artifact> getTransitiveSourcesSet() throws EvalException {
    Object value = info.getValue("transitive_sources");
    return Depset.cast(value, Artifact.class, "transitive_sources");
  }

  public boolean getUsesSharedLibraries() throws EvalException {
    return info.getValue("uses_shared_libraries", Boolean.class);
  }

  public NestedSet<String> getImportsSet() throws EvalException {
    Object value = info.getValue("imports");
    return Depset.cast(value, String.class, "imports");
  }

  public boolean getHasPy2OnlySources() throws EvalException {
    return info.getValue("has_py2_only_sources", Boolean.class);
  }

  public boolean getHasPy3OnlySources() throws EvalException {
    return info.getValue("has_py3_only_sources", Boolean.class);
  }

  /** The singular PyInfo provider type object. */
  public static class PyInfoProvider extends StarlarkProviderWrapper<PyInfo> {

    private PyInfoProvider() {
      super(Label.parseCanonicalUnchecked("@_builtins//:common/python/providers.bzl"), "PyInfo");
    }

    @Override
    public PyInfo wrap(Info value) {
      return new PyInfo((StarlarkInfo) value);
    }
  }
}
