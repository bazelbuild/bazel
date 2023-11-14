// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.FragmentCollectionApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Represents a collection of configuration fragments in Starlark. */
// Documentation can be found at ctx.fragments
@Immutable
public class FragmentCollection implements FragmentCollectionApi {
  private final RuleContext ruleContext;

  public FragmentCollection(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  @Nullable
  public Object getValue(String name) throws EvalException {
    return ruleContext.getStarlarkFragment(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ruleContext.getStarlarkFragmentNames();
  }

  @Override
  public String toString() {
    return "[ " + fieldsToString() + "]";
  }
}
