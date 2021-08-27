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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;

/** A natively-defined aspect that is may be referenced by Starlark attribute definitions. */
public abstract class StarlarkNativeAspect extends NativeAspectClass implements StarlarkAspect {
  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Function<Rule, AspectParameters> EMPTY_FUNCTION = input -> AspectParameters.EMPTY;

  @Override
  public void repr(Printer printer) {
    printer.append("<native aspect>");
  }

  @Override
  public void attachToAspectsList(
      String baseAspectName,
      AspectsListBuilder aspectsList,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> inheritedRequiredProviders,
      ImmutableList<String> inheritedAttributeAspects,
      boolean allowAspectsParameters)
      throws EvalException {

    if (!allowAspectsParameters && !this.getParamAttributes().isEmpty()) {
      throw Starlark.errorf("Cannot use parameterized aspect %s at the top level.", this.getName());
    }

    aspectsList.addAspect(
        this, baseAspectName, inheritedRequiredProviders, inheritedAttributeAspects);
  }

  @Override
  public AspectClass getAspectClass() {
    return this;
  }

  @Override
  public ImmutableSet<String> getParamAttributes() {
    return ImmutableSet.of();
  }

  @Override
  public Function<Rule, AspectParameters> getDefaultParametersExtractor() {
    return EMPTY_FUNCTION;
  }
}
