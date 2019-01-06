// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.skylark;

import static com.google.devtools.build.lib.analysis.skylark.FunctionTransitionUtil.applyAndValidate;
import static com.google.devtools.build.lib.analysis.skylark.SkylarkAttributesCollection.ERROR_MESSAGE_FOR_NO_ATTR;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * This class implements a {@link SplitTransitionProvider} to provide a starlark-defined transition
 * that rules can apply to their dependencies' configurations. This transition has access to (1) the
 * a map of the current configuration's build settings and (2) the configured attributes of the
 * given rule (not its dependencies').
 *
 * <p>For starlark defined rule class transitions, see {@link StarlarkRuleTransitionProvider}.
 *
 * <p>TODO(bazel-team): Consider allowing dependency-typed attributes to actually return providers
 * instead of just labels (see {@link SkylarkAttributesCollection#addAttribute}).
 */
public class StarlarkAttributeTransitionProvider implements SplitTransitionProvider {
  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  StarlarkAttributeTransitionProvider(
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  @Override
  public SplitTransition apply(AttributeMap attributeMap) {
    Preconditions.checkArgument(attributeMap instanceof ConfiguredAttributeMapper);
    return new FunctionSplitTransition(
        starlarkDefinedConfigTransition, (ConfiguredAttributeMapper) attributeMap);
  }

  private static class FunctionSplitTransition implements SplitTransition {
    private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;
    private final StructImpl attrObject;

    FunctionSplitTransition(
        StarlarkDefinedConfigTransition starlarkDefinedConfigTransition,
        ConfiguredAttributeMapper attributeMap) {
      this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;

      LinkedHashMap<String, Object> attributes = new LinkedHashMap<>();
      for (String attribute : attributeMap.getAttributeNames()) {
        Object val = attributeMap.get(attribute, attributeMap.getAttributeType(attribute));
        attributes.put(
            Attribute.getSkylarkName(attribute),
            val == null ? Runtime.NONE : SkylarkType.convertToSkylark(val, (Environment) null));
      }
      attrObject = StructProvider.STRUCT.create(attributes, ERROR_MESSAGE_FOR_NO_ATTR);
    }

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      return applyAndValidate(buildOptions, starlarkDefinedConfigTransition, attrObject);
    }
  }
}
