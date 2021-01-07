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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.devtools.build.lib.analysis.starlark.FunctionTransitionUtil.applyAndValidate;
import static com.google.devtools.build.lib.analysis.starlark.StarlarkAttributesCollection.ERROR_MESSAGE_FOR_NO_ATTR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import java.util.LinkedHashMap;
import java.util.Map;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;

/**
 * This class implements {@link TransitionFactory} to provide a starlark-defined transition that
 * rules can apply to their dependencies' configurations. This transition has access to (1) the a
 * map of the current configuration's build settings and (2) the configured attributes of the given
 * rule (not its dependencies').
 *
 * <p>For starlark defined rule class transitions, see {@link StarlarkRuleTransitionProvider}.
 *
 * <p>TODO(bazel-team): Consider allowing dependency-typed attributes to actually return providers
 * instead of just labels (see {@link StarlarkAttributesCollection#addAttribute}).
 */
public class StarlarkAttributeTransitionProvider
    implements TransitionFactory<AttributeTransitionData>, SplitTransitionProviderApi {
  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  StarlarkAttributeTransitionProvider(
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  @VisibleForTesting
  public StarlarkDefinedConfigTransition getStarlarkDefinedConfigTransitionForTesting() {
    return starlarkDefinedConfigTransition;
  }

  @Override
  public SplitTransition create(AttributeTransitionData data) {
    AttributeMap attributeMap = data.attributes();
    Preconditions.checkArgument(attributeMap instanceof ConfiguredAttributeMapper);
    return new FunctionSplitTransition(
        starlarkDefinedConfigTransition, (ConfiguredAttributeMapper) attributeMap);
  }

  @Override
  public boolean isSplit() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<transition object>");
  }

  class FunctionSplitTransition extends StarlarkTransition implements SplitTransition {
    private final StructImpl attrObject;

    FunctionSplitTransition(
        StarlarkDefinedConfigTransition starlarkDefinedConfigTransition,
        ConfiguredAttributeMapper attributeMap) {
      super(starlarkDefinedConfigTransition);

      LinkedHashMap<String, Object> attributes = new LinkedHashMap<>();
      for (String attribute : attributeMap.getAttributeNames()) {
        Object val = attributeMap.get(attribute, attributeMap.getAttributeType(attribute));
        attributes.put(Attribute.getStarlarkName(attribute), Attribute.valueToStarlark(val));
      }
      attrObject = StructProvider.STRUCT.create(attributes, ERROR_MESSAGE_FOR_NO_ATTR);
    }

    /**
     * @return the post-transition build options or a clone of the original build options if an
     *     error was encountered during transition application/validation.
     */
    @Override
    public final Map<String, BuildOptions> split(
        BuildOptionsView buildOptionsView, EventHandler eventHandler) throws InterruptedException {
      // Starlark transitions already have logic to enforce they only access declared inputs and
      // outputs. Rather than complicate BuildOptionsView with more access points to BuildOptions,
      // we just use the original BuildOptions and trust the transition's enforcement logic.
      BuildOptions buildOptions = buildOptionsView.underlying();
      try {
        return applyAndValidate(
            buildOptions, starlarkDefinedConfigTransition, attrObject, eventHandler);
      } catch (EvalException e) {
        eventHandler.handle(
            Event.error(
                starlarkDefinedConfigTransition.getLocationForErrorReporting(),
                e.getMessageWithStack()));
        return ImmutableMap.of("error", buildOptions.clone());
      }
    }
  }
}
