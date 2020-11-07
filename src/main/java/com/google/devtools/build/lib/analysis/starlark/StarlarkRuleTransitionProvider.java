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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import net.starlark.java.eval.EvalException;

/**
 * This class implements {@link TransitionFactory} to provide a starlark-defined transition that
 * rules can apply to their own configuration. This transition has access to (1) the a map of the
 * current configuration's build settings and (2) the configured* attributes of the given rule (not
 * its dependencies').
 *
 * <p>*In some corner cases, we can't access the configured attributes the configuration of the
 * child may be different than the configuration of the parent. For now, forbid all access to
 * attributes that read selects.
 *
 * <p>For starlark-defined attribute transitions, see {@link StarlarkAttributeTransitionProvider}.
 */
public class StarlarkRuleTransitionProvider implements TransitionFactory<Rule> {

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  StarlarkRuleTransitionProvider(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  @VisibleForTesting
  public StarlarkDefinedConfigTransition getStarlarkDefinedConfigTransitionForTesting() {
    return starlarkDefinedConfigTransition;
  }

  @Override
  public PatchTransition create(Rule rule) {
    return new FunctionPatchTransition(starlarkDefinedConfigTransition, rule);
  }

  @Override
  public boolean isSplit() {
    // The transitions returned by this factory are guaranteed not to be splits.
    return false;
  }

  /** The actual transition used by the rule. */
  class FunctionPatchTransition extends StarlarkTransition implements PatchTransition {
    private final StructImpl attrObject;

    FunctionPatchTransition(
        StarlarkDefinedConfigTransition starlarkDefinedConfigTransition, Rule rule) {
      super(starlarkDefinedConfigTransition);
      LinkedHashMap<String, Object> attributes = new LinkedHashMap<>();
      RawAttributeMapper attributeMapper = RawAttributeMapper.of(rule);
      for (Attribute attribute : rule.getAttributes()) {
        Object val = attributeMapper.getRawAttributeValue(rule, attribute);
        if (val instanceof BuildType.SelectorList) {
          // For now, don't allow access to attributes that read selects.
          // TODO(b/121134880): make this restriction more fine grained.
          continue;
        }
        attributes.put(
            Attribute.getStarlarkName(attribute.getPublicName()), Attribute.valueToStarlark(val));
      }
      attrObject =
          StructProvider.STRUCT.create(
              attributes,
              "No attribute '%s'. Either this attribute does "
                  + "not exist for this rule or is set by a select. Starlark rule transitions "
                  + "currently cannot read attributes behind selects.");
    }

    /**
     * @return the post-transition build options or a clone of the original build options if an
     *     error was encountered during transition application/validation.
     */
    // TODO(b/121134880): validate that the targets these transitions are applied on don't read any
    // attributes that are then configured by the outputs of these transitions.
    @Override
    public BuildOptions patch(BuildOptionsView buildOptionsView, EventHandler eventHandler)
        throws InterruptedException {
      Map<String, BuildOptions> result;
      // Starlark transitions already have logic to enforce they only access declared inputs and
      // outputs. Rather than complicate BuildOptionsView with more access points to BuildOptions,
      // we just use the original BuildOptions and trust the transition's enforcement logic.
      BuildOptions buildOptions = buildOptionsView.underlying();
      try {
        result =
            applyAndValidate(
                buildOptions, starlarkDefinedConfigTransition, attrObject, eventHandler);
      } catch (EvalException e) {
        eventHandler.handle(
            Event.error(
                starlarkDefinedConfigTransition.getLocationForErrorReporting(),
                e.getMessageWithStack()));
        return buildOptions.clone();
      }
      if (result.size() != 1) {
        eventHandler.handle(
            Event.error(
                starlarkDefinedConfigTransition.getLocationForErrorReporting(),
                "Rule transition only allowed to return a single transitioned configuration."));
        return buildOptions.clone();
      }
      return Iterables.getOnlyElement(result.values());
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      }
      if (!(object instanceof FunctionPatchTransition)) {
        return false;
      }
      FunctionPatchTransition other = (FunctionPatchTransition) object;
      return Objects.equals(attrObject, other.attrObject) && super.equals(other);
    }

    @Override
    public int hashCode() {
      return Objects.hash(attrObject, super.hashCode());
    }
  }
}
