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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Implements {@link TransitionFactory} to provide a starlark-defined transition that rules can
 * apply to their own configuration. This transition has access to (1) a map of the current
 * configuration's build settings and (2) the configured attributes of the given rule (not its
 * dependencies').
 *
 * <p>In some corner cases, we can't access the configured attributes the configuration of the child
 * may be different than the configuration of the parent. For now, forbid all access to attributes
 * that read selects.
 *
 * <p>For starlark-defined attribute transitions, see {@link StarlarkAttributeTransitionProvider}.
 */
public final class StarlarkRuleTransitionProvider implements TransitionFactory<RuleTransitionData> {

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  StarlarkRuleTransitionProvider(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  @VisibleForTesting
  public StarlarkDefinedConfigTransition getStarlarkDefinedConfigTransitionForTesting() {
    return starlarkDefinedConfigTransition;
  }

  @Override
  public PatchTransition create(RuleTransitionData ruleData) {
    // This wouldn't be safe if rule transitions could read attributes with select(), in which case
    // the rule alone isn't sufficient to define the transition's semantics (both the rule and its
    // configuration are needed). Rule transitions can't read select()s, so this is a non-issue.
    //
    // We could cache-optimize further by distinguishing transitions that read attributes vs. those
    // that don't. Every transition has a {@code def impl(settings, attr) } signature, even if the
    // transition never reads {@code attr}. If we had a way to formally identify such transitions,
    // we wouldn't need {@code rule} in the cache key.
    return starlarkDefinedConfigTransition.createRuleTransition(ruleData, this::createTransition);
  }

  @Override
  public TransitionType transitionType() {
    return TransitionType.RULE;
  }

  public boolean allowImmutableFlagChanges() {
    return false;
  }

  private FunctionPatchTransition createTransition(RuleTransitionData ruleData) {
    Rule rule = ruleData.rule();
    ImmutableMap<Label, ConfigMatchingProvider> configConditions = ruleData.configConditions();
    String configHash = ruleData.configHash();
    LinkedHashMap<String, Object> attributes = new LinkedHashMap<>();
    RawAttributeMapper attributeMapper = RawAttributeMapper.of(rule);
    ConfiguredAttributeMapper configuredAttributeMapper =
        ConfiguredAttributeMapper.of(rule, configConditions, configHash, false);
    ImmutableList<String> transitionOutputs = this.starlarkDefinedConfigTransition.getOutputs();

    for (Attribute attribute : rule.getAttributes()) {
      // If the value is present, even if it is null, add to the attribute map.
      Object val = attributeMapper.getRawAttributeValue(rule, attribute);
      if (val instanceof SelectorList<?> sl) {
        Result result =
            handleConfiguredAttribute(
                configConditions, configuredAttributeMapper, transitionOutputs, attribute, sl);
        if (!result.success()) {
          // Skip this attribute.
          continue;
        } else {
          val = result.resolved;
        }
      }

      attributes.put(
          Attribute.getStarlarkName(attribute.getPublicName()), Attribute.valueToStarlark(val));
    }

    StructImpl attrObject =
        StructProvider.STRUCT.create(
            attributes,
            "No attribute '%s'. Either this attribute does not exist for this rule or the attribute"
                + " was not resolved because it is set by a select that reads flags the transition"
                + " may set.");
    return new FunctionPatchTransition(attrObject);
  }

  /**
   * A container class for the result of {@link #handleConfiguredAttribute}.
   *
   * <p>The most important point is that the {@code success} field tells whether the attribute was
   * resolved. It is entirely possible to resolve an attribute to {@code null}.
   */
  private record Result(boolean success, @Nullable Object resolved) {
    static Result failure() {
      return new Result(false, null);
    }

    static Result success(@Nullable Object resolved) {
      return new Result(true, resolved);
    }
  }

  private Result handleConfiguredAttribute(
      @Nullable ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ConfiguredAttributeMapper configuredAttributeMapper,
      ImmutableList<String> transitionOutputs,
      Attribute attribute,
      SelectorList<?> val) {
    // If there are no configConditions then nothing is resolvable.
    if (configConditions == null || configConditions.isEmpty()) {
      return Result.failure();
    }

    // If any of the select keys reference the outputs, this isn't resolvable.
    if (selectBranchesReferenceOutputs(configConditions, transitionOutputs, val)) {
      return Result.failure();
    }

    // Resolve the attribute, ignoring any failures. They will be reported (and fail analysis) later
    // in the rule analysis.
    ConfiguredAttributeMapper.AttributeResolutionResult<?> result =
        configuredAttributeMapper.getResolvedAttribute(attribute);
    return switch (result.getType()) {
      case FAILURE -> Result.failure();
      case SUCCESS -> Result.success(result.getSuccess().orElse(null));
    };
  }

  private boolean selectBranchesReferenceOutputs(
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      ImmutableList<String> transitionOutputs,
      SelectorList<?> val) {
    for (Object label : val.getKeyLabels()) {
      ConfigMatchingProvider configMatchingProvider = configConditions.get(label);
      if (checkIfAttributeSelectOnAFlagTransitionChanges(
          configMatchingProvider, transitionOutputs)) {
        return true;
      }
    }
    return false;
  }

  private boolean checkIfAttributeSelectOnAFlagTransitionChanges(
      ConfigMatchingProvider configMatchingProvider, ImmutableList<String> transitionOutputs) {
    // check settingMap
    Set<String> nativeFlagLabels = new HashSet<>();
    for (String key : configMatchingProvider.settingsMap().keySet()) {
      String modified = "//command_line_option:" + key;
      nativeFlagLabels.add(modified);
    }
    // check flags values
    ImmutableMap<Label, String> flagSettingsMap = configMatchingProvider.flagSettingsMap();
    Set<String> flagLabels = new HashSet<>();
    for (Label flag : flagSettingsMap.keySet()) {
      flagLabels.add(flag.getCanonicalForm());
    }

    for (String output : transitionOutputs) {
      if (nativeFlagLabels.contains(output) || flagLabels.contains(output)) {
        return true;
      }
    }
    return false;
  }

  /** The actual transition used by the rule. */
  private final class FunctionPatchTransition extends StarlarkTransition
      implements PatchTransition {
    private final StructImpl attrObject;
    private final int hashCode;

    private FunctionPatchTransition(StructImpl attrObject) {
      super(starlarkDefinedConfigTransition);
      this.attrObject = attrObject;
      this.hashCode = Objects.hash(attrObject, super.hashCode());
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
      // Starlark transitions already have logic to enforce they only access declared inputs and
      // outputs. Rather than complicate BuildOptionsView with more access points to BuildOptions,
      // we just use the original BuildOptions and trust the transition's enforcement logic.
      BuildOptions buildOptions = buildOptionsView.underlying();
      Map<String, BuildOptions> result =
          applyAndValidate(
              buildOptions,
              starlarkDefinedConfigTransition,
              allowImmutableFlagChanges(),
              attrObject,
              eventHandler);
      if (result == null) {
        return buildOptions.clone();
      }
      if (result.size() != 1) {
        eventHandler.handle(
            Event.error(
                starlarkDefinedConfigTransition.getLocation(),
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
      if (!(object instanceof FunctionPatchTransition other)) {
        return false;
      }
      return Objects.equals(attrObject, other.attrObject) && super.equals(other);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }
}
