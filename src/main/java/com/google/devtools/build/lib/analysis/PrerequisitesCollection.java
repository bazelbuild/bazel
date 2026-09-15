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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Collection of the attributes dependencies available in a {@link RuleContext}. */
public final class PrerequisitesCollection {

  private interface AttributesMapper {
    Attribute getAttribute(String attributeName);
  }

  private final ImmutableSortedKeyListMultimap<String, ConfiguredTargetAndData>
      attributeToPrerequisitesMap;
  private final AttributesMapper attributesMapper;

  private final RuleErrorConsumer ruleErrorConsumer;
  private final Rule rule;
  private final String ruleClassNameForLogging;

  PrerequisitesCollection(
      ImmutableSortedKeyListMultimap<String, ConfiguredTargetAndData> attributeToPrerequisitesMap,
      ImmutableMap<String, Attribute> attributes,
      RuleErrorConsumer ruleErrorConsumer,
      Rule rule,
      String ruleClassNameForLogging) {
    this(
        attributeToPrerequisitesMap,
        /* attributesMapper= */ attributes::get,
        ruleErrorConsumer,
        rule,
        ruleClassNameForLogging);
  }

  PrerequisitesCollection(
      ImmutableSortedKeyListMultimap<String, ConfiguredTargetAndData> attributeToPrerequisitesMap,
      AttributeMap attributes,
      RuleErrorConsumer ruleErrorConsumer,
      Rule rule,
      String ruleClassNameForLogging) {
    this(
        attributeToPrerequisitesMap,
        /* attributesMapper= */ attributes::getAttributeDefinition,
        ruleErrorConsumer,
        rule,
        ruleClassNameForLogging);
  }

  private PrerequisitesCollection(
      ImmutableSortedKeyListMultimap<String, ConfiguredTargetAndData> attributeToPrerequisitesMap,
      AttributesMapper attributesMapper,
      RuleErrorConsumer ruleErrorConsumer,
      Rule rule,
      String ruleClassNameForLogging) {
    this.attributeToPrerequisitesMap = attributeToPrerequisitesMap;
    this.attributesMapper = attributesMapper;
    this.ruleErrorConsumer = ruleErrorConsumer;
    this.rule = rule;
    this.ruleClassNameForLogging = ruleClassNameForLogging;
  }

  boolean has(String attributeName) {
    return attributesMapper.getAttribute(attributeName) != null;
  }

  /** Returns a list of all prerequisites as {@code ConfiguredTarget} objects. */
  public ImmutableList<? extends TransitiveInfoCollection> getAllPrerequisites() {
    return attributeToPrerequisitesMap.values().stream()
        .map(ConfiguredTargetAndData::getConfiguredTarget)
        .collect(toImmutableList());
  }

  /** Returns the {@link ConfiguredTargetAndData} the given attribute. */
  public List<ConfiguredTargetAndData> getPrerequisiteConfiguredTargets(String attributeName) {
    return attributeToPrerequisitesMap.get(attributeName);
  }

  /**
   * Returns the list of transitive info collections that feed into this target through the
   * specified attribute.
   */
  public List<? extends TransitiveInfoCollection> getPrerequisites(String attributeName) {
    Attribute attribute = attributesMapper.getAttribute(attributeName);
    if (attribute == null) {
      return ImmutableList.of();
    }

    List<ConfiguredTargetAndData> prerequisiteConfiguredTargets;
    // android_binary, android_test, and android_binary_internal override deps to use a split
    // transition.
    if ((rule.getRuleClass().equals("android_binary")
            || rule.getRuleClass().equals("android_test")
            || rule.getRuleClass().equals("android_binary_internal"))
        && attributeName.equals("deps")
        && attribute.getTransitionFactory().isSplit()) {
      // TODO(b/168038145): Restore legacy behavior of returning the prerequisites from the first
      // portion of the split transition.
      // Callers should be identified, cleaned up, and this check removed.
      Map<Optional<String>, List<ConfiguredTargetAndData>> map =
          getSplitPrerequisites(attributeName);
      prerequisiteConfiguredTargets =
          map.isEmpty() ? ImmutableList.of() : map.entrySet().iterator().next().getValue();
    } else {
      prerequisiteConfiguredTargets = getPrerequisiteConfiguredTargets(attributeName);
    }

    return Lists.transform(
        prerequisiteConfiguredTargets, ConfiguredTargetAndData::getConfiguredTarget);
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file.
   */
  public <C extends TransitiveInfoProvider> List<C> getPrerequisites(
      String attributeName, Class<C> classType) {
    AnalysisUtils.checkProvider(classType);
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), classType);
  }

  /**
   * Returns all the declared Starlark wrapped providers for the specified constructor under the
   * specified attribute of this target in the BUILD file.
   */
  public <T> ImmutableList<T> getPrerequisites(
      String attributeName, StarlarkProviderWrapper<T> starlarkKey) throws RuleErrorException {
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), starlarkKey);
  }

  /**
   * Returns all the declared providers (native and Starlark) for the specified constructor under
   * the specified attribute of this target in the BUILD file.
   */
  public <T extends Info> List<T> getPrerequisites(
      String attributeName, BuiltinProvider<T> builtinProvider) {
    return AnalysisUtils.getProviders(getPrerequisites(attributeName), builtinProvider);
  }

  /**
   * Returns the prerequisites keyed by their transition keys. If the split transition is not active
   * (e.g. split() returned an empty list), the key is an empty Optional.
   */
  public Map<Optional<String>, List<ConfiguredTargetAndData>> getSplitPrerequisites(
      String attributeName) {
    checkAttributeIsDependency(attributeName);
    // Use an ImmutableListMultimap.Builder here to preserve ordering.
    ImmutableListMultimap.Builder<Optional<String>, ConfiguredTargetAndData> result =
        ImmutableListMultimap.builder();
    List<ConfiguredTargetAndData> deps = getPrerequisiteConfiguredTargets(attributeName);
    for (ConfiguredTargetAndData t : deps) {
      ImmutableList<String> transitionKeys = t.getTransitionKeys();
      if (transitionKeys.isEmpty()) {
        // The split transition is not active, i.e. does not change build configurations.
        // TODO(jungjw): Investigate if we need to do a check here.
        return ImmutableMap.of(Optional.absent(), deps);
      }
      for (String key : transitionKeys) {
        result.put(Optional.of(key), t);
      }
    }
    return Multimaps.asMap(result.build());
  }

  /**
   * Returns the transitive info collection that feeds into this target through the specified
   * attribute. Returns null if the attribute is empty.
   */
  @Nullable
  public TransitiveInfoCollection getPrerequisite(String attributeName) {
    checkAttributeIsDependency(attributeName);
    List<ConfiguredTargetAndData> elements = getPrerequisiteConfiguredTargets(attributeName);
    Preconditions.checkState(
        elements.size() <= 1,
        "%s attribute %s produces more than one prerequisite",
        ruleClassNameForLogging,
        attributeName);
    return elements.isEmpty() ? null : elements.get(0).getConfiguredTarget();
  }

  /**
   * Returns the declared provider (native and Starlark) for the specified constructor under the
   * specified attribute of this target in the BUILD file. May return null if there is no
   * TransitiveInfoCollection under the specified attribute.
   */
  @Nullable
  public <T extends Info> T getPrerequisite(
      String attributeName, BuiltinProvider<T> builtinProvider) {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.get(builtinProvider);
  }

  /**
   * Returns the specified provider of the prerequisite referenced by the attribute in the argument.
   * If the attribute is empty or it does not support the specified provider, returns null.
   */
  @Nullable
  public <C extends TransitiveInfoProvider> C getPrerequisite(
      String attributeName, Class<C> provider) {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.getProvider(provider);
  }

  @Nullable
  public <T> T getPrerequisite(String attributeName, StarlarkProviderWrapper<T> key)
      throws RuleErrorException {
    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    return prerequisite == null ? null : prerequisite.get(key);
  }

  /**
   * For the specified attribute "attributeName" (which must be of type label), resolves the
   * ConfiguredTarget and returns its single build artifact.
   *
   * <p>If the attribute is optional, has no default and was not specified, then null will be
   * returned. Note also that null is returned (and an attribute error is raised) if there wasn't
   * exactly one build artifact for the target.
   */
  public Artifact getPrerequisiteArtifact(String attributeName) {
    TransitiveInfoCollection target = getPrerequisite(attributeName);
    return transitiveInfoCollectionToArtifact(attributeName, target);
  }

  @Nullable
  private Artifact transitiveInfoCollectionToArtifact(
      String attributeName, TransitiveInfoCollection target) {
    if (target != null) {
      NestedSet<Artifact> artifacts = target.getProvider(FileProvider.class).getFilesToBuild();
      if (artifacts.isSingleton()) {
        return artifacts.getSingleton();
      } else {
        ruleErrorConsumer.attributeError(
            attributeName, target.getLabel() + " expected a single artifact");
      }
    }
    return null;
  }

  /**
   * Returns the prerequisite referred to by the specified attribute. Also checks whether the
   * attribute is marked as executable and that the target referred to can actually be executed.
   *
   * @param attributeName the name of the attribute
   * @return the {@link FilesToRunProvider} interface of the prerequisite.
   */
  @Nullable
  public FilesToRunProvider getExecutablePrerequisite(String attributeName) {
    Attribute ruleDefinition = attributesMapper.getAttribute(attributeName);

    Preconditions.checkNotNull(
        ruleDefinition, "%s attribute %s is not defined", ruleClassNameForLogging, attributeName);
    Preconditions.checkState(
        ruleDefinition.isExecutable(),
        "%s attribute %s is not configured to be executable",
        ruleClassNameForLogging,
        attributeName);

    TransitiveInfoCollection prerequisite = getPrerequisite(attributeName);
    if (prerequisite == null) {
      return null;
    }

    FilesToRunProvider result = prerequisite.getProvider(FilesToRunProvider.class);
    if (result == null || result.getExecutable() == null) {
      ruleErrorConsumer.attributeError(
          attributeName, prerequisite.getLabel() + " does not refer to a valid executable target");
    }
    return result;
  }

  /**
   * Returns all the providers of the specified type that are listed under the specified attribute
   * of this target in the BUILD file, and that contain the specified provider.
   */
  public <C extends TransitiveInfoProvider>
      Iterable<? extends TransitiveInfoCollection> getPrerequisitesIf(
          String attributeName, Class<C> classType) {
    AnalysisUtils.checkProvider(classType);
    return AnalysisUtils.filterByProvider(getPrerequisites(attributeName), classType);
  }

  private void checkAttributeIsDependency(String attributeName) {
    Attribute attributeDefinition = attributesMapper.getAttribute(attributeName);
    Preconditions.checkNotNull(
        attributeDefinition,
        "%s: %s attribute %s is not defined",
        rule.getLocation(),
        ruleClassNameForLogging,
        attributeName);
    Preconditions.checkState(
        attributeDefinition.getType().getLabelClass() == LabelClass.DEPENDENCY,
        "%s attribute %s is not a label type attribute",
        ruleClassNameForLogging,
        attributeName);
  }
}
