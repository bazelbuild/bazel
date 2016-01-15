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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.NativeAspectClass.NativeAspectFactory;
import com.google.devtools.build.lib.util.BinaryPredicate;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * The definition of an aspect (see {@link Aspect} for moreinformation.)
 *
 * <p>Contains enough information to build up the configured target graph except for the actual way
 * to build the Skyframe node (that is the territory of
 * {@link com.google.devtools.build.lib.view AspectFactory}). In particular:
 * <ul>
 *   <li>The condition that must be fulfilled for an aspect to be able to operate on a configured
 *       target
 *   <li>The (implicit or late-bound) attributes of the aspect that denote dependencies the aspect
 *       itself needs (e.g. runtime libraries for a new language for protocol buffers)
 *   <li>The aspects this aspect requires from its direct dependencies
 * </ul>
 *
 * <p>The way to build the Skyframe node is not here because this data needs to be accessible from
 * the {@code .packages} package and that one requires references to the {@code .view} package.
 */
@Immutable
public final class AspectDefinition {

  private final String name;
  private final ImmutableSet<Class<?>> requiredProviders;
  private final ImmutableSet<String> requiredProviderNames;
  private final ImmutableMap<String, Attribute> attributes;
  private final ImmutableMultimap<String, AspectClass> attributeAspects;
  @Nullable private final ConfigurationFragmentPolicy configurationFragmentPolicy;

  private AspectDefinition(
      String name,
      ImmutableSet<Class<?>> requiredProviders,
      ImmutableMap<String, Attribute> attributes,
      ImmutableMultimap<String, AspectClass> attributeAspects,
      @Nullable ConfigurationFragmentPolicy configurationFragmentPolicy) {
    this.name = name;
    this.requiredProviders = requiredProviders;
    this.requiredProviderNames = toStringSet(requiredProviders);
    this.attributes = attributes;
    this.attributeAspects = attributeAspects;
    this.configurationFragmentPolicy = configurationFragmentPolicy;
  }

  public String getName() {
    return name;
  }

  /**
   * Returns the attributes of the aspect in the form of a String -&gt; {@link Attribute} map.
   *
   * <p>All attributes are either implicit or late-bound.
   */
  public ImmutableMap<String, Attribute> getAttributes() {
    return attributes;
  }

  /**
   * Returns the set of {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider}
   * instances that must be present on a configured target so that this aspect can be applied to it.
   *
   * <p>We cannot refer to that class here due to our dependency structure, so this returns a set
   * of unconstrained class objects.
   *
   * <p>If a configured target does not have a required provider, the aspect is silently not created
   * for it.
   */
  public ImmutableSet<Class<?>> getRequiredProviders() {
    return requiredProviders;
  }

  /**
   * Returns the set of class names of
   * {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider} instances that must be
   * present on a configured target so that this aspect can be applied to it.
   *
   * <p>This set is a mirror of the set returned by {@link #getRequiredProviders}, but contains the
   * names of the classes rather than the class objects themselves.
   *
   * <p>If a configured target does not have a required provider, the aspect is silently not created
   * for it.
   */
  public ImmutableSet<String> getRequiredProviderNames() {
    return requiredProviderNames;
  }

  /**
   * Returns the attribute -&gt; set of required aspects map.
   */
  public ImmutableMultimap<String, AspectClass> getAttributeAspects() {
    return attributeAspects;
  }

  /**
   * Returns the set of configuration fragments required by this Aspect, or {@code null} if it has
   * not set a configuration fragment policy, meaning it should inherit from the attached rule.
   */
  @Nullable public ConfigurationFragmentPolicy getConfigurationFragmentPolicy() {
    // TODO(mstaib): When all existing aspects properly set their configuration fragment policy,
    // this method and the associated member should no longer be nullable.
    // "inherit from the attached rule" should go away.
    return configurationFragmentPolicy;
  }

  /**
   * Returns the attribute -&gt; set of labels that are provided by aspects of attribute.
   */
  public static ImmutableMultimap<Attribute, Label> visitAspectsIfRequired(
      Target from, Attribute attribute, Target to) {
    // Aspect can be declared only for Rules.
    if (!(from instanceof Rule) || !(to instanceof Rule)) {
      return ImmutableMultimap.of();
    }
    RuleClass ruleClass = ((Rule) to).getRuleClassObject();
    ImmutableSet<Class<?>> providers = ruleClass.getAdvertisedProviders();
    return visitAspectsIfRequired((Rule) from, attribute, toStringSet(providers));
  }

  /**
   * Returns the attribute -&gt; set of labels that are provided by aspects of attribute.
   */
  public static ImmutableMultimap<Attribute, Label> visitAspectsIfRequired(
      Rule from, Attribute attribute, Set<String> advertisedProviders) {
    if (advertisedProviders.isEmpty()) {
      return ImmutableMultimap.of();
    }

    LinkedHashMultimap<Attribute, Label> result = LinkedHashMultimap.create();
    for (Aspect candidateClass : attribute.getAspects(from)) {
      // Check if target satisfies condition for this aspect (has to provide all required
      // TransitiveInfoProviders)
      if (!advertisedProviders.containsAll(
          candidateClass.getDefinition().getRequiredProviderNames())) {
        continue;
      }
      addAllAttributesOfAspect(from, result, candidateClass.getDefinition(), Rule.ALL_DEPS);
    }
    return ImmutableMultimap.copyOf(result);
  }

  private static ImmutableSet<String> toStringSet(ImmutableSet<Class<?>> classes) {
    ImmutableSet.Builder<String> classStrings = new ImmutableSet.Builder<>();
    for (Class<?> clazz : classes) {
      classStrings.add(clazz.getName());
    }
    return classStrings.build();
  }

  /**
   * Collects all attribute labels from the specified aspectDefinition.
   */
  public static void addAllAttributesOfAspect(Rule from,
      Multimap<Attribute, Label> labelBuilder, AspectDefinition aspectDefinition,
      BinaryPredicate<Rule, Attribute> predicate) {
    ImmutableMap<String, Attribute> attributes = aspectDefinition.getAttributes();
    for (Attribute aspectAttribute : attributes.values()) {
      if (!predicate.apply(from, aspectAttribute)) {
        continue;
      }
      if (aspectAttribute.getType() == BuildType.LABEL) {
        Label label = BuildType.LABEL.cast(aspectAttribute.getDefaultValue(from));
        if (label != null) {
          labelBuilder.put(aspectAttribute, label);
        }
      } else if (aspectAttribute.getType() == BuildType.LABEL_LIST) {
        List<Label> labelList = BuildType.LABEL_LIST.cast(aspectAttribute.getDefaultValue(from));
        labelBuilder.putAll(aspectAttribute, labelList);
      }
    }
  }

  /**
   * Builder class for {@link AspectDefinition}.
   */
  public static final class Builder {
    private final String name;
    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private final Set<Class<?>> requiredProviders = new LinkedHashSet<>();
    private final Multimap<String, AspectClass> attributeAspects = LinkedHashMultimap.create();
    private final ConfigurationFragmentPolicy.Builder configurationFragmentPolicy =
        new ConfigurationFragmentPolicy.Builder();
    // TODO(mstaib): When all existing aspects properly set their configuration fragment policy,
    // remove this flag and the code that interacts with it.
    /**
     * True if the aspect definition has intentionally specified a configuration fragment policy by
     * calling any of the methods which set up the policy, and thus needs the built AspectDefinition
     * to retain the policy.
     */
    private boolean hasConfigurationFragmentPolicy = false;

    public Builder(String name) {
      this.name = name;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply the specified provider.
     */
    public Builder requireProvider(Class<?> requiredProvider) {
      this.requiredProviders.add(requiredProvider);
      return this;
    }

    /**
     * Declares that this aspect depends on the given aspects in {@code aspectFactories} provided
     * by direct dependencies through attribute {@code attribute} on the target associated with this
     * aspect.
     *
     * <p>Note that {@code ConfiguredAspectFactory} instances are expected in the second argument,
     * but we cannot reference that interface here.
     */
    @SafeVarargs
    public final Builder attributeAspect(
        String attribute, Class<? extends NativeAspectFactory>... aspectFactories) {
      Preconditions.checkNotNull(attribute);
      for (Class<? extends NativeAspectFactory> aspectFactory : aspectFactories) {
        this
            .attributeAspect(
                attribute, new NativeAspectClass<>(Preconditions.checkNotNull(aspectFactory)));
      }
      return this;
    }

    /**
     * Declares that this aspect depends on the given {@link AspectClass} provided
     * by direct dependencies through attribute {@code attribute} on the target associated with this
     * aspect.
     */
    public final Builder attributeAspect(String attribute, AspectClass aspectClass) {
      Preconditions.checkNotNull(attribute);

      this.attributeAspects.put(attribute, Preconditions.checkNotNull(aspectClass));

      return this;
    }

    /**
     * Adds an attribute to the aspect.
     *
     * <p>Since aspects do not appear in BUILD files, the attribute must be either implicit
     * (not available in the BUILD file, starting with '$') or late-bound (determined after the
     * configuration is available, starting with ':')
     */
    public <TYPE> Builder add(Attribute.Builder<TYPE> attr) {
      Attribute attribute = attr.build();
      return add(attribute);
    }

    /**
     * Adds an attribute to the aspect.
     *
     * <p>Since aspects do not appear in BUILD files, the attribute must be either implicit
     * (not available in the BUILD file, starting with '$') or late-bound (determined after the
     * configuration is available, starting with ':')
     */
    public Builder add(Attribute attribute) {
      Preconditions.checkArgument(attribute.isImplicit() || attribute.isLateBound());
      Preconditions.checkArgument(!attributes.containsKey(attribute.getName()),
          "An attribute with the name '%s' already exists.", attribute.getName());
      attributes.put(attribute.getName(), attribute);
      return this;
    }

    /**
     * Declares that the implementation of the associated aspect definition requires the given
     * fragments to be present in this rule's host and target configurations.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(Class<?>... configurationFragments) {
      hasConfigurationFragmentPolicy = true;
      configurationFragmentPolicy
          .requiresConfigurationFragments(ImmutableSet.copyOf(configurationFragments));
      return this;
    }

    /**
     * Declares that the implementation of the associated aspect definition requires the given
     * fragments to be present in the host configuration.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresHostConfigurationFragments(Class<?>... configurationFragments) {
      hasConfigurationFragmentPolicy = true;
      configurationFragmentPolicy
          .requiresHostConfigurationFragments(ImmutableSet.copyOf(configurationFragments));
      return this;
    }

    /**
     * Declares the configuration fragments that are required by this rule for the target
     * configuration.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Class...)}, this method takes the
     * Skylark module names of fragments instead of their classes.
     */
    public Builder requiresConfigurationFragmentsBySkylarkModuleName(
        Collection<String> configurationFragmentNames) {
      // This method is unconditionally called from Skylark code, so only consider the user to have
      // specified a configuration policy if the collection actually has anything in it.
      // TODO(mstaib): Stop caring about this as soon as all aspects have configuration policies.
      hasConfigurationFragmentPolicy =
          hasConfigurationFragmentPolicy || !configurationFragmentNames.isEmpty();
      configurationFragmentPolicy
          .requiresConfigurationFragmentsBySkylarkModuleName(configurationFragmentNames);
      return this;
    }

    /**
     * Declares the configuration fragments that are required by this rule for the host
     * configuration.
     *
     * <p>In contrast to {@link #requiresHostConfigurationFragments(Class...)}, this method takes
     * the Skylark module names of fragments instead of their classes.
     */
    public Builder requiresHostConfigurationFragmentsBySkylarkModuleName(
        Collection<String> configurationFragmentNames) {
      // This method is unconditionally called from Skylark code, so only consider the user to have
      // specified a configuration policy if the collection actually has anything in it.
      // TODO(mstaib): Stop caring about this as soon as all aspects have configuration policies.
      hasConfigurationFragmentPolicy =
          hasConfigurationFragmentPolicy || !configurationFragmentNames.isEmpty();
      configurationFragmentPolicy
          .requiresHostConfigurationFragmentsBySkylarkModuleName(configurationFragmentNames);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing required fragments (see
     * {@link #requiresConfigurationFragments}).
     */
    public Builder setMissingFragmentPolicy(MissingFragmentPolicy missingFragmentPolicy) {
      hasConfigurationFragmentPolicy = true;
      configurationFragmentPolicy.setMissingFragmentPolicy(missingFragmentPolicy);
      return this;
    }

    /**
     * Builds the aspect definition.
     *
     * <p>The builder object is reusable afterwards.
     */
    public AspectDefinition build() {
      return new AspectDefinition(name, ImmutableSet.copyOf(requiredProviders),
          ImmutableMap.copyOf(attributes), ImmutableSetMultimap.copyOf(attributeAspects),
          hasConfigurationFragmentPolicy ? configurationFragmentPolicy.build() : null);
    }
  }
}
