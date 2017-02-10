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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.LabelClass;
import com.google.devtools.build.lib.syntax.Type.LabelVisitor;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
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
  private final AspectClass aspectClass;
  private final AdvertisedProviderSet advertisedProviders;
  private final RequiredProviders requiredProviders;
  private final RequiredProviders requiredProvidersForAspects;
  private final ImmutableMap<String, Attribute> attributes;

  /**
   * Which attributes aspect should propagate along:
   * <ul>
   *  <li>A {@code null} value means propagate along all attributes</li>
   *  <li>A (possibly empty) set means to propagate only along the attributes in a set</li>
   * </ul>
   */
  @Nullable private final ImmutableSet<String> restrictToAttributes;
  @Nullable private final ConfigurationFragmentPolicy configurationFragmentPolicy;

  public AdvertisedProviderSet getAdvertisedProviders() {
    return advertisedProviders;
  }


  private AspectDefinition(
      AspectClass aspectClass,
      AdvertisedProviderSet advertisedProviders,
      RequiredProviders requiredProviders,
      RequiredProviders requiredAspectProviders,
      ImmutableMap<String, Attribute> attributes,
      @Nullable ImmutableSet<String> restrictToAttributes,
      @Nullable ConfigurationFragmentPolicy configurationFragmentPolicy) {
    this.aspectClass = aspectClass;
    this.advertisedProviders = advertisedProviders;
    this.requiredProviders = requiredProviders;
    this.requiredProvidersForAspects = requiredAspectProviders;

    this.attributes = attributes;
    this.restrictToAttributes = restrictToAttributes;
    this.configurationFragmentPolicy = configurationFragmentPolicy;
  }

  public String getName() {
    return aspectClass.getName();
  }

  public AspectClass getAspectClass() {
    return aspectClass;
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
   * Returns {@link RequiredProviders} that a configured target must have so that
   * this aspect can be applied to it.
   *
   * <p>If a configured target does not satisfy required providers, the aspect is
   * silently not created for it.
   */
  public RequiredProviders getRequiredProviders() {
    return requiredProviders;
  }

  /**
   * Aspects do not depend on other aspects applied to the same target <em>unless</em>
   * the other aspect satisfies the {@link RequiredProviders} this method returns
   */
  public RequiredProviders getRequiredProvidersForAspects() {
    return requiredProvidersForAspects;
  }


  /**
   * Returns the set of required aspects for a given attribute.
   */
  public boolean propagateAlong(Attribute attribute) {
    if (restrictToAttributes != null) {
      return restrictToAttributes.contains(attribute.getName());
    }
    return true;
  }

  /**
   * Returns the set of configuration fragments required by this Aspect.
   */
  public ConfigurationFragmentPolicy getConfigurationFragmentPolicy() {
    return configurationFragmentPolicy;
  }

  /**
   * Returns the attribute -&gt; set of labels that are provided by aspects of attribute.
   */
  public static ImmutableMultimap<Attribute, Label> visitAspectsIfRequired(
      Target from, Attribute attribute, Target to,
      DependencyFilter dependencyFilter) {
    // Aspect can be declared only for Rules.
    if (!(from instanceof Rule) || !(to instanceof Rule)) {
      return ImmutableMultimap.of();
    }
    RuleClass ruleClass = ((Rule) to).getRuleClassObject();
    AdvertisedProviderSet providers = ruleClass.getAdvertisedProviders();
    return visitAspectsIfRequired((Rule) from, attribute,
        providers, dependencyFilter);
  }

  /**
   * Returns the attribute -&gt; set of labels that are provided by aspects of attribute.
   */
  public static ImmutableMultimap<Attribute, Label> visitAspectsIfRequired(
      Rule from, Attribute attribute,
      AdvertisedProviderSet advertisedProviders,
      DependencyFilter dependencyFilter) {
    SetMultimap<Attribute, Label> result = LinkedHashMultimap.create();
    for (Aspect candidateClass : attribute.getAspects(from)) {
      // Check if target satisfies condition for this aspect (has to provide all required
      // TransitiveInfoProviders)
      RequiredProviders requiredProviders =
          candidateClass.getDefinition().getRequiredProviders();
      if (requiredProviders.isSatisfiedBy(advertisedProviders)) {
        addAllAttributesOfAspect(from, result, candidateClass, dependencyFilter);
      }
    }
    return ImmutableMultimap.copyOf(result);
  }

  @Nullable
  private static Label maybeGetRepositoryRelativeLabel(Rule from, @Nullable Label label) {
    return label == null ? null : from.getLabel().resolveRepositoryRelative(label);
  }

  /**
   * Collects all attribute labels from the specified aspectDefinition.
   */
  public static void addAllAttributesOfAspect(
      final Rule from,
      final Multimap<Attribute, Label> labelBuilder,
      Aspect aspect,
      DependencyFilter dependencyFilter) {
    ImmutableMap<String, Attribute> attributes = aspect.getDefinition().getAttributes();
    for (final Attribute aspectAttribute : attributes.values()) {
      if (!dependencyFilter.apply(aspect, aspectAttribute)) {
        continue;
      }
      Type type = aspectAttribute.getType();
      if (type.getLabelClass() != LabelClass.DEPENDENCY) {
        continue;
      }
      try {
        type.visitLabels(
            new LabelVisitor() {
              @Override
              public void visit(Label label) {
                Label repositoryRelative = maybeGetRepositoryRelativeLabel(from, label);
                if (repositoryRelative == null) {
                  return;
                }
                labelBuilder.put(aspectAttribute, repositoryRelative);
              }
            },
            aspectAttribute.getDefaultValue(from));
      } catch (InterruptedException ex) {
        // Because the LabelVisitor does not throw InterruptedException, it should not be thrown
        // by visitLabels here.
        throw new AssertionError(ex);
      }
    }
  }

  /**
   * Builder class for {@link AspectDefinition}.
   */
  public static final class Builder {
    private final AspectClass aspectClass;
    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private final AdvertisedProviderSet.Builder advertisedProviders =
        AdvertisedProviderSet.builder();
    private RequiredProviders.Builder requiredProviders = RequiredProviders.acceptAnyBuilder();
    private RequiredProviders.Builder requiredAspectProviders =
        RequiredProviders.acceptNoneBuilder();
    @Nullable
    private LinkedHashSet<String> propagateAlongAttributes = new LinkedHashSet<>();
    private final ConfigurationFragmentPolicy.Builder configurationFragmentPolicy =
        new ConfigurationFragmentPolicy.Builder();

    public Builder(AspectClass aspectClass) {
      this.aspectClass = aspectClass;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the providers
     * from at least one set of required providers.
     */
    public Builder requireProviderSets(Iterable<ImmutableSet<Class<?>>> providerSets) {
      for (ImmutableSet<Class<?>> providerSet : providerSets) {
        requiredProviders.addNativeSet(providerSet);
      }
      return this;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the specified
     * providers.
     */
    public Builder requireProviders(Class<?>... providers) {
      requiredProviders.addNativeSet(ImmutableSet.copyOf(providers));
      return this;
    }

    public Builder requireAspectsWithProviders(
        Iterable<ImmutableSet<SkylarkProviderIdentifier>> providerSets) {
      for (ImmutableSet<SkylarkProviderIdentifier> providerSet : providerSets) {
        if (!providerSet.isEmpty()) {
          requiredAspectProviders.addSkylarkSet(providerSet);
        }
      }
      return this;
    }

    public Builder requireAspectsWithNativeProviders(
        Iterable<ImmutableSet<SkylarkProviderIdentifier>> providerSets) {
      for (ImmutableSet<SkylarkProviderIdentifier> providerSet : providerSets) {
        requiredAspectProviders.addSkylarkSet(providerSet);
      }
      return this;
    }

    /**
     * State that the aspect being built provides given providers.
     */
    public Builder advertiseProvider(Class<?>... providers) {
      for (Class<?> provider : providers) {
        advertisedProviders.addNative(provider);
      }
      return this;
    }

    /**
     * State that the aspect being built provides given providers.
     */
    public Builder advertiseProvider(ImmutableList<SkylarkProviderIdentifier> providers) {
      for (SkylarkProviderIdentifier provider : providers) {
        // todo(dslomov,vladmos): support declared providers
        Preconditions.checkState(provider.isLegacy());
        advertisedProviders.addSkylark(provider.getLegacyId());
      }
      return this;
    }



    /**
     * Declares that this aspect propagates along an {@code attribute} on the target
     * associated with this aspect.
     *
     * Specify multiple attributes by calling {@link #propagateAlongAttribute(String)}
     * repeatedly.
     *
     * Aspect can also declare to propagate along all attributes with
     * {@link #propagateAlongAttributes}.
     */
    public final Builder propagateAlongAttribute(String attribute) {
      Preconditions.checkNotNull(attribute);
      Preconditions.checkState(this.propagateAlongAttributes != null,
          "Either propagate along all attributes, or along specific attributes, not both");

      this.propagateAlongAttributes.add(attribute);

      return this;
    }

    /**
     * Declares that this aspect propagates along all attributes on the target
     * associated with this aspect.
     *
     * Specify either this or {@link #propagateAlongAttribute(String)}, not both.
     */
    public final Builder propagateAlongAllAttributes() {
      Preconditions.checkState(this.propagateAlongAttributes != null,
          "Aspects for all attributes must only be specified once");

      Preconditions.checkState(this.propagateAlongAttributes.isEmpty(),
          "Specify either aspects for all attributes, or for specific attributes, not both");
      this.propagateAlongAttributes = null;
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
      Preconditions.checkArgument(attribute.isImplicit() || attribute.isLateBound()
          || (attribute.getType() == Type.STRING && attribute.checkAllowedValues()),
          "Invalid attribute '%s' (%s)", attribute.getName(), attribute.getType());
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
      configurationFragmentPolicy
          .requiresHostConfigurationFragmentsBySkylarkModuleName(configurationFragmentNames);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing required fragments (see
     * {@link #requiresConfigurationFragments}).
     */
    public Builder setMissingFragmentPolicy(MissingFragmentPolicy missingFragmentPolicy) {
      configurationFragmentPolicy.setMissingFragmentPolicy(missingFragmentPolicy);
      return this;
    }


    /**
     * Builds the aspect definition.
     *
     * <p>The builder object is reusable afterwards.
     */
    public AspectDefinition build() {
      return new AspectDefinition(aspectClass,
          advertisedProviders.build(),
          requiredProviders.build(),
          requiredAspectProviders.build(),
          ImmutableMap.copyOf(attributes),
          propagateAlongAttributes == null
              ? null
              : ImmutableSet.copyOf(propagateAlongAttributes),
          configurationFragmentPolicy.build());
    }
  }
}
