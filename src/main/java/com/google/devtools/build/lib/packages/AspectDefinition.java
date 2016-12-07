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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Collection;
import java.util.LinkedHashMap;
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

  private final AspectClass aspectClass;
  private final ImmutableList<ImmutableSet<Class<?>>> requiredProviderSets;
  private final ImmutableList<ImmutableSet<String>> requiredProviderNameSets;
  private final ImmutableMap<String, Attribute> attributes;
  private final PropagationFunction attributeAspects;
  @Nullable private final ConfigurationFragmentPolicy configurationFragmentPolicy;

  private interface PropagationFunction {
    ImmutableCollection<AspectClass> propagate(Attribute attribute);
  }

  private AspectDefinition(
      AspectClass aspectClass,
      ImmutableList<ImmutableSet<Class<?>>> requiredProviderSets,
      ImmutableMap<String, Attribute> attributes,
      PropagationFunction attributeAspects,
      @Nullable ConfigurationFragmentPolicy configurationFragmentPolicy) {
    this.aspectClass = aspectClass;
    this.requiredProviderSets = requiredProviderSets;

    this.attributes = attributes;
    this.attributeAspects = attributeAspects;
    this.configurationFragmentPolicy = configurationFragmentPolicy;

    ImmutableList.Builder<ImmutableSet<String>> requiredProviderNameSetsBuilder =
        new ImmutableList.Builder<>();
    for (ImmutableSet<Class<?>> requiredProviderSet : requiredProviderSets) {
      requiredProviderNameSetsBuilder.add(toStringSet(requiredProviderSet));
    }
    this.requiredProviderNameSets = requiredProviderNameSetsBuilder.build();
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
   * Returns the list of {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider}
   * sets. All required providers from at least one set must be present on a configured target so
   * that this aspect can be applied to it.
   *
   * <p>We cannot refer to that class here due to our dependency structure, so this returns a set
   * of unconstrained class objects.
   *
   * <p>If a configured target does not have a required provider, the aspect is silently not created
   * for it.
   */
  public ImmutableList<ImmutableSet<Class<?>>> getRequiredProviders() {
    return requiredProviderSets;
  }

  /**
   * Returns the list of class name sets of
   * {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider}. All required providers
   * from at least one set must be present on a configured target so that this aspect can be applied
   * to it.
   *
   * <p>This set is a mirror of the set returned by {@link #getRequiredProviders}, but contains the
   * names of the classes rather than the class objects themselves.
   *
   * <p>If a configured target does not have a required provider, the aspect is silently not created
   * for it.
   */
  public ImmutableList<ImmutableSet<String>> getRequiredProviderNames() {
    return requiredProviderNameSets;
  }

  /**
   * Returns the set of required aspects for a given atribute.
   */
  public ImmutableCollection<AspectClass> getAttributeAspects(Attribute attribute) {
    return attributeAspects.propagate(attribute);
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
    ImmutableSet<Class<?>> providers = ruleClass.getAdvertisedProviders();
    return visitAspectsIfRequired((Rule) from, attribute, ruleClass.canHaveAnyProvider(),
        toStringSet(providers), dependencyFilter);
  }

  /**
   * Returns the attribute -&gt; set of labels that are provided by aspects of attribute.
   */
  public static ImmutableMultimap<Attribute, Label> visitAspectsIfRequired(
      Rule from, Attribute attribute, boolean canHaveAnyProvider, Set<String> advertisedProviders,
      DependencyFilter dependencyFilter) {
    SetMultimap<Attribute, Label> result = LinkedHashMultimap.create();
    for (Aspect candidateClass : attribute.getAspects(from)) {
      // Check if target satisfies condition for this aspect (has to provide all required
      // TransitiveInfoProviders)
      if (!canHaveAnyProvider) {
        ImmutableList<ImmutableSet<String>> providerNamesList =
            candidateClass.getDefinition().getRequiredProviderNames();

        for (ImmutableSet<String> providerNames : providerNamesList) {
          if (advertisedProviders.containsAll(providerNames)) {
            addAllAttributesOfAspect(from, result, candidateClass, dependencyFilter);
            break;
          }
        }
      } else {
        addAllAttributesOfAspect(from, result, candidateClass, dependencyFilter);
      }
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

  @Nullable
  private static Label maybeGetRepositoryRelativeLabel(Rule from, @Nullable Label label) {
    return label == null ? null : from.getLabel().resolveRepositoryRelative(label);
  }

  /**
   * Collects all attribute labels from the specified aspectDefinition.
   */
  public static void addAllAttributesOfAspect(
      Rule from,
      Multimap<Attribute, Label> labelBuilder,
      Aspect aspect,
      DependencyFilter dependencyFilter) {
    ImmutableMap<String, Attribute> attributes = aspect.getDefinition().getAttributes();
    for (Attribute aspectAttribute : attributes.values()) {
      if (!dependencyFilter.apply(aspect, aspectAttribute)) {
        continue;
      }
      if (aspectAttribute.getType() == BuildType.LABEL) {
        Label label = maybeGetRepositoryRelativeLabel(
            from, BuildType.LABEL.cast(aspectAttribute.getDefaultValue(from)));
        if (label != null) {
          labelBuilder.put(aspectAttribute, label);
        }
      } else if (aspectAttribute.getType() == BuildType.LABEL_LIST) {
        List<Label> defaultLabels = BuildType.LABEL_LIST.cast(
            aspectAttribute.getDefaultValue(from));
        if (defaultLabels != null) {
          for (Label defaultLabel : defaultLabels) {
            Label label = maybeGetRepositoryRelativeLabel(from, defaultLabel);
            if (label != null) {
              labelBuilder.put(aspectAttribute, label);
            }
          }
        }
      }
    }
  }

  /**
   * Builder class for {@link AspectDefinition}.
   */
  public static final class Builder {
    private final AspectClass aspectClass;
    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private ImmutableList<ImmutableSet<Class<?>>> requiredProviderSets = ImmutableList.of();
    private final Multimap<String, AspectClass> attributeAspects = LinkedHashMultimap.create();
    private ImmutableCollection<AspectClass> allAttributesAspects = null;
    private final ConfigurationFragmentPolicy.Builder configurationFragmentPolicy =
        new ConfigurationFragmentPolicy.Builder();

    public Builder(AspectClass aspectClass) {
      this.aspectClass = aspectClass;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the providers
     * from at least one set of required providers.
     */
    public Builder requireProviderSets(Iterable<? extends Set<Class<?>>> providerSets) {
      ImmutableList.Builder<ImmutableSet<Class<?>>> requiredProviderSetsBuilder =
          ImmutableList.builder();
      for (Iterable<Class<?>> providerSet : providerSets) {
        requiredProviderSetsBuilder.add(ImmutableSet.copyOf(providerSet));
      }
      requiredProviderSets = requiredProviderSetsBuilder.build();
      return this;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the specified
     * providers.
     */
    public Builder requireProviders(Class<?>... requiredProviders) {
      requireProviderSets(ImmutableList.of(ImmutableSet.copyOf(requiredProviders)));
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
    public final Builder attributeAspect(String attribute, NativeAspectClass... aspectClasses) {
      Preconditions.checkNotNull(attribute);
      for (NativeAspectClass aspectClass : aspectClasses) {
        this.attributeAspect(attribute, Preconditions.checkNotNull(aspectClass));
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
      Preconditions.checkState(this.allAttributesAspects == null,
          "Specify either aspects for all attributes, or for specific attributes, not both");

      this.attributeAspects.put(attribute, Preconditions.checkNotNull(aspectClass));

      return this;
    }

    public final Builder allAttributesAspect(AspectClass... aspectClasses) {
      Preconditions.checkState(this.attributeAspects.isEmpty(),
          "Specify either aspects for all attributes, or for specific attributes, not both");
      Preconditions.checkState(this.allAttributesAspects == null,
          "Aspects for all attributes must only be specified once");
      this.allAttributesAspects = ImmutableList.copyOf(aspectClasses);
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

    @Immutable
    private static final class AllAttributesPropagationFunction implements PropagationFunction {
      private final ImmutableCollection<AspectClass> aspects;

      private AllAttributesPropagationFunction(ImmutableCollection<AspectClass> aspects) {
        this.aspects = aspects;
      }

      @Override
      public ImmutableCollection<AspectClass> propagate(Attribute attribute) {
        return aspects;
      }
    }

    @Immutable
    private static final class PerAttributePropagationFunction implements PropagationFunction {
      ImmutableSetMultimap<String, AspectClass> aspects;

      public PerAttributePropagationFunction(
          ImmutableSetMultimap<String, AspectClass> aspects) {
        this.aspects = aspects;
      }

      @Override
      public ImmutableCollection<AspectClass> propagate(Attribute attribute) {
        return aspects.get(attribute.getName());
      }
    }

    /**
     * Builds the aspect definition.
     *
     * <p>The builder object is reusable afterwards.
     */
    public AspectDefinition build() {
      // If there is no required provider set, we still need to at least provide one empty set of
      // providers. We consider this case specially because aspects with no required providers
      // should match all rules, and having an empty set faciliates the matching logic.
      ImmutableList<ImmutableSet<Class<?>>> requiredProviders =
          requiredProviderSets.isEmpty()
          ? ImmutableList.of(ImmutableSet.<Class<?>>of())
          : requiredProviderSets;

      return new AspectDefinition(aspectClass, ImmutableList.copyOf(requiredProviders),
          ImmutableMap.copyOf(attributes),
          allAttributesAspects != null
              ? new AllAttributesPropagationFunction(allAttributesAspects)
              : new PerAttributePropagationFunction(ImmutableSetMultimap.copyOf(attributeAspects)),
          configurationFragmentPolicy.build());
    }
  }
}
