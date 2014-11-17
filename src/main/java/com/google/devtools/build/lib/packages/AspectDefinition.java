// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * The definition of an aspect (see {@link com.google.devtools.build.lib.view.Aspect} for more
 * information.)
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
  /**
   * Information about a direct dependency used to determine whether an aspect is applicable.
   */
  @Immutable
   public static final class CandidateDependency {
    private final Rule rule;
    private final ImmutableSet<Class<?>> providers;

    private CandidateDependency(Rule rule, ImmutableSet<Class<?>> providers) {
      this.rule = rule;
      this.providers = providers;
    }

    /**
     * Returns the rule the aspect is associated with.
     */
    public Rule getRule() {
      return rule;
    }

    /**
     * Returns the transitive info providers of the direct dependency.
     *
     * <p>Note that the set will contain
     * {@link com.google.devtools.build.lib.view.TransitiveInfoProvider} subclasses, but we cannot
     * reference that class here.
     */
    public ImmutableSet<Class<?>> getProviders() {
      return providers;
    }
  }

  /**
   * Returns a predicate that is true if the direct dependency contains every specified provider.
   *
   * @param requiredProviders the providers that are required. Should contain class objects of
   *     subclasses of {@link com.google.devtools.build.lib.view.TransitiveInfoProvider} (we cannot
   *     reference that class here)
   */
  public static Predicate<CandidateDependency> requiresProviders(
      final Class<?>... requiredProviders) {
    return new Predicate<CandidateDependency>() {
      @Override
      public boolean apply(CandidateDependency input) {
        for (Class<?> required : requiredProviders) {
          if (!input.getProviders().contains(required)) {
            return false;
          }
        }

        return true;
      }
    };
  }

  private final Predicate<CandidateDependency> condition;
  private final ImmutableMap<String, Attribute> attributes;
  private final ImmutableMultimap<String, Class<?>> attributeAspects;

  private AspectDefinition(
      Predicate<CandidateDependency> condition,
      ImmutableMap<String, Attribute> attributes,
      ImmutableMultimap<String, Class<?>> attributeAspects) {
    this.condition = condition;
    this.attributes = attributes;
    this.attributeAspects = attributeAspects;
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
   * Returns the attribute -&gt; set of required aspects map.
   *
   * <p>Note that the map actually contains {@link AspectFactory}
   * instances, except that we cannot reference that class here.
   */
  public ImmutableMultimap<String, Class<?>> getAttributeAspects() {
    return attributeAspects;
  }

  /**
   * Returns whether this aspect applies to a rule and a direct dependency with the specified set
   * of providers.
   */
  public boolean appliesTo(Rule rule, Set<Class<?>> providers) {
    return condition.apply(new CandidateDependency(rule, ImmutableSet.copyOf(providers)));
  }

  /**
   * Builder class for {@link AspectDefinition}.
   */
  public static final class Builder {
    private Predicate<CandidateDependency> condition = Predicates.alwaysFalse();
    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private Multimap<String, Class<?>> attributeAspects =
        LinkedHashMultimap.create();

    public Builder() {
    }

    /**
     * Sets the condition which determines whether this aspect can operate on a direct dependency.
     */
    public Builder condition(Predicate<CandidateDependency> condition) {
      this.condition = Preconditions.checkNotNull(condition);
      return this;
    }

    /**
     * Tells that in order for this aspect to work, the given aspect must be computed for the
     * direct dependencies in the attribute with the specified name on the associated configured
     * target.
     *
     * <p>Note that {@code AspectFactory} instances are expected in the second argument, but we
     * cannot reference that interface here.
     */
    public Builder attributeAspect(String attribute, Class<?> aspectFactory) {
      this.attributeAspects.put(
          Preconditions.checkNotNull(attribute), Preconditions.checkNotNull(aspectFactory));
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
      Preconditions.checkState(attribute.isImplicit() || attribute.isLateBound());
      Preconditions.checkState(!attributes.containsKey(attribute.getName()),
          "An attribute with the name '%s' already exists.", attribute.getName());
      attributes.put(attribute.getName(), attribute);
      return this;
    }

    /**
     * Builds the aspect definition.
     *
     * <p>The builder object is reusable afterwards.
     */
    public AspectDefinition build() {
      return new AspectDefinition(condition, ImmutableMap.copyOf(attributes),
          ImmutableMultimap.copyOf(attributeAspects));
    }
  }
}
