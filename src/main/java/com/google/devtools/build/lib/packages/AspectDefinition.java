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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.Type.LabelVisitor;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;

/**
 * The definition of an aspect (see {@link Aspect} for more information).
 *
 * <p>Contains enough information to build up the configured target graph except for the actual way
 * to build the Skyframe node (that is the territory of {@link com.google.devtools.build.lib.view
 * AspectFactory}). In particular:
 *
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
  private final ImmutableSet<ToolchainTypeRequirement> toolchainTypes;

  /**
   * Which attributes aspect should propagate along:
   *
   * <ul>
   *   <li>A {@code null} value means propagate along all attributes
   *   <li>A (possibly empty) set means to propagate only along the attributes in a set
   * </ul>
   */
  @Nullable private final ImmutableSet<String> restrictToAttributes;

  @Nullable private final ConfigurationFragmentPolicy configurationFragmentPolicy;
  private final boolean applyToFiles;
  private final boolean applyToGeneratingRules;

  private final ImmutableSet<AspectClass> requiredAspectClasses;

  private final ImmutableSet<Label> execCompatibleWith;
  private final ImmutableMap<String, ExecGroup> execGroups;

  public AdvertisedProviderSet getAdvertisedProviders() {
    return advertisedProviders;
  }

  private AspectDefinition(
      AspectClass aspectClass,
      AdvertisedProviderSet advertisedProviders,
      RequiredProviders requiredProviders,
      RequiredProviders requiredProvidersForAspects,
      ImmutableMap<String, Attribute> attributes,
      ImmutableSet<ToolchainTypeRequirement> toolchainTypes,
      @Nullable ImmutableSet<String> restrictToAttributes,
      @Nullable ConfigurationFragmentPolicy configurationFragmentPolicy,
      boolean applyToFiles,
      boolean applyToGeneratingRules,
      ImmutableSet<AspectClass> requiredAspectClasses,
      ImmutableSet<Label> execCompatibleWith,
      ImmutableMap<String, ExecGroup> execGroups) {
    this.aspectClass = aspectClass;
    this.advertisedProviders = advertisedProviders;
    this.requiredProviders = requiredProviders;
    this.requiredProvidersForAspects = requiredProvidersForAspects;
    this.attributes = attributes;
    this.toolchainTypes = toolchainTypes;
    this.restrictToAttributes = restrictToAttributes;
    this.configurationFragmentPolicy = configurationFragmentPolicy;
    this.applyToFiles = applyToFiles;
    this.applyToGeneratingRules = applyToGeneratingRules;
    this.requiredAspectClasses = requiredAspectClasses;
    this.execCompatibleWith = execCompatibleWith;
    this.execGroups = execGroups;
  }

  public String getName() {
    return aspectClass.getName();
  }

  /**
   * Returns the attributes of the aspect in the form of a String -&gt; {@link Attribute} map.
   *
   * <p>All attributes are either implicit or late-bound.
   */
  public ImmutableMap<String, Attribute> getAttributes() {
    return attributes;
  }

  /** Returns the required toolchains declared by this aspect. */
  public ImmutableSet<ToolchainTypeRequirement> getToolchainTypes() {
    return toolchainTypes;
  }

  /**
   * Returns the constraint values that must be present on an execution platform for this aspect.
   */
  public ImmutableSet<Label> execCompatibleWith() {
    return execCompatibleWith;
  }

  /** Returns the execution groups that this aspect can use when creating actions. */
  public ImmutableMap<String, ExecGroup> execGroups() {
    return execGroups;
  }

  /**
   * Returns {@link RequiredProviders} that a configured target must have so that this aspect can be
   * applied to it.
   *
   * <p>If a configured target does not satisfy required providers, the aspect is silently not
   * created for it.
   */
  public RequiredProviders getRequiredProviders() {
    return requiredProviders;
  }

  /**
   * Aspects do not depend on other aspects applied to the same target <em>unless</em> the other
   * aspect satisfies the {@link RequiredProviders} this method returns
   */
  public RequiredProviders getRequiredProvidersForAspects() {
    return requiredProvidersForAspects;
  }

  /** Returns whether the aspect propagates along the give {@code attributeName} or not. */
  public boolean propagateAlong(String attributeName) {
    if (restrictToAttributes != null) {
      return restrictToAttributes.contains(attributeName);
    }
    return true;
  }

  /** Returns the set of configuration fragments required by this Aspect. */
  public ConfigurationFragmentPolicy getConfigurationFragmentPolicy() {
    return configurationFragmentPolicy;
  }

  /**
   * Returns whether this aspect applies to (output) files.
   *
   * <p>Currently only supported for top-level aspects and targets, and only for output files.
   */
  public boolean applyToFiles() {
    return applyToFiles;
  }

  /**
   * Returns whether this aspect should, when it would be applied to an output file, instead apply
   * to the generating rule of that output file.
   */
  public boolean applyToGeneratingRules() {
    return applyToGeneratingRules;
  }

  public static boolean satisfies(Aspect aspect, AdvertisedProviderSet advertisedProviderSet) {
    return aspect.getDefinition().requiredProviders.isSatisfiedBy(advertisedProviderSet);
  }

  /** Checks if the given {@code maybeRequiredAspect} is required by this aspect definition */
  public boolean requires(Aspect maybeRequiredAspect) {
    return requiredAspectClasses.contains(maybeRequiredAspect.getAspectClass());
  }

  /** Collects all attribute labels from the specified aspectDefinition. */
  public static void addAllAttributesOfAspect(
      Multimap<Attribute, Label> labelBuilder, Aspect aspect, DependencyFilter dependencyFilter) {
    forEachLabelDepFromAllAttributesOfAspect(aspect, dependencyFilter, labelBuilder::put);
  }

  public static void forEachLabelDepFromAllAttributesOfAspect(
      Aspect aspect,
      DependencyFilter dependencyFilter,
      BiConsumer<Attribute, Label> consumer) {
    LabelVisitor labelVisitor =
        (label, aspectAttribute) -> {
          if (label == null) {
            return;
          }
          consumer.accept(aspectAttribute, label);
        };
    for (Attribute aspectAttribute : aspect.getDefinition().attributes.values()) {
      if (!dependencyFilter.test(aspect, aspectAttribute)) {
        continue;
      }
      Type<?> type = aspectAttribute.getType();
      if (type.getLabelClass() != LabelClass.DEPENDENCY) {
        continue;
      }
      visitSingleAttribute(aspectAttribute, aspectAttribute.getType(), labelVisitor);
    }
  }

  private static <T> void visitSingleAttribute(
      Attribute attribute, Type<T> type, LabelVisitor labelVisitor) {
    type.visitLabels(labelVisitor, type.cast(attribute.getDefaultValue(null)), attribute);
  }

  public static Builder builder(AspectClass aspectClass) {
    return new Builder(aspectClass);
  }

  /** Builder class for {@link AspectDefinition}. */
  public static final class Builder {
    private final AspectClass aspectClass;
    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private final AdvertisedProviderSet.Builder advertisedProviders =
        AdvertisedProviderSet.builder();
    private final RequiredProviders.Builder requiredProviders =
        RequiredProviders.acceptAnyBuilder();
    private final RequiredProviders.Builder requiredAspectProviders =
        RequiredProviders.acceptNoneBuilder();
    @Nullable private LinkedHashSet<String> propagateAlongAttributes = new LinkedHashSet<>();
    private final ConfigurationFragmentPolicy.Builder configurationFragmentPolicy =
        new ConfigurationFragmentPolicy.Builder();
    private boolean applyToFiles = false;
    private boolean applyToGeneratingRules = false;
    private final Set<ToolchainTypeRequirement> toolchainTypes = new HashSet<>();
    private ImmutableSet<AspectClass> requiredAspectClasses = ImmutableSet.of();
    private ImmutableSet<Label> execCompatibleWith = ImmutableSet.of();
    private ImmutableMap<String, ExecGroup> execGroups = ImmutableMap.of();

    public Builder(AspectClass aspectClass) {
      this.aspectClass = aspectClass;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the providers
     * from at least one set of required providers.
     */
    @CanIgnoreReturnValue
    public Builder requireProviderSets(
        Iterable<ImmutableSet<Class<? extends TransitiveInfoProvider>>> providerSets) {
      for (ImmutableSet<Class<? extends TransitiveInfoProvider>> providerSet : providerSets) {
        requiredProviders.addBuiltinSet(providerSet);
      }
      return this;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the specified
     * providers.
     */
    @CanIgnoreReturnValue
    public Builder requireProviders(Class<? extends TransitiveInfoProvider>... providers) {
      requiredProviders.addBuiltinSet(ImmutableSet.copyOf(providers));
      return this;
    }

    /**
     * Same as the equivalent calls to {@link #requireProviderSets} and {@link
     * #requireStarlarkProviderSets} for specifying the required providers conveyed by {@code
     * requiredProviders}.
     */
    @CanIgnoreReturnValue
    public Builder requireProviders(RequiredProviders requiredProviders) {
      requiredProviders.addToAspectDefinitionBuilder(this);
      return this;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the providers
     * from at least one set of required providers.
     */
    @CanIgnoreReturnValue
    public Builder requireStarlarkProviderSets(
        Iterable<ImmutableSet<StarlarkProviderIdentifier>> providerSets) {
      for (ImmutableSet<StarlarkProviderIdentifier> providerSet : providerSets) {
        if (!providerSet.isEmpty()) {
          requiredProviders.addStarlarkSet(providerSet);
        }
      }
      return this;
    }

    /**
     * Asserts that this aspect can only be evaluated for rules that supply all of the specified
     * Starlark providers.
     */
    @CanIgnoreReturnValue
    public Builder requireStarlarkProviders(StarlarkProviderIdentifier... starlarkProviders) {
      requiredProviders.addStarlarkSet(ImmutableSet.copyOf(starlarkProviders));
      return this;
    }

    /**
     * Asserts that this aspect requires a list of aspects to be applied before it on the configured
     * target.
     */
    @CanIgnoreReturnValue
    public Builder requiredAspectClasses(ImmutableSet<AspectClass> requiredAspectClasses) {
      this.requiredAspectClasses = requiredAspectClasses;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder requireAspectsWithProviders(
        Iterable<ImmutableSet<StarlarkProviderIdentifier>> providerSets) {
      for (ImmutableSet<StarlarkProviderIdentifier> providerSet : providerSets) {
        if (!providerSet.isEmpty()) {
          requiredAspectProviders.addStarlarkSet(providerSet);
        }
      }
      return this;
    }

    @CanIgnoreReturnValue
    public Builder requireAspectsWithBuiltinProviders(
        Class<? extends TransitiveInfoProvider>... providers) {
      requiredAspectProviders.addBuiltinSet(ImmutableSet.copyOf(providers));
      return this;
    }

    /** State that the aspect being built provides given providers. */
    @CanIgnoreReturnValue
    public Builder advertiseProvider(Class<?>... providers) {
      for (Class<?> provider : providers) {
        advertisedProviders.addBuiltin(provider);
      }
      return this;
    }

    /** State that the aspect being built provides given providers. */
    @CanIgnoreReturnValue
    public Builder advertiseProvider(ImmutableList<StarlarkProviderIdentifier> providers) {
      for (StarlarkProviderIdentifier provider : providers) {
        advertisedProviders.addStarlark(provider);
      }
      return this;
    }

    /**
     * Declares that this aspect propagates along an {@code attribute} on the target associated with
     * this aspect.
     *
     * <p>Specify multiple attributes by calling this method repeatedly.
     *
     * <p>Aspect can also declare to propagate along all attributes with {@link
     * #propagateAlongAttributes}.
     */
    @CanIgnoreReturnValue
    public Builder propagateAlongAttribute(String attribute) {
      Preconditions.checkNotNull(attribute);
      Preconditions.checkState(
          this.propagateAlongAttributes != null,
          "Either propagate along all attributes, or along specific attributes, not both");

      this.propagateAlongAttributes.add(attribute);

      return this;
    }

    /**
     * Declares that this aspect propagates along all attributes on the target associated with this
     * aspect.
     *
     * <p>Specify either this or {@link #propagateAlongAttribute(String)}, not both.
     */
    @CanIgnoreReturnValue
    public Builder propagateAlongAllAttributes() {
      Preconditions.checkState(
          this.propagateAlongAttributes != null,
          "Aspects for all attributes must only be specified once");

      Preconditions.checkState(
          this.propagateAlongAttributes.isEmpty(),
          "Specify either aspects for all attributes, or for specific attributes, not both");
      this.propagateAlongAttributes = null;
      return this;
    }

    /**
     * Adds an attribute to the aspect.
     */
    public <TYPE> Builder add(Attribute.Builder<TYPE> attr) {
      Attribute attribute = attr.build();
      return add(attribute);
    }

    /**
     * Adds an attribute to the aspect.
     *
     * <p>Aspects attributes can be of any data type if they are not public, i.e. implicit (starting
     * with '$') or late-bound (starting with ':'). While public attributes can only be of types
     * string, integer or boolean.
     *
     * <p>Aspect definition currently cannot handle {@link ComputedDefault} dependencies (type LABEL
     * or LABEL_LIST), because all the dependencies are resolved from the aspect definition and the
     * defining rule.
     */
    @CanIgnoreReturnValue
    public Builder add(Attribute attribute) {
      Preconditions.checkArgument(
          attribute.isImplicit()
              || attribute.isLateBound()
              || (attribute.getType() == Type.STRING && attribute.checkAllowedValues())
              || (attribute.getType() == Type.INTEGER && attribute.checkAllowedValues())
              || attribute.getType() == Type.BOOLEAN,
          "%s: Invalid attribute '%s' (%s)",
          aspectClass.getName(),
          attribute.getName(),
          attribute.getType());

      // Attributes specifying dependencies using ComputedDefault value are currently not supported.
      // The limitation is in place because:
      //  - blaze query requires that all possible values are knowable without BuildConguration
      //  - aspects can attach to any rule
      // Current logic in #forEachLabelDepFromAllAttributesOfAspect is not enough,
      // however {Conservative,Precise}AspectResolver can probably be improved to make that work.
      Preconditions.checkArgument(
          !(attribute.getType().getLabelClass() == LabelClass.DEPENDENCY
              && (attribute.getDefaultValueUnchecked() instanceof ComputedDefault)),
          "%s: Invalid attribute '%s' (%s) with computed default dependencies",
          aspectClass.getName(),
          attribute.getName(),
          attribute.getType());
      Preconditions.checkArgument(
          !attributes.containsKey(attribute.getName()),
          "%s: An attribute with the name '%s' already exists.",
          aspectClass.getName(),
          attribute.getName());
      attributes.put(attribute.getName(), attribute);
      return this;
    }

    /**
     * Declares that the implementation of the associated aspect definition requires the given
     * fragments to be present in this rule's exec and target configurations.
     *
     * <p>The value is inherited by subclasses.
     */
    @CanIgnoreReturnValue
    public Builder requiresConfigurationFragments(
        Class<? extends Fragment>... configurationFragments) {
      configurationFragmentPolicy.requiresConfigurationFragments(
          ImmutableSet.copyOf(configurationFragments));
      return this;
    }

    /**
     * Declares the configuration fragments that are required by this rule for the target
     * configuration.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(Class...)}, this method takes the
     * Starlark module names of fragments instead of their classes.
     */
    @CanIgnoreReturnValue
    public Builder requiresConfigurationFragmentsByStarlarkBuiltinName(
        Collection<String> configurationFragmentNames) {
      configurationFragmentPolicy.requiresConfigurationFragmentsByStarlarkBuiltinName(
          configurationFragmentNames);
      return this;
    }

    /**
     * Sets the policy for the case where the configuration is missing the required fragment class
     * (see {@link #requiresConfigurationFragments}).
     */
    @CanIgnoreReturnValue
    public Builder setMissingFragmentPolicy(
        Class<?> fragmentClass, MissingFragmentPolicy missingFragmentPolicy) {
      configurationFragmentPolicy.setMissingFragmentPolicy(fragmentClass, missingFragmentPolicy);
      return this;
    }

    /**
     * Sets whether this aspect should apply to files.
     *
     * <p>Default is <code>false</code>. Currently only supported for top-level aspects and targets,
     * and only for output files.
     */
    @CanIgnoreReturnValue
    public Builder applyToFiles(boolean propagateOverGeneratedFiles) {
      this.applyToFiles = propagateOverGeneratedFiles;
      return this;
    }

    /**
     * Sets whether this aspect should, when it would be applied to an output file, instead apply to
     * the generating rule of that output file.
     *
     * <p>Default is <code>false</code>. Currently only supported for aspects which do not have a
     * "required providers" list.
     */
    @CanIgnoreReturnValue
    public Builder applyToGeneratingRules(boolean applyToGeneratingRules) {
      this.applyToGeneratingRules = applyToGeneratingRules;
      return this;
    }

    /** Adds the given toolchains as requirements for this aspect. */
    public Builder addToolchainTypes(ToolchainTypeRequirement... toolchainTypes) {
      return this.addToolchainTypes(ImmutableSet.copyOf(toolchainTypes));
    }

    /** Adds the given toolchains as requirements for this aspect. */
    @CanIgnoreReturnValue
    public Builder addToolchainTypes(Collection<ToolchainTypeRequirement> toolchainTypes) {
      this.toolchainTypes.addAll(toolchainTypes);
      return this;
    }

    /**
     * Adds the given constraint values to the set required for execution platforms for this aspect.
     */
    @CanIgnoreReturnValue
    public Builder execCompatibleWith(ImmutableSet<Label> execCompatibleWith) {
      this.execCompatibleWith = execCompatibleWith;
      return this;
    }

    /** Sets the execution groups that are available for actions created by this aspect. */
    @CanIgnoreReturnValue
    public Builder execGroups(ImmutableMap<String, ExecGroup> execGroups) {
      // TODO(b/230337573): validate names
      // TODO(b/230337573): handle copy_from_default
      this.execGroups = execGroups;
      return this;
    }

    /**
     * Builds the aspect definition.
     *
     * <p>The builder object is reusable afterwards.
     */
    public AspectDefinition build() {
      RequiredProviders requiredProviders = this.requiredProviders.build();
      if (applyToGeneratingRules && !requiredProviders.acceptsAny()) {
        throw new IllegalStateException(
            "An aspect cannot simultaneously have required providers "
                + "and apply to generating rules.");
      }

      return new AspectDefinition(
          aspectClass,
          advertisedProviders.build(),
          requiredProviders,
          requiredAspectProviders.build(),
          ImmutableMap.copyOf(attributes),
          ImmutableSet.copyOf(toolchainTypes),
          propagateAlongAttributes == null ? null : ImmutableSet.copyOf(propagateAlongAttributes),
          configurationFragmentPolicy.build(),
          applyToFiles,
          applyToGeneratingRules,
          requiredAspectClasses,
          execCompatibleWith,
          execGroups);
    }
  }
}
