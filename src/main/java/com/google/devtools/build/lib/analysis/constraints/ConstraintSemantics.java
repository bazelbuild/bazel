// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.constraints.EnvironmentCollection.EnvironmentWithGroup;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Implementation of the semantics of Bazel's constraint specification and enforcement system.
 *
 * <p>This is how the system works:
 *
 * <p>All build rules can declare which "environments" they can be built for, where an "environment"
 * is a label instance of an {@link EnvironmentRule} rule declared in a BUILD file. There are
 * various ways to do this:
 *
 * <ul>
 *   <li>Through a "restricted to" attribute setting
 *   ({@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR}). This is the most direct form of
 *   specification - it declares the exact set of environments the rule supports (for its group -
 *   see precise details below).
 *   <li>Through a "compatible with" attribute setting
 *   ({@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR}. This declares <b>additional</b>
 *   environments a rule supports in addition to "standard" environments that are supported by
 *   default (see below).
 *   <li>Through "default" specifications in {@link EnvironmentGroup} rules. Every environment
 *   belongs to a group of thematically related peers (e.g. "target architectures", "JDK versions",
 *   or "mobile devices"). An environment group's definition includes which of these
 *   environments should be supported "by default" if not otherwise specified by one of the above
 *   mechanisms. In particular, a rule with no environment-related attributes automatically
 *   inherits all defaults.
 *   <li>Through a rule class default ({@link RuleClass.Builder#restrictedTo} and
 *   {@link RuleClass.Builder#compatibleWith}). This overrides global defaults for all instances
 *   of the given rule class. This can be used, for example, to make all *_test rules "testable"
 *   without each instance having to explicitly declare this capability.
 * </ul>
 *
 * <p>Groups exist to model the idea that some environments are related while others have nothing
 * to do with each other. Say, for example, we want to say a rule works for PowerPC platforms but
 * not x86. We can do so by setting its "restricted to" attribute to
 * {@code ['//sample/path:powerpc']}. Because both PowerPC and x86 are in the same
 * "target architectures" group, this setting removes x86 from the set of supported environments.
 * But since JDK support belongs to its own group ("JDK versions") it says nothing about which JDK
 * the rule supports.
 *
 * <p>More precisely, if a rule has a "restricted to" value of [A, B, C], this removes support
 * for all default environments D such that group(D) is in [group(A), group(B), group(C)] AND
 * D is not in [A, B, C] (in other words, D isn't explicitly opted back in). The rule's full
 * set of supported environments thus becomes [A, B, C] + all defaults that belong to unrelated
 * groups.
 *
 * <p>If the rule has a "compatible with" value of [E, F, G], these are unconditionally
 * added to its set of supported environments (in addition to the results from above).
 *
 * <p>An environment may not appear in both a rule's "restricted to" and "compatible with" values.
 * If two environments belong to the same group, they must either both be in "restricted to",
 * both be in "compatible with", or not explicitly specified.
 *
 * <p>Given all the above, constraint enforcement is this: rule A can depend on rule B if, for
 * every environment A supports, B also supports that environment.
 */
public class ConstraintSemantics {
  private ConstraintSemantics() {
  }

  /**
   * Provides a set of default environments for a given environment group.
   */
  private interface DefaultsProvider {
    Collection<Label> getDefaults(EnvironmentGroup group);
  }

  /**
   * Provides a group's defaults as specified in the environment group's BUILD declaration.
   */
  private static class GroupDefaultsProvider implements DefaultsProvider {
    @Override
    public Collection<Label> getDefaults(EnvironmentGroup group) {
      return group.getDefaults();
    }
  }

  /**
   * Provides a group's defaults, factoring in rule class defaults as specified by
   * {@link com.google.devtools.build.lib.packages.RuleClass.Builder#compatibleWith}
   * and {@link com.google.devtools.build.lib.packages.RuleClass.Builder#restrictedTo}.
   */
  private static class RuleClassDefaultsProvider implements DefaultsProvider {
    private final EnvironmentCollection ruleClassDefaults;
    private final GroupDefaultsProvider groupDefaults;

    RuleClassDefaultsProvider(EnvironmentCollection ruleClassDefaults) {
      this.ruleClassDefaults = ruleClassDefaults;
      this.groupDefaults = new GroupDefaultsProvider();
    }

    @Override
    public Collection<Label> getDefaults(EnvironmentGroup group) {
      if (ruleClassDefaults.getGroups().contains(group)) {
        return ruleClassDefaults.getEnvironments(group);
      } else {
        // If there are no rule class defaults for this group, just inherit global defaults.
        return groupDefaults.getDefaults(group);
      }
    }
  }

  /**
   * Collects the set of supported environments for a given rule by merging its
   * restriction-style and compatibility-style environment declarations as specified by
   * the given attributes. Only includes environments from "known" groups, i.e. the groups
   * owning the environments explicitly referenced from these attributes.
   */
  private static class EnvironmentCollector {
    private final RuleContext ruleContext;
    private final String restrictionAttr;
    private final String compatibilityAttr;
    private final DefaultsProvider defaultsProvider;

    private final EnvironmentCollection restrictionEnvironments;
    private final EnvironmentCollection compatibilityEnvironments;
    private final EnvironmentCollection supportedEnvironments;

    /**
     * Constructs a new collector on the given attributes.
     *
     * @param ruleContext analysis context for the rule
     * @param restrictionAttr the name of the attribute that declares "restricted to"-style
     *     environments. If the rule doesn't have this attribute, this is considered an
     *     empty declaration.
     * @param compatibilityAttr the name of the attribute that declares "compatible with"-style
     *     environments. If the rule doesn't have this attribute, this is considered an
     *     empty declaration.
     * @param defaultsProvider provider for the default environments within a group if not
     *     otherwise overriden by the above attributes
     */
    EnvironmentCollector(RuleContext ruleContext, String restrictionAttr, String compatibilityAttr,
        DefaultsProvider defaultsProvider) {
      this.ruleContext = ruleContext;
      this.restrictionAttr = restrictionAttr;
      this.compatibilityAttr = compatibilityAttr;
      this.defaultsProvider = defaultsProvider;

      EnvironmentCollection.Builder environmentsBuilder = new EnvironmentCollection.Builder();
      restrictionEnvironments = collectRestrictionEnvironments(environmentsBuilder);
      compatibilityEnvironments = collectCompatibilityEnvironments(environmentsBuilder);
      supportedEnvironments = environmentsBuilder.build();
    }

    /**
     * Returns the set of environments supported by this rule, as determined by the
     * restriction-style attribute, compatibility-style attribute, and group defaults
     * provider instantiated with this class.
     */
    EnvironmentCollection getEnvironments() {
      return supportedEnvironments;
    }

    /**
     * Validity-checks that no group has its environment referenced in both the "compatible with"
     * and restricted to" attributes. Returns true if all is good, returns false and reports
     * appropriate errors if there are any problems.
     */
    boolean validateEnvironmentSpecifications() {
      ImmutableCollection<EnvironmentGroup> restrictionGroups = restrictionEnvironments.getGroups();
      boolean hasErrors = false;

      for (EnvironmentGroup group : compatibilityEnvironments.getGroups()) {
        if (restrictionGroups.contains(group)) {
          // To avoid error-spamming the user, when we find a conflict we only report one example
          // environment from each attribute for that group.
          Label compatibilityEnv =
              compatibilityEnvironments.getEnvironments(group).iterator().next();
          Label restrictionEnv = restrictionEnvironments.getEnvironments(group).iterator().next();

          if (compatibilityEnv.equals(restrictionEnv)) {
            ruleContext.attributeError(compatibilityAttr, compatibilityEnv
                + " cannot appear both here and in " + restrictionAttr);
          } else {
            ruleContext.attributeError(compatibilityAttr, compatibilityEnv + " and "
                + restrictionEnv + " belong to the same environment group. They should be declared "
                + "together either here or in " + restrictionAttr);
          }
          hasErrors = true;
        }
      }

      return !hasErrors;
    }

    /**
     * Adds environments specified in the "restricted to" attribute to the set of supported
     * environments and returns the environments added.
     */
    private EnvironmentCollection collectRestrictionEnvironments(
        EnvironmentCollection.Builder supportedEnvironments) {
      return collectEnvironments(restrictionAttr, supportedEnvironments);
    }

    /**
     * Adds environments specified in the "compatible with" attribute to the set of supported
     * environments, along with all defaults from the groups they belong to. Returns these
     * environments, not including the defaults.
     */
    private EnvironmentCollection collectCompatibilityEnvironments(
        EnvironmentCollection.Builder supportedEnvironments) {
      EnvironmentCollection compatibilityEnvironments =
          collectEnvironments(compatibilityAttr, supportedEnvironments);
      for (EnvironmentGroup group : compatibilityEnvironments.getGroups()) {
        supportedEnvironments.putAll(group, defaultsProvider.getDefaults(group));
      }
      return compatibilityEnvironments;
    }

    /**
     * Adds environments specified by the given attribute to the set of supported environments
     * and returns the environments added.
     *
     * <p>If this rule doesn't have the given attributes, returns an empty set.
     */
    private EnvironmentCollection collectEnvironments(String attrName,
        EnvironmentCollection.Builder supportedEnvironments) {
      if (!ruleContext.getRule().isAttrDefined(attrName,  BuildType.LABEL_LIST)) {
        return EnvironmentCollection.EMPTY;
      }
      EnvironmentCollection.Builder environments = new EnvironmentCollection.Builder();
      for (TransitiveInfoCollection envTarget :
          ruleContext.getPrerequisites(attrName, RuleConfiguredTarget.Mode.DONT_CHECK)) {
        EnvironmentWithGroup envInfo = resolveEnvironment(envTarget);
        environments.put(envInfo.group(), envInfo.environment());
        supportedEnvironments.put(envInfo.group(), envInfo.environment());
      }
      return environments.build();
    }

    /**
     * Returns the environment and its group. An {@link Environment} rule only "supports" one
     * environment: itself. Extract that from its more generic provider interface and sanity
     * check that that's in fact what we see.
     */
    private static EnvironmentWithGroup resolveEnvironment(TransitiveInfoCollection envRule) {
      SupportedEnvironmentsProvider prereq =
          Preconditions.checkNotNull(envRule.getProvider(SupportedEnvironmentsProvider.class));
      return Iterables.getOnlyElement(prereq.getEnvironments().getGroupedEnvironments());
    }
  }

  /**
   * Exception indicating errors finding/parsing environments or their containing groups.
   */
  public static class EnvironmentLookupException extends Exception {
    private EnvironmentLookupException(String message) {
      super(message);
    }
  }

  /**
   * Returns the environment group that owns the given environment. Both must belong to
   * the same package.
   *
   * @throws EnvironmentLookupException if the input is not an {@link EnvironmentRule} or no
   *     matching group is found
   */
  public static EnvironmentGroup getEnvironmentGroup(Target envTarget)
      throws EnvironmentLookupException {
    if (!(envTarget instanceof Rule)
        || !((Rule) envTarget).getRuleClass().equals(EnvironmentRule.RULE_NAME)) {
      throw new EnvironmentLookupException(
          envTarget.getLabel() + " is not a valid environment definition");
    }
    for (EnvironmentGroup group : envTarget.getPackage().getTargets(EnvironmentGroup.class)) {
      if (group.getEnvironments().contains(envTarget.getLabel())) {
        return group;
      }
    }
    throw new EnvironmentLookupException(
        "cannot find the group for environment " + envTarget.getLabel());
  }

  /**
   * Returns the set of environments this rule supports, applying the logic described in
   * {@link ConstraintSemantics}.
   *
   * <p>Note this set is <b>not complete</b> - it doesn't include environments from groups we don't
   * "know about". Environments and groups can be declared in any package. If the rule includes
   * no references to that package, then it simply doesn't know anything about them. But the
   * constraint semantics say the rule should support the defaults for that group. We encode this
   * implicitly: given the returned set, for any group that's not in the set the rule is also
   * considered to support that group's defaults.
   *
   * @param ruleContext analysis context for the rule. A rule error is triggered here if
   *     invalid constraint settings are discovered.
   * @return the environments this rule supports, not counting defaults "unknown" to this rule
   *     as described above. Returns null if any errors are encountered.
   */
  @Nullable
  public static EnvironmentCollection getSupportedEnvironments(RuleContext ruleContext) {
    if (!validateAttributes(ruleContext)) {
      return null;
    }

    // This rule's rule class defaults (or null if the rule class has no defaults).
    EnvironmentCollector ruleClassCollector = maybeGetRuleClassDefaults(ruleContext);
    // Default environments for this rule. If the rule has rule class defaults, this is
    // those defaults. Otherwise it's the global defaults specified by environment_group
    // declarations.
    DefaultsProvider ruleDefaults;

    if (ruleClassCollector != null) {
      if (!ruleClassCollector.validateEnvironmentSpecifications()) {
        return null;
      }
      ruleDefaults = new RuleClassDefaultsProvider(ruleClassCollector.getEnvironments());
    } else {
      ruleDefaults = new GroupDefaultsProvider();
    }

    EnvironmentCollector ruleCollector = new EnvironmentCollector(ruleContext,
        RuleClass.RESTRICTED_ENVIRONMENT_ATTR, RuleClass.COMPATIBLE_ENVIRONMENT_ATTR, ruleDefaults);
    if (!ruleCollector.validateEnvironmentSpecifications()) {
      return null;
    }

    EnvironmentCollection supportedEnvironments = ruleCollector.getEnvironments();
    if (ruleClassCollector != null) {
      // If we have rule class defaults from groups that aren't referenced from the rule itself,
      // we need to add them in too to override the global defaults.
      supportedEnvironments =
          addUnknownGroupsToCollection(supportedEnvironments, ruleClassCollector.getEnvironments());
    }
    return supportedEnvironments;
  }

  /**
   * Returns the rule class defaults specified for this rule, or null if there are
   * no such defaults.
   */
  @Nullable
  private static EnvironmentCollector maybeGetRuleClassDefaults(RuleContext ruleContext) {
    Rule rule = ruleContext.getRule();
    String restrictionAttr = RuleClass.DEFAULT_RESTRICTED_ENVIRONMENT_ATTR;
    String compatibilityAttr = RuleClass.DEFAULT_COMPATIBLE_ENVIRONMENT_ATTR;

    if (rule.isAttrDefined(restrictionAttr, BuildType.LABEL_LIST)
      || rule.isAttrDefined(compatibilityAttr, BuildType.LABEL_LIST)) {
      return new EnvironmentCollector(ruleContext, restrictionAttr, compatibilityAttr,
          new GroupDefaultsProvider());
    } else {
      return null;
    }
  }

  /**
   * Adds environments to an {@link EnvironmentCollection} from groups that aren't already
   * a part of that collection.
   *
   * @param environments the collection to add to
   * @param toAdd the collection to add. All environments in this collection in groups
   *     that aren't represented in {@code environments} are added to {@code environments}.
   * @return the expanded collection.
   */
  private static EnvironmentCollection addUnknownGroupsToCollection(
      EnvironmentCollection environments, EnvironmentCollection toAdd) {
    EnvironmentCollection.Builder builder = new EnvironmentCollection.Builder();
    builder.putAll(environments);
    for (EnvironmentGroup candidateGroup : toAdd.getGroups()) {
      if (!environments.getGroups().contains(candidateGroup)) {
        builder.putAll(candidateGroup, toAdd.getEnvironments(candidateGroup));
      }
    }
    return builder.build();
  }

  /**
   * Validity-checks this rule's constraint-related attributes. Returns true if all is good,
   * returns false and reports appropriate errors if there are any problems.
   */
  private static boolean validateAttributes(RuleContext ruleContext) {
    AttributeMap attributes = ruleContext.attributes();

    // Report an error if "restricted to" is explicitly set to nothing. Even if this made
    // conceptual sense, we don't know which groups we should apply that to.
    String restrictionAttr = RuleClass.RESTRICTED_ENVIRONMENT_ATTR;
    List<? extends TransitiveInfoCollection> restrictionEnvironments = ruleContext
        .getPrerequisites(restrictionAttr, RuleConfiguredTarget.Mode.DONT_CHECK);
    if (restrictionEnvironments.isEmpty()
        && attributes.isAttributeValueExplicitlySpecified(restrictionAttr)) {
      ruleContext.attributeError(restrictionAttr, "attribute cannot be empty");
      return false;
    }

    return true;
  }

  /**
   * Performs constraint checking on the given rule's dependencies and reports any errors.
   *
   * @param ruleContext the rule to analyze
   * @param ruleEnvironments the rule's supported environments, as defined by the return
   *     value of {@link #getSupportedEnvironments}. In particular, for any environment group that's
   *     not in this collection, the rule is assumed to support the defaults for that group.
   */
  public static void checkConstraints(RuleContext ruleContext,
      EnvironmentCollection ruleEnvironments) {
    for (TransitiveInfoCollection dependency : getConstraintCheckedDependencies(ruleContext)) {
      SupportedEnvironmentsProvider depProvider =
          dependency.getProvider(SupportedEnvironmentsProvider.class);
      Collection<Label> unsupportedEnvironments =
          getUnsupportedEnvironments(depProvider.getEnvironments(), ruleEnvironments);

      if (!unsupportedEnvironments.isEmpty()) {
        ruleContext.ruleError("dependency " + dependency.getLabel()
            + " doesn't support expected environment"
            + (unsupportedEnvironments.size() == 1 ? "" : "s")
            + ": " + Joiner.on(", ").join(unsupportedEnvironments));
      }
    }
  }

  /**
   * Given a collection of environments and a collection of expected environments, returns the
   * missing environments that would cause constraint expectations to be violated. Includes
   * the effects of environment group defaults.
   */
  public static Collection<Label> getUnsupportedEnvironments(
      EnvironmentCollection actualEnvironments, EnvironmentCollection expectedEnvironments) {
    Set<Label> missingEnvironments = new LinkedHashSet<>();
    Collection<Label> actualEnvironmentLabels = actualEnvironments.getEnvironments();

    // Check if each explicitly expected environment is satisfied.
    for (EnvironmentWithGroup expectedEnv : expectedEnvironments.getGroupedEnvironments()) {
      EnvironmentGroup group = expectedEnv.group();
      Label environment = expectedEnv.environment();
      boolean isSatisfied = false;
      if (actualEnvironments.getGroups().contains(group)) {
        // If the actual environments include members from the expected environment's group, we
        // need to either find the environment itself or another one that transitively fulfills it.
        if (actualEnvironmentLabels.contains(environment)
            || intersect(actualEnvironmentLabels, group.getFulfillers(environment))) {
          isSatisfied = true;
        }
      } else {
        // If the actual environments don't reference the expected environment's group at all,
        // the group's defaults are implicitly included. So we need to check those defaults for
        // either the expected environment or another environment that transitively fulfills it.
        if (group.isDefault(environment)
            || intersect(group.getFulfillers(environment), group.getDefaults())) {
          isSatisfied = true;
        }
      }
      if (!isSatisfied) {
        missingEnvironments.add(environment);
      }
    }

    // For any environment group not referenced by the expected environments, its defaults are
    // implicitly expected. We can ignore this if the actual environments also don't reference the
    // group (since in that case the same defaults apply), otherwise we have to check.
    for (EnvironmentGroup group : actualEnvironments.getGroups()) {
      if (!expectedEnvironments.getGroups().contains(group)) {
        for (Label expectedDefault : group.getDefaults()) {
          if (!actualEnvironmentLabels.contains(expectedDefault)
              && !intersect(actualEnvironmentLabels, group.getFulfillers(expectedDefault))) {
            missingEnvironments.add(expectedDefault);
          }
        }
      }
    }

    return missingEnvironments;
  }

  private static boolean intersect(Iterable<Label> labels1, Iterable<Label> labels2) {
    return !Sets.intersection(Sets.newHashSet(labels1), Sets.newHashSet(labels2)).isEmpty();
  }

  /**
   * Returns all dependencies that should be constraint-checked against the current rule.
   */
  private static Iterable<TransitiveInfoCollection> getConstraintCheckedDependencies(
      RuleContext ruleContext) {
    Set<TransitiveInfoCollection> depsToCheck = new LinkedHashSet<>();
    AttributeMap attributes = ruleContext.attributes();

    for (String attr : attributes.getAttributeNames()) {
      Attribute attrDef = attributes.getAttributeDefinition(attr);
      Type<?> attrType = attributes.getAttributeType(attr);

      // TODO(bazel-team): support a user-definable API for choosing which attributes are checked
      if (!attrDef.checkConstraintsOverride()) {
        if ((attrType != BuildType.LABEL && attrType != BuildType.LABEL_LIST)
            || RuleClass.isConstraintAttribute(attr)
            || attr.equals("visibility")
            // Use the same implicit deps check that query uses. This facilitates running queries to
            // determine exactly which rules need to be constraint-annotated for depot migrations.
            || !DependencyFilter.NO_IMPLICIT_DEPS.apply(ruleContext.getRule(), attrDef)
            // We can't identify host deps by calling BuildConfiguration.isHostConfiguration()
            // because --nodistinct_host_configuration subverts that call.
            || attrDef.getConfigurationTransition() == Attribute.ConfigurationTransition.HOST) {
          continue;
        }
      }

      for (TransitiveInfoCollection dep :
          ruleContext.getPrerequisites(attr, RuleConfiguredTarget.Mode.DONT_CHECK)) {
        // Output files inherit the environment spec of their generating rule.
        if (dep instanceof OutputFileConfiguredTarget) {
          // Note this reassignment means constraint violation errors reference the generating
          // rule, not the file. This makes the source of the environmental mismatch more clear.
          dep = ((OutputFileConfiguredTarget) dep).getGeneratingRule();
        }
        // Input files don't support environments. We may subsequently opt them into constraint
        // checking, but for now just pass them by.
        if (dep.getProvider(SupportedEnvironmentsProvider.class) != null) {
          depsToCheck.add(dep);
        }
      }
    }

    return depsToCheck;
  }
}
