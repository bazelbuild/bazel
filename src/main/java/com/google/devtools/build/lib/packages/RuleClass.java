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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Instances of RuleClass encapsulate the set of attributes of a given "class" of rule, such as
 * <code>cc_binary</code>.
 *
 * <p>This is an instance of the "meta-class" pattern for Rules: we achieve using <i>values</i>
 * what subclasses achieve using <i>types</i>.  (The "Design Patterns" book doesn't include this
 * pattern, so think of it as something like a cross between a Flyweight and a State pattern. Like
 * Flyweight, we avoid repeatedly storing data that belongs to many instances. Like State, we
 * delegate from Rule to RuleClass for the specific behavior of that rule (though unlike state, a
 * Rule object never changes its RuleClass).  This avoids the need to declare one Java class per
 * class of Rule, yet achieves the same behavior.)
 *
 * <p>The use of a metaclass also allows us to compute a mapping from Attributes to small integers
 * and share this between all rules of the same metaclass.  This means we can save the attribute
 * dictionary for each rule instance using an array, which is much more compact than a hashtable.
 *
 * <p>Rule classes whose names start with "$" are considered "abstract"; since they are not valid
 * identifiers, they cannot be named in the build language. However, they are useful for grouping
 * related attributes which are inherited.
 *
 * <p>The exact values in this class are important.  In particular:
 * <ul>
 * <li>Changing an attribute from MANDATORY to OPTIONAL creates the potential for null-pointer
 *     exceptions in code that expects a value.
 * <li>Attributes whose names are preceded by a "$" or a ":" are "hidden", and cannot be redefined
 *     in a BUILD file.  They are a useful way of adding a special dependency. By convention,
 *     attributes starting with "$" are implicit dependencies, and those starting with a ":" are
 *     late-bound implicit dependencies, i.e. dependencies that can only be resolved when the
 *     configuration is known.
 * <li>Attributes should not be introduced into the hierarchy higher then necessary.
 * <li>The 'deps' and 'data' attributes are treated specially by the code that builds the runfiles
 *     tree.  All targets appearing in these attributes appears beneath the ".runfiles" tree; in
 *     addition, "deps" may have rule-specific semantics.
 * </ul>
 */
@Immutable
public final class RuleClass {
  /**
   * A constraint for the package name of the Rule instances.
   */
  public static class PackageNameConstraint implements PredicateWithMessage<Rule> {

    public static final int ANY_SEGMENT = 0;

    private final int pathSegment;

    private final Set<String> values;

    /**
     * The pathSegment-th segment of the package must be one of the specified values.
     * The path segment indexing starts from 1.
     */
    public PackageNameConstraint(int pathSegment, String... values) {
      this.values = ImmutableSet.copyOf(values);
      this.pathSegment = pathSegment;
    }

    @Override
    public boolean apply(Rule input) {
      PathFragment path = input.getLabel().getPackageFragment();
      if (pathSegment == ANY_SEGMENT) {
        return path.getFirstSegment(values) != PathFragment.INVALID_SEGMENT;
      } else {
        return path.segmentCount() >= pathSegment
            && values.contains(path.getSegment(pathSegment - 1));
      }
    }

    @Override
    public String getErrorReason(Rule param) {
      if (pathSegment == ANY_SEGMENT) {
        return param.getRuleClass() + " rules have to be under a "
            + StringUtil.joinEnglishList(values, "or", "'") + " directory";
      } else if (pathSegment == 1) {
        return param.getRuleClass() + " rules are only allowed in "
            + StringUtil.joinEnglishList(StringUtil.append(values, "//", ""), "or");
      } else {
          return param.getRuleClass() + " rules are only allowed in packages which "
              + StringUtil.ordinal(pathSegment) + " is " + StringUtil.joinEnglishList(values, "or");
      }
    }

    @VisibleForTesting
    public int getPathSegment() {
      return pathSegment;
    }

    @VisibleForTesting
    public Collection<String> getValues() {
      return values;
    }
  }

  /**
   * Using this callback function, rules can override their own configuration during the
   * analysis phase.
   */
  public interface Configurator<TConfig, TRule> {
    TConfig apply(TRule rule, TConfig configuration);
  }

  /**
   * A factory or builder class for rule implementations.
   */
  public interface ConfiguredTargetFactory<TConfiguredTarget, TContext> {
    /**
     * Returns a fully initialized configured target instance using the given context.
     */
    TConfiguredTarget create(TContext ruleContext) throws InterruptedException;
  }

  /**
   * Default rule configurator, it doesn't change the assigned configuration.
   */
  public static final RuleClass.Configurator<Object, Object> NO_CHANGE =
      new RuleClass.Configurator<Object, Object>() {
        @Override
        public Object apply(Object rule, Object configuration) {
          return configuration;
        }
  };

  /**
   * For Bazel's constraint system: the attribute that declares the set of environments a rule
   * supports, overriding the defaults for their respective groups.
   */
  public static final String RESTRICTED_ENVIRONMENT_ATTR = "restricted_to";

  /**
   * For Bazel's constraint system: the attribute that declares the set of environments a rule
   * supports, appending them to the defaults for their respective groups.
   */
  public static final String COMPATIBLE_ENVIRONMENT_ATTR = "compatible_with";

  /**
   * For Bazel's constraint system: the implicit attribute used to store rule class restriction
   * defaults as specified by {@link Builder#restrictedTo}.
   */
  public static final String DEFAULT_RESTRICTED_ENVIRONMENT_ATTR =
      "$" + RESTRICTED_ENVIRONMENT_ATTR;

  /**
   * For Bazel's constraint system: the implicit attribute used to store rule class compatibility
   * defaults as specified by {@link Builder#compatibleWith}.
   */
  public static final String DEFAULT_COMPATIBLE_ENVIRONMENT_ATTR =
      "$" + COMPATIBLE_ENVIRONMENT_ATTR;

  /**
   * Checks if an attribute is part of the constraint system.
   */
  public static boolean isConstraintAttribute(String attr) {
    return RESTRICTED_ENVIRONMENT_ATTR.equals(attr)
        || COMPATIBLE_ENVIRONMENT_ATTR.equals(attr)
        || DEFAULT_RESTRICTED_ENVIRONMENT_ATTR.equals(attr)
        || DEFAULT_COMPATIBLE_ENVIRONMENT_ATTR.equals(attr);
  }

  /**
   * A support class to make it easier to create {@code RuleClass} instances.
   * This class follows the 'fluent builder' pattern.
   *
   * <p>The {@link #addAttribute} method will throw an exception if an attribute
   * of that name already exists. Use {@link #overrideAttribute} in that case.
   */
  public static final class Builder {
    private static final Pattern RULE_NAME_PATTERN = Pattern.compile("[A-Za-z][A-Za-z0-9_]*");

    /**
     * The type of the rule class, which determines valid names and required
     * attributes.
     */
    public enum RuleClassType {
      /**
       * Abstract rules are intended for rule classes that are just used to
       * factor out common attributes, and for rule classes that are used only
       * internally. These rules cannot be instantiated by a BUILD file.
       *
       * <p>The rule name must contain a '$' and {@link
       * TargetUtils#isTestRuleName} must return false for the name.
       */
      ABSTRACT {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(
              (name.contains("$") && !TargetUtils.isTestRuleName(name)) || name.isEmpty());
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          // No required attributes.
        }
      },

      /**
       * Invisible rule classes should contain a dollar sign so that they cannot be instantiated
       * by the user. They are different from abstract rules in that they can be instantiated
       * at will.
       */
      INVISIBLE {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(name.contains("$"));
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          // No required attributes.
        }
      },

      /**
       * Normal rules are instantiable by BUILD files. Their names must therefore
       * obey the rules for identifiers in the BUILD language. In addition,
       * {@link TargetUtils#isTestRuleName} must return false for the name.
       */
      NORMAL {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(
              !TargetUtils.isTestRuleName(name) && RULE_NAME_PATTERN.matcher(name).matches(),
              "Invalid rule name: %s", name);
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          for (Attribute attribute : REQUIRED_ATTRIBUTES_FOR_NORMAL_RULES) {
            Attribute presentAttribute = attributes.get(attribute.getName());
            Preconditions.checkState(presentAttribute != null,
                "Missing mandatory '%s' attribute in normal rule class.", attribute.getName());
            Preconditions.checkState(presentAttribute.getType().equals(attribute.getType()),
                "Mandatory attribute '%s' in normal rule class has incorrect type (expected"
                + " %s).", attribute.getName(), attribute.getType());
          }
        }
      },

      /**
       * Workspace rules can only be instantiated from a WORKSPACE file. Their names obey the
       * rule for identifiers.
       */
      WORKSPACE {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(RULE_NAME_PATTERN.matcher(name).matches());
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          // No required attributes.
        }
      },

      /**
       * Test rules are instantiable by BUILD files and are handled specially
       * when run with the 'test' command. Their names must obey the rules
       * for identifiers in the BUILD language and {@link
       * TargetUtils#isTestRuleName} must return true for the name.
       *
       * <p>In addition, test rules must contain certain attributes. See {@link
       * Builder#REQUIRED_ATTRIBUTES_FOR_TESTS}.
       */
      TEST {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(TargetUtils.isTestRuleName(name)
              && RULE_NAME_PATTERN.matcher(name).matches());
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          for (Attribute attribute : REQUIRED_ATTRIBUTES_FOR_TESTS) {
            Attribute presentAttribute = attributes.get(attribute.getName());
            Preconditions.checkState(presentAttribute != null,
                "Missing mandatory '%s' attribute in test rule class.", attribute.getName());
            Preconditions.checkState(presentAttribute.getType().equals(attribute.getType()),
                "Mandatory attribute '%s' in test rule class has incorrect type (expcected %s).",
                attribute.getName(), attribute.getType());
          }
        }
      };

      /**
       * Checks whether the given name is valid for the current rule class type.
       *
       * @throws IllegalArgumentException if the name is not valid
       */
      public abstract void checkName(String name);

      /**
       * Checks whether the given set of attributes contains all the required
       * attributes for the current rule class type.
       *
       * @throws IllegalArgumentException if a required attribute is missing
       */
      public abstract void checkAttributes(Map<String, Attribute> attributes);
    }

    /**
     * A predicate that filters rule classes based on their names.
     */
    public static class RuleClassNamePredicate implements Predicate<RuleClass> {

      private final Set<String> ruleClasses;

      public RuleClassNamePredicate(Iterable<String> ruleClasses) {
        this.ruleClasses = ImmutableSet.copyOf(ruleClasses);
      }

      public RuleClassNamePredicate(String... ruleClasses) {
        this.ruleClasses = ImmutableSet.copyOf(ruleClasses);
      }

      public RuleClassNamePredicate() {
        this(ImmutableSet.<String>of());
      }

      @Override
      public boolean apply(RuleClass ruleClass) {
        return ruleClasses.contains(ruleClass.getName());
      }

      @Override
      public int hashCode() {
        return ruleClasses.hashCode();
      }

      @Override
      public boolean equals(Object o) {
        return (o instanceof RuleClassNamePredicate)
            && ruleClasses.equals(((RuleClassNamePredicate) o).ruleClasses);
      }

      @Override
      public String toString() {
        return ruleClasses.isEmpty() ? "nothing" : StringUtil.joinEnglishList(ruleClasses);
      }
    }

    /**
     * List of required attributes for normal rules, name and type.
     */
    public static final List<Attribute> REQUIRED_ATTRIBUTES_FOR_NORMAL_RULES = ImmutableList.of(
        attr("tags", Type.STRING_LIST).build()
    );

    /**
     * List of required attributes for test rules, name and type.
     */
    public static final List<Attribute> REQUIRED_ATTRIBUTES_FOR_TESTS = ImmutableList.of(
        attr("tags", Type.STRING_LIST).build(),
        attr("size", Type.STRING).build(),
        attr("timeout", Type.STRING).build(),
        attr("flaky", Type.BOOLEAN).build(),
        attr("shard_count", Type.INTEGER).build(),
        attr("local", Type.BOOLEAN).build()
    );

    private String name;
    private final RuleClassType type;
    private final boolean skylark;
    private boolean documented;
    private boolean publicByDefault = false;
    private boolean binaryOutput = true;
    private boolean workspaceOnly = false;
    private boolean outputsDefaultExecutable = false;
    private ImplicitOutputsFunction implicitOutputsFunction = ImplicitOutputsFunction.NONE;
    private Configurator<?, ?> configurator = NO_CHANGE;
    private ConfiguredTargetFactory<?, ?> configuredTargetFactory = null;
    private PredicateWithMessage<Rule> validityPredicate =
        PredicatesWithMessage.<Rule>alwaysTrue();
    private Predicate<String> preferredDependencyPredicate = Predicates.alwaysFalse();
    private List<Class<?>> advertisedProviders = new ArrayList<>();
    private UserDefinedFunction configuredTargetFunction = null;
    private SkylarkEnvironment ruleDefinitionEnvironment = null;
    private Set<Class<?>> configurationFragments = new LinkedHashSet<>();
    private boolean failIfMissingConfigurationFragment;
    private boolean supportsConstraintChecking = true;

    private final Map<String, Attribute> attributes = new LinkedHashMap<>();

    /**
     * Constructs a new {@code RuleClassBuilder} using all attributes from all
     * parent rule classes. An attribute cannot exist in more than one parent.
     *
     * <p>The rule type affects the the allowed names and the required
     * attributes (see {@link RuleClassType}).
     *
     * @throws IllegalArgumentException if an attribute with the same name exists
     * in more than one parent
     */
    public Builder(String name, RuleClassType type, boolean skylark, RuleClass... parents) {
      this.name = name;
      this.skylark = skylark;
      this.type = type;
      this.documented = type != RuleClassType.ABSTRACT;
      for (RuleClass parent : parents) {
        if (parent.getValidityPredicate() != PredicatesWithMessage.<Rule>alwaysTrue()) {
          setValidityPredicate(parent.getValidityPredicate());
        }
        if (parent.preferredDependencyPredicate != Predicates.<String>alwaysFalse()) {
          setPreferredDependencyPredicate(parent.preferredDependencyPredicate);
        }
        configurationFragments.addAll(parent.requiredConfigurationFragments);
        failIfMissingConfigurationFragment |= parent.failIfMissingConfigurationFragment;
        supportsConstraintChecking = parent.supportsConstraintChecking;

        for (Attribute attribute : parent.getAttributes()) {
          String attrName = attribute.getName();
          Preconditions.checkArgument(
              !attributes.containsKey(attrName) || attributes.get(attrName) == attribute,
              "Attribute %s is inherited multiple times in %s ruleclass",
              attrName,
              name);
          attributes.put(attrName, attribute);
        }
      }
      // TODO(bazel-team): move this testonly attribute setting to somewhere else
      // preferably to some base RuleClass implementation.
      if (this.type.equals(RuleClassType.TEST)) {
        Attribute.Builder<Boolean> testOnlyAttr = attr("testonly", BOOLEAN).value(true)
            .nonconfigurable("policy decision: this shouldn't depend on the configuration");
        if (attributes.containsKey("testonly")) {
          override(testOnlyAttr);
        } else {
          add(testOnlyAttr);
        }
      }
    }

    /**
     * Checks that required attributes for test rules are present, creates the
     * {@link RuleClass} object and returns it.
     *
     * @throws IllegalStateException if any of the required attributes is missing
     */
    public RuleClass build() {
      return build(name);
    }

    /**
     * Same as {@link #build} except with setting the name parameter.
     */
    public RuleClass build(String name) {
      Preconditions.checkArgument(this.name.isEmpty() || this.name.equals(name));
      type.checkName(name);
      type.checkAttributes(attributes);
      boolean skylarkExecutable =
          skylark && (type == RuleClassType.NORMAL || type == RuleClassType.TEST);
      Preconditions.checkState(
          (type == RuleClassType.ABSTRACT)
          == (configuredTargetFactory == null && configuredTargetFunction == null));
      Preconditions.checkState(skylarkExecutable == (configuredTargetFunction != null));
      Preconditions.checkState(skylarkExecutable == (ruleDefinitionEnvironment != null));
      return new RuleClass(name, skylarkExecutable, documented, publicByDefault, binaryOutput,
          workspaceOnly, outputsDefaultExecutable, implicitOutputsFunction, configurator,
          configuredTargetFactory, validityPredicate, preferredDependencyPredicate,
          ImmutableSet.copyOf(advertisedProviders), configuredTargetFunction,
          ruleDefinitionEnvironment, configurationFragments, failIfMissingConfigurationFragment,
          supportsConstraintChecking, attributes.values().toArray(new Attribute[0]));
    }

    /**
     * Declares that the implementation of this rule class requires the given configuration
     * fragments to be present in the configuration. The value is inherited by subclasses.
     *
     * <p>For backwards compatibility, if the set is empty, all fragments may be accessed. But note
     * that this is only enforced in the {@link com.google.devtools.build.lib.analysis.RuleContext}
     * class.
     */
    public Builder requiresConfigurationFragments(Class<?>... configurationFragment) {
      Collections.addAll(configurationFragments, configurationFragment);
      return this;
    }

    public Builder failIfMissingConfigurationFragment() {
      this.failIfMissingConfigurationFragment = true;
      return this;
    }

    public Builder setUndocumented() {
      documented = false;
      return this;
    }

    public Builder publicByDefault() {
      publicByDefault = true;
      return this;
    }

    public Builder setWorkspaceOnly() {
      workspaceOnly = true;
      return this;
    }

    /**
     * Determines the outputs of this rule to be created beneath the {@code
     * genfiles} directory. By default, files are created beneath the {@code bin}
     * directory.
     *
     * <p>This property is not inherited and this method should not be called by
     * builder of {@link RuleClassType#ABSTRACT} rule class.
     *
     * @throws IllegalStateException if called for abstract rule class builder
     */
    public Builder setOutputToGenfiles() {
      Preconditions.checkState(type != RuleClassType.ABSTRACT,
          "Setting not inherited property (output to genrules) of abstract rule class '%s'", name);
      this.binaryOutput = false;
      return this;
    }

    /**
     * Sets the implicit outputs function of the rule class. The default implicit
     * outputs function is {@link ImplicitOutputsFunction#NONE}.
     *
     * <p>This property is not inherited and this method should not be called by
     * builder of {@link RuleClassType#ABSTRACT} rule class.
     *
     * @throws IllegalStateException if called for abstract rule class builder
     */
    public Builder setImplicitOutputsFunction(
        ImplicitOutputsFunction implicitOutputsFunction) {
      Preconditions.checkState(type != RuleClassType.ABSTRACT,
          "Setting not inherited property (implicit output function) of abstract rule class '%s'",
          name);
      this.implicitOutputsFunction = implicitOutputsFunction;
      return this;
    }

    public Builder cfg(Configurator<?, ?> configurator) {
      Preconditions.checkState(type != RuleClassType.ABSTRACT,
          "Setting not inherited property (cfg) of abstract rule class '%s'", name);
      this.configurator = configurator;
      return this;
    }

    public Builder factory(ConfiguredTargetFactory<?, ?> factory) {
      this.configuredTargetFactory = factory;
      return this;
    }

    public Builder setValidityPredicate(PredicateWithMessage<Rule> predicate) {
      this.validityPredicate = predicate;
      return this;
    }

    public Builder setPreferredDependencyPredicate(Predicate<String> predicate) {
      this.preferredDependencyPredicate = predicate;
      return this;
    }

    /**
     * State that the rule class being built possibly supplies the specified provider to its direct
     * dependencies.
     *
     * <p>When computing the set of aspects required for a rule, only the providers listed here are
     * considered. The presence of a provider here does not mean that the rule <b>must</b> implement
     * said provider, merely that it <b>can</b>. After the configured target is constructed from
     * this rule, aspects will be filtered according to the set of actual providers.
     *
     * <p>This is here so that we can do the loading phase overestimation required for
     * "blaze query", which does not have the configured targets available.
     *
     * <p>It's okay for the rule class eventually not to supply it (possibly based on analysis phase
     * logic), but if a provider is not advertised but is supplied, aspects that require the it will
     * not be evaluated for the rule.
     */
    public Builder advertiseProvider(Class<?>... providers) {
      Collections.addAll(advertisedProviders, providers);
      return this;
    }

    private void addAttribute(Attribute attribute) {
      Preconditions.checkState(!attributes.containsKey(attribute.getName()),
          "An attribute with the name '%s' already exists.", attribute.getName());
      attributes.put(attribute.getName(), attribute);
    }

    private void overrideAttribute(Attribute attribute) {
      String attrName = attribute.getName();
      Preconditions.checkState(attributes.containsKey(attrName),
          "No such attribute '%s' to override in ruleclass '%s'.", attrName, name);
      Type<?> origType = attributes.get(attrName).getType();
      Type<?> newType = attribute.getType();
      Preconditions.checkState(origType.equals(newType),
          "The type of the new attribute '%s' is different from the original one '%s'.",
          newType, origType);
      attributes.put(attrName, attribute);
    }

    /**
     * Builds attribute from the attribute builder and adds it to this rule
     * class.
     *
     * @param attr attribute builder
     */
    public <TYPE> Builder add(Attribute.Builder<TYPE> attr) {
      addAttribute(attr.build());
      return this;
    }

    /**
     * Builds attribute from the attribute builder and overrides the attribute
     * with the same name.
     *
     * @throws IllegalArgumentException if the attribute does not override one of the same name
     */
    public <TYPE> Builder override(Attribute.Builder<TYPE> attr) {
      overrideAttribute(attr.build());
      return this;
    }

    /**
     * Adds or overrides the attribute in the rule class. Meant for Skylark usage.
     */
    public void addOrOverrideAttribute(Attribute attribute) {
      if (attributes.containsKey(attribute.getName())) {
        overrideAttribute(attribute);
      } else {
        addAttribute(attribute);
      }
    }

    /**
     * Sets the rule implementation function. Meant for Skylark usage.
     */
    public Builder setConfiguredTargetFunction(UserDefinedFunction func) {
      this.configuredTargetFunction = func;
      return this;
    }

    /**
     *  Sets the rule definition environment. Meant for Skylark usage.
     */
    public Builder setRuleDefinitionEnvironment(SkylarkEnvironment env) {
      this.ruleDefinitionEnvironment = env;
      return this;
    }

    /**
     * Removes an attribute with the same name from this rule class.
     *
     * @throws IllegalArgumentException if the attribute with this name does
     * not exist
     */
    public <TYPE> Builder removeAttribute(String name) {
      Preconditions.checkState(attributes.containsKey(name), "No such attribute '%s' to remove.",
          name);
      attributes.remove(name);
      return this;
    }

    /**
     * This rule class outputs a default executable for every rule with the same name as
     * the rules's. Only works for Skylark.
     */
    public <TYPE> Builder setOutputsDefaultExecutable() {
      this.outputsDefaultExecutable = true;
      return this;
    }

    /**
     * Declares that instances of this rule are compatible with the specified environments,
     * in addition to the defaults declared by their environment groups. This can be overridden
     * by rule-specific declarations. See
     * {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics} for details.
     */
    public <TYPE> Builder compatibleWith(Label... environments) {
      add(attr(DEFAULT_COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST).cfg(HOST)
          .value(ImmutableList.copyOf(environments)));
      return this;
    }

    /**
     * Declares that instances of this rule are restricted to the specified environments, i.e.
     * these override the defaults declared by their environment groups. This can be overridden
     * by rule-specific declarations. See
     * {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics} for details.
     *
     * <p>The input list cannot be empty.
     */
    public <TYPE> Builder restrictedTo(Label firstEnvironment, Label... otherEnvironments) {
      ImmutableList<Label> environments = ImmutableList.<Label>builder().add(firstEnvironment)
          .add(otherEnvironments).build();
      add(attr(DEFAULT_RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST).cfg(HOST).value(environments));
      return this;

    }

    /**
     * Exempts rules of this type from the constraint enforcement system. This should only be
     * applied to rules that are intrinsically incompatible with constraint checking (any
     * application of this method weakens the reach and strength of the system).
     *
     * @param reason user-informative message explaining the reason for exemption (not used)
     */
    public <TYPE> Builder exemptFromConstraintChecking(String reason) {
      Preconditions.checkState(this.supportsConstraintChecking);
      this.supportsConstraintChecking = false;
      return this;
    }

    /**
     * Returns an Attribute.Builder object which contains a replica of the
     * same attribute in the parent rule if exists.
     *
     * @param name the name of the attribute
     */
    public Attribute.Builder<?> copy(String name) {
      Preconditions.checkArgument(attributes.containsKey(name),
          "Attribute %s does not exist in parent rule class.", name);
      return attributes.get(name).cloneBuilder();
    }
  }

  private final String name; // e.g. "cc_library"

  /**
   * The kind of target represented by this RuleClass (e.g. "cc_library rule").
   * Note: Even though there is partial duplication with the {@link RuleClass#name} field,
   * we want to store this as a separate field instead of generating it on demand in order to
   * avoid string duplication.
   */
  private final String targetKind;

  private final boolean skylarkExecutable;
  private final boolean documented;
  private final boolean publicByDefault;
  private final boolean binaryOutput;
  private final boolean workspaceOnly;
  private final boolean outputsDefaultExecutable;

  /**
   * A (unordered) mapping from attribute names to small integers indexing into
   * the {@code attributes} array.
   */
  private final Map<String, Integer> attributeIndex = new HashMap<>();

  /**
   *  All attributes of this rule class (including inherited ones) ordered by
   *  attributeIndex value.
   */
  private final Attribute[] attributes;

  /**
   * The set of implicit outputs generated by a rule, expressed as a function
   * of that rule.
   */
  private final ImplicitOutputsFunction implicitOutputsFunction;

  /**
   * The set of implicit outputs generated by a rule, expressed as a function
   * of that rule.
   */
  private final Configurator<?, ?> configurator;

  /**
   * The factory that creates configured targets from this rule.
   */
  private final ConfiguredTargetFactory<?, ?> configuredTargetFactory;

  /**
   * The constraint the package name of the rule instance must fulfill
   */
  private final PredicateWithMessage<Rule> validityPredicate;

  /**
   * See {@link #isPreferredDependency}.
   */
  private final Predicate<String> preferredDependencyPredicate;

  /**
   * The list of transitive info providers this class advertises to aspects.
   */
  private final ImmutableSet<Class<?>> advertisedProviders;

  /**
   * The Skylark rule implementation of this RuleClass. Null for non Skylark executable RuleClasses.
   */
  @Nullable private final UserDefinedFunction configuredTargetFunction;

  /**
   * The Skylark rule definition environment of this RuleClass.
   * Null for non Skylark executable RuleClasses.
   */
  @Nullable private final SkylarkEnvironment ruleDefinitionEnvironment;

  /**
   * The set of required configuration fragments; this should list all fragments that can be
   * accessed by the rule implementation. If empty, all fragments are allowed to be accessed for
   * backwards compatibility.
   */
  private final ImmutableSet<Class<?>> requiredConfigurationFragments;

  /**
   * Whether to fail during analysis if a configuration fragment is missing. The default behavior is
   * to create fail actions for all declared outputs, i.e., to fail during execution, if any of the
   * outputs is actually attempted to be built.
   */
  private final boolean failIfMissingConfigurationFragment;


  /**
   * Determines whether instances of this rule should be checked for constraint compatibility
   * with their dependencies and the rules that depend on them. This should be true for
   * everything except for rules that are intrinsically incompatible with the constraint system.
   */
  private final boolean supportsConstraintChecking;

  /**
   * Constructs an instance of RuleClass whose name is 'name', attributes
   * are 'attributes'. The {@code srcsAllowedFiles} determines which types of
   * files are allowed as parameters to the "srcs" attribute; rules are always
   * allowed. For the "deps" attribute, there are four cases:
   * <ul>
   *   <li>if the parameter is a file, it is allowed if its file type is given
   *       in {@code depsAllowedFiles},
   *   <li>if the parameter is a rule and the rule class is accepted by
   *       {@code depsAllowedRules}, then it is allowed,
   *   <li>if the parameter is a rule and the rule class is not accepted by
   *       {@code depsAllowedRules}, but accepted by
   *       {@code depsAllowedRulesWithWarning}, then it is allowed, but
   *       triggers a warning;
   *   <li>all other parameters trigger an error.
   * </ul>
   *
   * <p>The {@code depsAllowedRules} predicate should have a {@code toString}
   * method which returns a plain English enumeration of the allowed rule class
   * names, if it does not allow all rule classes.
   * @param workspaceOnly
   */
  @VisibleForTesting
  RuleClass(String name,
      boolean skylarkExecutable, boolean documented, boolean publicByDefault,
      boolean binaryOutput, boolean workspaceOnly, boolean outputsDefaultExecutable,
      ImplicitOutputsFunction implicitOutputsFunction,
      Configurator<?, ?> configurator,
      ConfiguredTargetFactory<?, ?> configuredTargetFactory,
      PredicateWithMessage<Rule> validityPredicate, Predicate<String> preferredDependencyPredicate,
      ImmutableSet<Class<?>> advertisedProviders,
      @Nullable UserDefinedFunction configuredTargetFunction,
      @Nullable SkylarkEnvironment ruleDefinitionEnvironment,
      Set<Class<?>> allowedConfigurationFragments, boolean failIfMissingConfigurationFragment,
      boolean supportsConstraintChecking,
      Attribute... attributes) {
    this.name = name;
    this.targetKind = name + " rule";
    this.skylarkExecutable = skylarkExecutable;
    this.documented = documented;
    this.publicByDefault = publicByDefault;
    this.binaryOutput = binaryOutput;
    this.implicitOutputsFunction = implicitOutputsFunction;
    this.configurator = Preconditions.checkNotNull(configurator);
    this.configuredTargetFactory = configuredTargetFactory;
    this.validityPredicate = validityPredicate;
    this.preferredDependencyPredicate = preferredDependencyPredicate;
    this.advertisedProviders = advertisedProviders;
    this.configuredTargetFunction = configuredTargetFunction;
    this.ruleDefinitionEnvironment = ruleDefinitionEnvironment;
    // Do not make a defensive copy as builder does that already
    this.attributes = attributes;
    this.workspaceOnly = workspaceOnly;
    this.outputsDefaultExecutable = outputsDefaultExecutable;
    this.requiredConfigurationFragments = ImmutableSet.copyOf(allowedConfigurationFragments);
    this.failIfMissingConfigurationFragment = failIfMissingConfigurationFragment;
    this.supportsConstraintChecking = supportsConstraintChecking;

    // create the index:
    int index = 0;
    for (Attribute attribute : attributes) {
      attributeIndex.put(attribute.getName(), index++);
    }
  }

  /**
   * Returns the function which determines the set of implicit outputs
   * generated by a given rule.
   *
   * <p>An implicit output is an OutputFile that automatically comes into
   * existence when a rule of this class is declared, and whose name is derived
   * from the name of the rule.
   *
   * <p>Implicit outputs are a widely-relied upon.  All ".so",
   * and "_deploy.jar" targets referenced in BUILD files are examples.
   */
  @VisibleForTesting
  public ImplicitOutputsFunction getImplicitOutputsFunction() {
    return implicitOutputsFunction;
  }

  @SuppressWarnings("unchecked")
  public <C, R> Configurator<C, R> getConfigurator() {
    return (Configurator<C, R>) configurator;
  }

  @SuppressWarnings("unchecked")
  public <CT, RC> ConfiguredTargetFactory<CT, RC> getConfiguredTargetFactory() {
    return (ConfiguredTargetFactory<CT, RC>) configuredTargetFactory;
  }

  /**
   * Returns the class of rule that this RuleClass represents (e.g. "cc_library").
   */
  public String getName() {
    return name;
  }

  /**
   * Returns the target kind of this class of rule (e.g. "cc_library rule").
   */
  String getTargetKind() {
    return targetKind;
  }

  public boolean getWorkspaceOnly() {
    return workspaceOnly;
  }

  /**
   * Returns true iff the attribute 'attrName' is defined for this rule class,
   * and has type 'type'.
   */
  public boolean hasAttr(String attrName, Type<?> type) {
    Integer index = getAttributeIndex(attrName);
    return index != null && getAttribute(index).getType() == type;
  }

  /**
   * Returns the index of the specified attribute name. Use of indices allows
   * space-efficient storage of attribute values in rules, since hashtables are
   * not required. (The index mapping is specific to each RuleClass and an
   * attribute may have a different index in the parent RuleClass.)
   *
   * <p>Returns null if the named attribute is not defined for this class of Rule.
   */
  Integer getAttributeIndex(String attrName) {
    return attributeIndex.get(attrName);
  }

  /**
   * Returns the attribute whose index is 'attrIndex'.  Fails if attrIndex is
   * not in range.
   */
  Attribute getAttribute(int attrIndex) {
    return attributes[attrIndex];
  }

  /**
   * Returns the attribute whose name is 'attrName'; fails if not found.
   */
  public Attribute getAttributeByName(String attrName) {
    return attributes[getAttributeIndex(attrName)];
  }

  /**
   * Returns the attribute whose name is {@code attrName}, or null if not
   * found.
   */
  Attribute getAttributeByNameMaybe(String attrName) {
    Integer i = getAttributeIndex(attrName);
    return i == null ? null : attributes[i];
  }

  /**
   * Returns the number of attributes defined for this rule class.
   */
  public int getAttributeCount() {
    return attributeIndex.size();
  }

  /**
   * Returns an (immutable) list of all Attributes defined for this class of
   * rule, ordered by increasing index.
   */
  public List<Attribute> getAttributes() {
    return ImmutableList.copyOf(attributes);
  }

  public PredicateWithMessage<Rule> getValidityPredicate() {
    return validityPredicate;
  }

  /**
   * Returns the set of advertised transitive info providers.
   *
   * <p>When computing the set of aspects required for a rule, only the providers listed here are
   * considered. The presence of a provider here does not mean that the rule <b>must</b> implement
   * said provider, merely that it <b>can</b>. After the configured target is constructed from this
   * rule, aspects will be filtered according to the set of actual providers.
   *
   * <p>This is here so that we can do the loading phase overestimation required for "blaze query",
   * which does not have the configured targets available.
   *
   * <p>This should in theory only contain subclasses of
   * {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider}, but
   * our current dependency structure does not allow a reference to that class here.
   */
  public ImmutableSet<Class<?>> getAdvertisedProviders() {
    return advertisedProviders;
  }

  /**
   * For --compile_one_dependency: if multiple rules consume the specified target,
   * should we choose this one over the "unpreferred" options?
   */
  public boolean isPreferredDependency(String filename) {
    return preferredDependencyPredicate.apply(filename);
  }

  /**
   * The set of required configuration fragments; this contains all fragments that can be
   * accessed by the rule implementation. If empty, all fragments are allowed to be accessed for
   * backwards compatibility.
   */
  public Set<Class<?>> getRequiredConfigurationFragments() {
    return requiredConfigurationFragments;
  }

  /**
   * Checks if the configuration fragment may be accessed (i.e., if it's declared). If no fragments
   * are declared, this allows access to all fragments for backwards compatibility.
   */
  public boolean isLegalConfigurationFragment(Class<?> configurationFragment) {
    // For now, we allow all rules that don't declare allowed fragments to access any fragment.
    // TODO(bazel-team): Declare fragment dependencies for all rules and remove this.
    if (requiredConfigurationFragments.isEmpty()) {
      return true;
    }
    return requiredConfigurationFragments.contains(configurationFragment);
  }

  /**
   * Whether to fail analysis if any of the required configuration fragments are missing.
   */
  public boolean failIfMissingConfigurationFragment() {
    return failIfMissingConfigurationFragment;
  }

  /**
   * Returns true if rules of this type can be used with the constraint enforcement system.
   */
  public boolean supportsConstraintChecking() {
    return supportsConstraintChecking;
  }

  /**
   * Helper function for {@link RuleFactory#createAndAddRule}.
   */
  Rule createRuleWithLabel(Package.AbstractBuilder<?, ?> pkgBuilder, Label ruleLabel,
      Map<String, Object> attributeValues, EventHandler eventHandler, FuncallExpression ast,
      Location location) throws SyntaxException {
    Rule rule = pkgBuilder.newRuleWithLabel(ruleLabel, this, null, location);
    createRuleCommon(rule, pkgBuilder, attributeValues, eventHandler, ast);
    return rule;
  }

  private void createRuleCommon(Rule rule, Package.AbstractBuilder<?, ?> pkgBuilder,
      Map<String, Object> attributeValues, EventHandler eventHandler, FuncallExpression ast)
          throws SyntaxException {
    populateRuleAttributeValues(
        rule, pkgBuilder, attributeValues, eventHandler, ast);
    rule.populateOutputFiles(eventHandler, pkgBuilder);
    rule.checkForNullLabels();
    rule.checkValidityPredicate(eventHandler);
  }

  static class ParsedAttributeValue {
    private final boolean explicitlySpecified;
    private final Object value;
    private final Location location;

    ParsedAttributeValue(boolean explicitlySpecified, Object value, Location location) {
      this.explicitlySpecified = explicitlySpecified;
      this.value = value;
      this.location = location;
    }

    public boolean getExplicitlySpecified() {
      return explicitlySpecified;
    }

    public Object getValue() {
      return value;
    }

    public Location getLocation() {
      return location;
    }
  }

  /**
   * Creates a rule with the attribute values that are already parsed.
   *
   * <p><b>WARNING:</b> This assumes that the attribute values here have the right type and
   * bypasses some sanity checks. If they are of the wrong type, everything will come down burning.
   */
  @SuppressWarnings("unchecked")
  Rule createRuleWithParsedAttributeValues(Label label,
      Package.AbstractBuilder<?, ?> pkgBuilder, Location ruleLocation,
      Map<String, ParsedAttributeValue> attributeValues, EventHandler eventHandler)
          throws SyntaxException{
    Rule rule = pkgBuilder.newRuleWithLabel(label, this, null, ruleLocation);
    rule.checkValidityPredicate(eventHandler);

    for (Attribute attribute : rule.getRuleClassObject().getAttributes()) {
      ParsedAttributeValue value = attributeValues.get(attribute.getName());
      if (attribute.isMandatory()) {
        Preconditions.checkState(value != null);
      }

      if (value == null) {
        continue;
      }

      checkAllowedValues(rule, attribute, value.getValue(), eventHandler);
      rule.setAttributeValue(attribute, value.getValue(), value.getExplicitlySpecified());
      rule.setAttributeLocation(attribute, value.getLocation());

      if (attribute.getName().equals("visibility")) {
        // TODO(bazel-team): Verify that this cast works
        rule.setVisibility(PackageFactory.getVisibility((List<Label>) value.getValue()));
      }
    }

    rule.populateOutputFiles(eventHandler, pkgBuilder);
    Preconditions.checkState(!rule.containsErrors());
    return rule;
  }

  /**
   * Populates the attributes table of new rule "rule" from the
   * "attributeValues" mapping from attribute names to values in the build
   * language.  Errors are reported on "reporter".  "ast" is used to associate
   * location information with each rule attribute.
   */
  private void populateRuleAttributeValues(Rule rule,
                                           Package.AbstractBuilder<?, ?> pkgBuilder,
                                           Map<String, Object> attributeValues,
                                           EventHandler eventHandler,
                                           FuncallExpression ast) {
    BitSet definedAttrs = new BitSet(); //  set of attr indices

    for (Map.Entry<String, Object> entry : attributeValues.entrySet()) {
      String attributeName = entry.getKey();
      Object attributeValue = entry.getValue();
      if (attributeValue == Environment.NONE) {  // Ignore all None values.
        continue;
      }
      Integer attrIndex = setRuleAttributeValue(rule, eventHandler, attributeName, attributeValue);
      if (attrIndex != null) {
        definedAttrs.set(attrIndex);
        checkAttrValNonEmpty(rule, eventHandler, attributeValue, attrIndex);
      }
    }

    // Save the location of each non-default attribute definition:
    if (ast != null) {
      for (Argument.Passed arg : ast.getArguments()) {
        if (arg.isKeyword()) {
          String name = arg.getName();
          Integer attrIndex = getAttributeIndex(name);
          if (attrIndex != null) {
            rule.setAttributeLocation(attrIndex, arg.getValue().getLocation());
          }
        }
      }
    }

    List<Attribute> attrsWithComputedDefaults = new ArrayList<>();

    // Set defaults; ensure that every mandatory attribute has a value.  Use
    // the default if none is specified.
    int numAttributes = getAttributeCount();
    for (int attrIndex = 0; attrIndex < numAttributes; ++attrIndex) {
      if (!definedAttrs.get(attrIndex)) {
        Attribute attr = getAttribute(attrIndex);
        if (attr.isMandatory()) {
          rule.reportError(rule.getLabel() + ": missing value for mandatory "
                           + "attribute '" + attr.getName() + "' in '"
                           + name + "' rule", eventHandler);
        }

        if (attr.hasComputedDefault()) {
          attrsWithComputedDefaults.add(attr);
        } else {
          Object defaultValue = getAttributeNoncomputedDefaultValue(attr, pkgBuilder);
          checkAttrValNonEmpty(rule, eventHandler, defaultValue, attrIndex);
          checkAllowedValues(rule, attr, defaultValue, eventHandler);
          rule.setAttributeValue(attr, defaultValue, /*explicit=*/false);
        }
      }
    }

    // Evaluate and set any computed defaults now that all non-computed
    // TODO(bazel-team): remove this special casing. Thanks to configurable attributes refactoring,
    // computed defaults don't get bound to their final values at this point, so we no longer
    // have to wait until regular attributes have been initialized.
    for (Attribute attr : attrsWithComputedDefaults) {
      rule.setAttributeValue(attr, attr.getDefaultValue(rule), /*explicit=*/false);
    }

    // Now that all attributes are bound to values, collect and store configurable attribute keys.
    populateConfigDependenciesAttribute(rule);
    checkForDuplicateLabels(rule, eventHandler);
    checkThirdPartyRuleHasLicense(rule, pkgBuilder, eventHandler);
    checkForValidSizeAndTimeoutValues(rule, eventHandler);
  }

  /**
   * Collects all labels used as keys for configurable attributes and places them into
   * the special implicit attribute that tracks them.
   */
  private static void populateConfigDependenciesAttribute(Rule rule) {
    RawAttributeMapper attributes = RawAttributeMapper.of(rule);
    Attribute configDepsAttribute = attributes.getAttributeDefinition("$config_dependencies");
    if (configDepsAttribute == null) {
      // Not currently compatible with Skylark rules.
      return;
    }

    Set<Label> configLabels = new LinkedHashSet<>();
    for (Attribute attr : rule.getAttributes()) {
      Type.Selector<?> selector = attributes.getSelector(attr.getName(), attr.getType());
      if (selector != null) {
        for (Label label : selector.getEntries().keySet()) {
          if (!Type.Selector.isReservedLabel(label)) {
            configLabels.add(label);
          }
        }
      }
    }

    rule.setAttributeValue(configDepsAttribute, ImmutableList.copyOf(configLabels),
        /*explicit=*/false);
  }

  private void checkAttrValNonEmpty(
      Rule rule, EventHandler eventHandler, Object attributeValue, Integer attrIndex) {
    if (attributeValue instanceof List<?>) {
      Attribute attr = getAttribute(attrIndex);
      if (attr.isNonEmpty() && ((List<?>) attributeValue).isEmpty()) {
        rule.reportError(rule.getLabel() + ": non empty " + "attribute '" + attr.getName()
            + "' in '" + name + "' rule '" + rule.getLabel() + "' has to have at least one value",
            eventHandler);
      }
    }
  }

  /**
   * Report an error for each label that appears more than once in a LABEL_LIST attribute
   * of the given rule.
   *
   * @param rule The rule.
   * @param eventHandler The eventHandler to use to report the duplicated deps.
   */
  private static void checkForDuplicateLabels(Rule rule, EventHandler eventHandler) {
    for (Attribute attribute : rule.getAttributes()) {
      if (attribute.getType() == Type.LABEL_LIST) {
        checkForDuplicateLabels(rule, attribute, eventHandler);
      }
    }
  }

  /**
   * Reports an error against the specified rule if it's beneath third_party
   * but does not have a declared license.
   */
  private static void checkThirdPartyRuleHasLicense(Rule rule,
      Package.AbstractBuilder<?, ?> pkgBuilder, EventHandler eventHandler) {
    if (rule.getLabel().getPackageName().startsWith("third_party/")) {
      License license = rule.getLicense();
      if (license == null) {
        license = pkgBuilder.getDefaultLicense();
      }
      if (license == License.NO_LICENSE) {
        rule.reportError("third-party rule '" + rule.getLabel() + "' lacks a license declaration "
                         + "with one of the following types: notice, reciprocal, permissive, "
                         + "restricted, unencumbered, by_exception_only",
                         eventHandler);
      }
    }
  }

  /**
   * Report an error for each label that appears more than once in the given attribute
   * of the given rule.
   *
   * @param rule The rule.
   * @param attribute The attribute to check. Must exist in rule and be of type LABEL_LIST.
   * @param eventHandler The eventHandler to use to report the duplicated deps.
   */
  private static void checkForDuplicateLabels(Rule rule, Attribute attribute,
       EventHandler eventHandler) {
    final String attrName = attribute.getName();
    // This attribute may be selectable, so iterate over each selection possibility in turn.
    // TODO(bazel-team): merge '*' condition into all lists when implemented.
    AggregatingAttributeMapper attributeMap = AggregatingAttributeMapper.of(rule);
    for (List<Label> labels : attributeMap.visitAttribute(attrName, Type.LABEL_LIST)) {
      if (!labels.isEmpty()) {
        Set<Label> duplicates = CollectionUtils.duplicatedElementsOf(labels);
        for (Label label : duplicates) {
          rule.reportError(
              String.format("Label '%s' is duplicated in the '%s' attribute of rule '%s'",
              label, attrName, rule.getName()), eventHandler);
        }
      }
    }
  }

  /**
   * Report an error if the rule has a timeout or size attribute that is not a
   * legal value. These attributes appear on all tests.
   *
   * @param rule the rule to check
   * @param eventHandler the eventHandler to use to report the duplicated deps
   */
  private static void checkForValidSizeAndTimeoutValues(Rule rule, EventHandler eventHandler) {
    if (rule.getRuleClassObject().hasAttr("size", Type.STRING)) {
      String size = NonconfigurableAttributeMapper.of(rule).get("size", Type.STRING);
      if (TestSize.getTestSize(size) == null) {
        rule.reportError(
          String.format("In rule '%s', size '%s' is not a valid size.", rule.getName(), size),
          eventHandler);
      }
    }
    if (rule.getRuleClassObject().hasAttr("timeout", Type.STRING)) {
      String timeout = NonconfigurableAttributeMapper.of(rule).get("timeout", Type.STRING);
      if (TestTimeout.getTestTimeout(timeout) == null) {
        rule.reportError(
            String.format(
                "In rule '%s', timeout '%s' is not a valid timeout.", rule.getName(), timeout),
            eventHandler);
      }
    }
  }

  /**
   * Returns the default value for the specified rule attribute.
   *
   * <p>For most rule attributes, the default value is either explicitly specified
   * in the attribute, or implicitly based on the type of the attribute, except
   * for some special cases (e.g. "licenses", "distribs") where it comes from
   * some other source, such as state in the package.
   *
   * <p>Precondition: {@code !attr.hasComputedDefault()}.  (Computed defaults are
   * evaluated in second pass.)
   */
  private static Object getAttributeNoncomputedDefaultValue(Attribute attr,
      Package.AbstractBuilder<?, ?> pkgBuilder) {
    if (attr.getName().equals("licenses")) {
      return pkgBuilder.getDefaultLicense();
    }
    if (attr.getName().equals("distribs")) {
      return pkgBuilder.getDefaultDistribs();
    }
    return attr.getDefaultValue(null);
  }

  /**
   * Sets the value of attribute "attrName" in rule "rule", by converting the
   * build-language value "attrVal" to the appropriate type for the attribute.
   * Returns the attribute index iff successful, null otherwise.
   *
   * <p>In case of failure, error messages are reported on "handler", and "rule"
   * is marked as containing errors.
   */
  @SuppressWarnings("unchecked")
  private Integer setRuleAttributeValue(Rule rule,
                                        EventHandler eventHandler,
                                        String attrName,
                                        Object attrVal) {
    if (attrName.equals("name")) {
      return null; // "name" is handled specially
    }

    Integer attrIndex = getAttributeIndex(attrName);
    if (attrIndex == null) {
      rule.reportError(rule.getLabel() + ": no such attribute '" + attrName
          + "' in '" + name + "' rule", eventHandler);
      return null;
    }

    Attribute attr = getAttribute(attrIndex);
    Object converted;
    try {
      String what = "attribute '" + attrName + "' in '" + name + "' rule";
      converted = attr.getType().selectableConvert(attrVal, what, rule.getLabel());

      if ((converted instanceof Type.Selector<?>) && !attr.isConfigurable()) {
        rule.reportError(rule.getLabel() + ": attribute \"" + attr.getName()
            + "\" is not configurable", eventHandler);
        return null;
      }

      if ((converted instanceof List<?>) && !(converted instanceof GlobList<?>)) {
        if (attr.isOrderIndependent()) {
          converted = Ordering.natural().sortedCopy((List<? extends Comparable<?>>) converted);
        }
        converted = ImmutableList.copyOf((List<?>) converted);
      }
    } catch (Type.ConversionException e) {
      rule.reportError(rule.getLabel() + ": " + e.getMessage(), eventHandler);
      return null;
    }

    if (attrName.equals("visibility")) {
      List<Label> attrList = (List<Label>) converted;
      if (!attrList.isEmpty()
          && ConstantRuleVisibility.LEGACY_PUBLIC_LABEL.equals(attrList.get(0))) {
        rule.reportError(rule.getLabel() + ": //visibility:legacy_public only allowed in package "
            + "declaration", eventHandler);
      }
      rule.setVisibility(PackageFactory.getVisibility(attrList));
    }

    checkAllowedValues(rule, attr, converted, eventHandler);
    rule.setAttributeValue(attr, converted, /*explicit=*/true);
    return attrIndex;
  }

  private void checkAllowedValues(Rule rule, Attribute attribute, Object value,
      EventHandler eventHandler) {
    if (attribute.checkAllowedValues()) {
      PredicateWithMessage<Object> allowedValues = attribute.getAllowedValues();
      if (!allowedValues.apply(value)) {
        rule.reportError(String.format(rule.getLabel() + ": invalid value in '%s' attribute: %s",
            attribute.getName(),
            allowedValues.getErrorReason(value)), eventHandler);
      }
    }
  }

  @Override
  public String toString() {
    return name;
  }

  public boolean isDocumented() {
    return documented;
  }

  public boolean isPublicByDefault() {
    return publicByDefault;
  }

  /**
   * Returns true iff the outputs of this rule should be created beneath the
   * <i>bin</i> directory, false if beneath <i>genfiles</i>.  For most rule
   * classes, this is a constant, but for genrule, it is a property of the
   * individual rule instance, derived from the 'output_to_bindir' attribute;
   * see Rule.hasBinaryOutput().
   */
  @VisibleForTesting
  public boolean hasBinaryOutput() {
    return binaryOutput;
  }

  /**
   * Returns this RuleClass's custom Skylark rule implementation.
   */
  @Nullable public UserDefinedFunction getConfiguredTargetFunction() {
    return configuredTargetFunction;
  }

  /**
   * Returns this RuleClass's rule definition environment.
   */
  @Nullable public SkylarkEnvironment getRuleDefinitionEnvironment() {
    return ruleDefinitionEnvironment;
  }

  /**
   * Returns true if this RuleClass is an executable Skylark RuleClass (i.e. it is
   * Skylark and Normal or Test RuleClass).
   */
  public boolean isSkylarkExecutable() {
    return skylarkExecutable;
  }

  /**
   * Returns true if this rule class outputs a default executable for every rule.
   */
  public boolean outputsDefaultExecutable() {
    return outputsDefaultExecutable;
  }
}
