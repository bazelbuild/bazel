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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Ident;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
   * The type of configuration that is needed for all dependencies of rules of
   * this class on the parent/Java/Python axis. This is orthogonal to the
   * host/target distinction.
   *
   * <p>For a bit more detailed explanation why this is necessary, check
   * {@code BuildView.restoreParentConfigurationIfNecessary}.
   */
  public enum DependentTargetConfiguration {
    /**
     * Use the same configuration as the original rule.
     */
    SAME,
    /**
     * Revert back to parent configuration, provided that the dependency
     * cannot contribute to the dynamically loaded library.
     */
    PARENT,
  }

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
        return param.getRuleClass() + " rules have to be under a " +
            StringUtil.joinEnglishList(values, "or", "'") + " directory";
      } else if (pathSegment == 1) {
        return param.getRuleClass() + " rules are only allowed in "
            + StringUtil.joinEnglishList(StringUtil.append(values, "//", ""), "or");
      } else {
          return param.getRuleClass() + " rules are only allowed in packages which " +
              StringUtil.ordinal(pathSegment) + " is " + StringUtil.joinEnglishList(values, "or");
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
              (name.contains("$") && !TargetUtils.isTestRuleName(name)) || name.equals(""));
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
          Preconditions.checkArgument(!TargetUtils.isTestRuleName(name)
              && RULE_NAME_PATTERN.matcher(name).matches());
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          for (Attribute attribute : REQUIRED_ATTRIBUTES_FOR_NORMAL_RULES) {
            Attribute presentAttribute = attributes.get(attribute.getName());
            Preconditions.checkState(presentAttribute != null,
                "Missing mandatory '%s' attribute in normal rule class.", attribute.getName());
            Preconditions.checkState(presentAttribute.getType().equals(attribute.getType()),
                "Mandatory attribute '%s' in normal rule class has incorrect type (expcected" +
                    " %s).", attribute.getName(), attribute.getType());
          }
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
        return (o instanceof RuleClassNamePredicate) &&
            ruleClasses.equals(((RuleClassNamePredicate) o).ruleClasses);
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
        attr("env", Type.STRING_LIST).build(),
        attr("flaky", Type.BOOLEAN).build(),
        attr("shard_count", Type.INTEGER).build(),
        attr("local", Type.BOOLEAN).build()
    );

    private final String name;
    private final RuleClassType type;
    private final boolean skylark;
    private boolean documented;
    private boolean binaryOutput = true;
    private DependentTargetConfiguration dependentTargetConfiguration;
    private ImplicitOutputsFunction implicitOutputsFunction = ImplicitOutputsFunction.NONE;
    private Configurator<?, ?> configurator = NO_CHANGE;
    private PredicateWithMessage<Rule> validityPredicate =
        PredicatesWithMessage.<Rule>alwaysTrue();
    private Predicate<String> preferredDependencyPredicate = Predicates.alwaysFalse();
    private UserDefinedFunction configuredTargetFunction = null;
    private SkylarkEnvironment ruleDefinitionEnvironment = null;
    private boolean allowConfigurableAttributes = false;

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
      type.checkName(name);
      this.name = name;
      this.skylark = skylark;
      this.type = type;
      this.documented = type != RuleClassType.ABSTRACT;
      this.dependentTargetConfiguration = DependentTargetConfiguration.SAME;
      for (RuleClass parent : parents) {
        if (parent.getDependentTargetConfiguration() != DependentTargetConfiguration.SAME) {
          setDependentTargetConfiguration(parent.getDependentTargetConfiguration());
        }
        if (parent.getValidityPredicate() != PredicatesWithMessage.<Rule>alwaysTrue()) {
          setValidityPredicate(parent.getValidityPredicate());
        }
        if (parent.preferredDependencyPredicate != Predicates.<String>alwaysFalse()) {
          setPreferredDependencyPredicate(parent.preferredDependencyPredicate);
        }

        for (Attribute attribute : parent.getAttributes()) {
          String attrName = attribute.getName();
          Preconditions.checkArgument(
              !attributes.containsKey(attrName) || attributes.get(attrName) == attribute,
              String.format("Attribute %s is inherited multiple times in %s ruleclass",
                  attrName, name));
          attributes.put(attrName, attribute);
        }
      }
      // TODO(bazel-team): move this testonly attribute setting to somewhere else
      // preferably to some base RuleClass implementation.
      if (this.type.equals(RuleClassType.TEST)) {
        if (attributes.containsKey("testonly")) {
          override(attr("testonly", BOOLEAN).nonconfigurable().value(true));
        } else {
          add(attr("testonly", BOOLEAN).nonconfigurable().value(true));
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
      type.checkAttributes(attributes);
      boolean skylarkExecutable =
          skylark && (type == RuleClassType.NORMAL || type == RuleClassType.TEST);
      Preconditions.checkState(skylarkExecutable == (configuredTargetFunction != null));
      Preconditions.checkState(skylarkExecutable == (ruleDefinitionEnvironment != null));
      return new RuleClass(name, skylarkExecutable, documented, binaryOutput,
          dependentTargetConfiguration, implicitOutputsFunction, configurator,
          validityPredicate, preferredDependencyPredicate,
          configuredTargetFunction, ruleDefinitionEnvironment, allowConfigurableAttributes,
          attributes.values().toArray(new Attribute[0]));
    }

    public Builder setUndocumented() {
      documented = false;
      return this;
    }

    public Builder allowConfigurableAttributes(boolean allow) {
      this.allowConfigurableAttributes = allow;
      return this;
    }

    /**
     * Determines the configuration dependent targets need to have. For more
     * information, see {@link DependentTargetConfiguration}.
     */
    public Builder setDependentTargetConfiguration(DependentTargetConfiguration config) {
      dependentTargetConfiguration = config;
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

    public Builder setValidityPredicate(PredicateWithMessage<Rule> predicate) {
      this.validityPredicate = predicate;
      return this;
    }

    public Builder setPreferredDependencyPredicate(Predicate<String> predicate) {
      this.preferredDependencyPredicate = predicate;
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
     * Adds the attribute to the rule class. Meant for Skylark usage.
     */
    public void add(Attribute attr) {
      addAttribute(attr);
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
     * Overrides the attribute in the rule class. Meant for Skylark usage.
     *
     * @throws IllegalArgumentException if the attribute does not override one of the same name
     */
    public void override(Attribute attr) {
      overrideAttribute(attr);
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
  private final boolean binaryOutput;

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
  private final Function<AttributeMap, Iterable<String>> implicitOutputsFunction;

  /**
   * The set of implicit outputs generated by a rule, expressed as a function
   * of that rule.
   */
  private final Configurator<?, ?> configurator;

  /**
   * The configuration dependent targets need to have.
   */
  private final DependentTargetConfiguration dependentTargetConfiguration;

  /**
   * The constraint the package name of the rule instance must fulfill
   */
  private final PredicateWithMessage<Rule> validityPredicate;

  /**
   * See {@link #isPreferredDependency}.
   */
  private final Predicate<String> preferredDependencyPredicate;

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
   * Temporary gate while the configurable attributes feature is under development
   */
  private final boolean allowConfigurableAttributes;

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
   */
  @VisibleForTesting
  RuleClass(String name, boolean skylarkExecutable, boolean documented, boolean binaryOutput,
      DependentTargetConfiguration dependentTargetConfiguration,
      Function<AttributeMap, Iterable<String>> implicitOutputsFunction,
      Configurator<?, ?> configurator,
      PredicateWithMessage<Rule> validityPredicate, Predicate<String> preferredDependencyPredicate,
      @Nullable UserDefinedFunction configuredTargetFunction,
      @Nullable SkylarkEnvironment ruleDefinitionEnvironment, boolean allowConfigurableAttributes,
      Attribute... attributes) {
    this.name = name;
    this.targetKind = name + " rule";
    this.skylarkExecutable = skylarkExecutable;
    this.documented = documented;
    this.binaryOutput = binaryOutput;
    this.dependentTargetConfiguration = dependentTargetConfiguration;
    this.implicitOutputsFunction = implicitOutputsFunction;
    this.configurator = Preconditions.checkNotNull(configurator);
    this.validityPredicate = validityPredicate;
    this.preferredDependencyPredicate = preferredDependencyPredicate;
    this.configuredTargetFunction = configuredTargetFunction;
    this.ruleDefinitionEnvironment = ruleDefinitionEnvironment;
    this.allowConfigurableAttributes = allowConfigurableAttributes;
    // Do not make a defensive copy as builder does that already
    this.attributes = attributes;

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
  public Function<AttributeMap, Iterable<String>> getImplicitOutputsFunction() {
    return implicitOutputsFunction;
  }

  public Configurator<?, ?> getConfigurator() {
    return configurator;
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

  public DependentTargetConfiguration getDependentTargetConfiguration() {
    return dependentTargetConfiguration;
  }

  public PredicateWithMessage<Rule> getValidityPredicate() {
    return validityPredicate;
  }

  /**
   * For --compile_one_dependency: if multiple rules consume the specified target,
   * should we choose this one over the "unpreferred" options?
   */
  public boolean isPreferredDependency(String filename) {
    return preferredDependencyPredicate.apply(filename);
  }

  /**
   * Helper function for {@link RuleFactory#createRule}.
   */
  Rule createRuleWithLabel(Package.AbstractPackageBuilder<?, ?> pkgBuilder, Label ruleLabel,
      Map<String, Object> attributeValues, ErrorEventListener listener, FuncallExpression ast,
      boolean retainAST, Location location) {
    Rule rule = pkgBuilder.newRuleWithLabel(ruleLabel, this, retainAST ? ast : null,
        location);
    createRuleCommon(rule, pkgBuilder, attributeValues, listener, ast);
    return rule;
  }

  private void createRuleCommon(Rule rule, Package.AbstractPackageBuilder<?, ?> pkgBuilder,
      Map<String, Object> attributeValues, ErrorEventListener listener, FuncallExpression ast) {
    populateRuleAttributeValues(
        rule, pkgBuilder, attributeValues, listener, ast);
    rule.populateOutputFiles(listener, pkgBuilder);
    rule.checkForNullLabels();
    rule.checkValidityPredicate(listener);
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
      Package.AbstractPackageBuilder<?, ?> pkgBuilder, Location ruleLocation,
      Map<String, ParsedAttributeValue> attributeValues, ErrorEventListener listener) {
    Rule rule = pkgBuilder.newRuleWithLabel(label, this, null, ruleLocation);
    rule.checkValidityPredicate(listener);

    for (Attribute attribute : rule.getRuleClassObject().getAttributes()) {
      ParsedAttributeValue value = attributeValues.get(attribute.getName());
      if (attribute.isMandatory()) {
        Preconditions.checkState(value != null);
      }

      if (value == null) {
        continue;
      }

      checkAllowedValues(rule, attribute, value.getValue(), listener);
      rule.setAttributeValue(attribute, value.getValue(), value.getExplicitlySpecified());
      rule.setAttributeLocation(attribute, value.getLocation());

      if (attribute.getName().equals("visibility")) {
        // TODO(bazel-team): Verify that this cast works
        rule.setVisibility(PackageFactory.getVisibility((List<Label>) value.getValue()));
      }
    }

    rule.populateOutputFiles(listener, pkgBuilder);
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
                                           Package.AbstractPackageBuilder<?, ?> pkgBuilder,
                                           Map<String, Object> attributeValues,
                                           ErrorEventListener listener,
                                           FuncallExpression ast) {
    BitSet definedAttrs = new BitSet(); //  set of attr indices

    for (Map.Entry<String, Object> entry : attributeValues.entrySet()) {
      String attributeName = entry.getKey();
      Object attributeValue = entry.getValue();
      Integer attrIndex = setRuleAttributeValue(rule, listener, attributeName, attributeValue);
      if (attrIndex != null) {
        definedAttrs.set(attrIndex);
        checkAttrValNonEmpty(rule, listener, attributeValue, attrIndex);
      }
    }

    // Save the location of each non-default attribute definition:
    if (ast != null) {
      for (Argument arg : ast.getArguments()) {
        Ident keyword = arg.getName();
        if (keyword != null) {
          String name = keyword.getName();
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
                           + name + "' rule", listener);
        }

        if (attr.hasComputedDefault()) {
          attrsWithComputedDefaults.add(attr);
        } else {
          Object defaultValue = getAttributeNoncomputedDefaultValue(attr, pkgBuilder);
          checkAttrValNonEmpty(rule, listener, defaultValue, attrIndex);
          checkAllowedValues(rule, attr, defaultValue, listener);
          rule.setAttributeValue(attr, defaultValue, /*explicit=*/false);
        }
      }
    }

    // Evaluate and set any computed defaults now that all non-computed
    // attributes have been set:
    for (Attribute attr : attrsWithComputedDefaults) {
      rule.setAttributeValue(attr, attr.getDefaultValue(rule), /*explicit=*/false);
    }
    checkForDuplicateLabels(rule, listener);
    checkThirdPartyRuleHasLicense(rule, pkgBuilder, listener);
    checkForValidSizeAndTimeoutValues(rule, listener);
  }

  private void checkAttrValNonEmpty(
      Rule rule, ErrorEventListener listener, Object attributeValue, Integer attrIndex) {
    if (attributeValue instanceof List<?>) {
      Attribute attr = getAttribute(attrIndex);
      if (attr.isNonEmpty() && ((List<?>) attributeValue).isEmpty()) {
        rule.reportError(rule.getLabel() + ": non empty " + "attribute '" + attr.getName()
            + "' in '" + name + "' rule '" + rule.getLabel() + "' has to have at least one value",
            listener);
      }
    }
  }

  /**
   * Report an error for each label that appears more than once in a LABEL_LIST attribute
   * of the given rule.
   *
   * @param rule The rule.
   * @param listener The listener to use to report the duplicated deps.
   */
  private static void checkForDuplicateLabels(Rule rule, ErrorEventListener listener) {
    for (Attribute attribute : rule.getAttributes()) {
      if (attribute.getType() == Type.LABEL_LIST) {
        checkForDuplicateLabels(rule, attribute, listener);
      }
    }
  }

  /**
   * Reports an error against the specified rule if it's beneath third_party
   * but does not have a declared license.
   */
  private static void checkThirdPartyRuleHasLicense(Rule rule,
      Package.AbstractPackageBuilder<?, ?> pkgBuilder, ErrorEventListener listener) {
    if (rule.getLabel().getPackageName().startsWith("third_party/")) {
      License license = rule.getLicense();
      if (license == null) {
        license = pkgBuilder.getDefaultLicense();
      }
      if (license == License.NO_LICENSE) {
        rule.reportError("third-party rule '" + rule.getLabel() + "' lacks a license declaration "
                         + "with one of the following types: notice, reciprocal, permissive, "
                         + "restricted, unencumbered, by_exception_only",
                         listener);
      }
    }
  }

  /**
   * Report an error for each label that appears more than once in the given attribute
   * of the given rule.
   *
   * @param rule The rule.
   * @param attribute The attribute to check. Must exist in rule and be of type LABEL_LIST.
   * @param listener The listener to use to report the duplicated deps.
   */
  private static void checkForDuplicateLabels(Rule rule, Attribute attribute,
       ErrorEventListener listener) {
    final String attrName = attribute.getName();
    // This attribute may be selectable, so iterate over each selection possibility in turn.
    // TODO(bazel-team): merge '*' condition into all lists when implemented.
    AggregatingAttributeMapper attributeMap = new AggregatingAttributeMapper(rule);
    for (List<Label> labels : attributeMap.visitAttribute(attrName, Type.LABEL_LIST)) {
      if (!labels.isEmpty()) {
        Set<Label> duplicates = CollectionUtils.duplicatedElementsOf(labels);
        for (Label label : duplicates) {
          rule.reportError(
              String.format("Label '%s' is duplicated in the '%s' attribute of rule '%s'",
              label, attrName, rule.getName()), listener);
        }
      }
    }
  }

  /**
   * Report an error if the rule has a timeout or size attribute that is not a
   * legal value. These attributes appear on all tests.
   *
   * @param rule the rule to check
   * @param listener the listener to use to report the duplicated deps
   */
  private static void checkForValidSizeAndTimeoutValues(Rule rule, ErrorEventListener listener) {
    if (rule.getRuleClassObject().hasAttr("size", Type.STRING)) {
      String size = NonconfigurableAttributeMapper.of(rule).get("size", Type.STRING);
      if (TestSize.getTestSize(size) == null) {
        rule.reportError(
          String.format("In rule '%s', size '%s' is not a valid size.", rule.getName(), size),
          listener);
      }
    }
    if (rule.getRuleClassObject().hasAttr("timeout", Type.STRING)) {
      String timeout = NonconfigurableAttributeMapper.of(rule).get("timeout", Type.STRING);
      if (TestTimeout.getTestTimeout(timeout) == null) {
        rule.reportError(
            String.format(
                "In rule '%s', timeout '%s' is not a valid timeout.", rule.getName(), timeout),
            listener);
      }
    }
  }

  /**
   * Returns the default value for the specified rule attribute.
   *
   * For most rule attributes, the default value is either explicitly specified
   * in the attribute, or implicitly based on the type of the attribute, except
   * for some special cases (e.g. "licenses", "distribs") where it comes from
   * some other source, such as state in the package.
   *
   * Precondition: {@code !attr.hasComputedDefault()}.  (Computed defaults are
   * evaluated in second pass.)
   */
  private static Object getAttributeNoncomputedDefaultValue(Attribute attr,
      Package.AbstractPackageBuilder<?, ?> pkgBuilder) {
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
                                        ErrorEventListener listener,
                                        String attrName,
                                        Object attrVal) {
    if (attrName.equals("name")) {
      return null; // "name" is handled specially
    }

    Integer attrIndex = getAttributeIndex(attrName);
    if (attrIndex == null) {
      rule.reportError(rule.getLabel() + ": no such attribute '" + attrName +
                       "' in '" + name + "' rule", listener);
      return null;
    }

    Attribute attr = getAttribute(attrIndex);
    Object converted;
    try {
      String what = "attribute '" + attrName + "' in '" + name + "' rule";
      // Temporary branch while configurable attributes is being developed. This will eventually
      // universally apply selectableConvert.
      converted = allowConfigurableAttributes
          ? attr.getType().selectableConvert(attrVal, what, rule.getLabel())
          : attr.getType().convert(attrVal, what, rule.getLabel());

      if ((converted instanceof Type.Selector<?>) && !attr.isConfigurable()) {
        rule.reportError(rule.getLabel() + ": attribute \"" + attr.getName()
            + "\" is not configurable", listener);
        return null;
      }

      if ((converted instanceof List<?>) && !(converted instanceof GlobList<?>)) {
        if (attr.isOrderIndependent()) {
          converted = Ordering.natural().sortedCopy((List<? extends Comparable<?>>) converted);
        }
        converted = ImmutableList.copyOf((List<?>) converted);
      }
    } catch (Type.ConversionException e) {
      rule.reportError(rule.getLabel() + ": " + e.getMessage(), listener);
      return null;
    }

    if (attrName.equals("visibility")) {
      List<Label> attrList = (List<Label>) converted;
      if (!attrList.isEmpty() &&
        ConstantRuleVisibility.LEGACY_PUBLIC_LABEL.equals(attrList.get(0))) {
        rule.reportError(rule.getLabel() + ": //visibility:legacy_public only allowed in package "
            + "declaration", listener);
      }
      rule.setVisibility(PackageFactory.getVisibility(attrList));
    }

    checkAllowedValues(rule, attr, converted, listener);
    rule.setAttributeValue(attr, converted, /*explicit=*/true);
    return attrIndex;
  }

  private void checkAllowedValues(Rule rule, Attribute attribute, Object value,
      ErrorEventListener listener) {
    if (attribute.checkAllowedValues()) {
      PredicateWithMessage<Object> allowedValues = attribute.getAllowedValues();
      if (!allowedValues.apply(value)) {
        rule.reportError(String.format(rule.getLabel() + ": invalid value in '%s' attribute: %s",
            attribute.getName(),
            allowedValues.getErrorReason(value)), listener);
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

  /**
   * Returns true iff the outputs of this rule should be created beneath the
   * <i>bin</i> directory, false if beneath <i>genfiles</i>.  For most rule
   * classes, this is a constant, but for genrule, it is a property of the
   * individual rule instance, derived from the 'output_to_bindir' attribute;
   * see Rule.hasBinaryOutput().
   */
  boolean hasBinaryOutput() {
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
}
