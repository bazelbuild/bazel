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

import static com.google.devtools.build.lib.packages.Attribute.ANY_RULE;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.Attribute.SkylarkComputedDefaultTemplate;
import com.google.devtools.build.lib.packages.Attribute.SkylarkComputedDefaultTemplate.CannotPrecomputeDefaultsException;
import com.google.devtools.build.lib.packages.BuildType.LabelConversionContext;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.Builder.ThirdPartyLicenseExistencePolicy;
import com.google.devtools.build.lib.packages.RuleFactory.AttributeValues;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Instances of RuleClass encapsulate the set of attributes of a given "class" of rule, such as
 * <code>cc_binary</code>.
 *
 * <p>This is an instance of the "meta-class" pattern for Rules: we achieve using <i>values</i> what
 * subclasses achieve using <i>types</i>. (The "Design Patterns" book doesn't include this pattern,
 * so think of it as something like a cross between a Flyweight and a State pattern. Like Flyweight,
 * we avoid repeatedly storing data that belongs to many instances. Like State, we delegate from
 * Rule to RuleClass for the specific behavior of that rule (though unlike state, a Rule object
 * never changes its RuleClass). This avoids the need to declare one Java class per class of Rule,
 * yet achieves the same behavior.)
 *
 * <p>The use of a metaclass also allows us to compute a mapping from Attributes to small integers
 * and share this between all rules of the same metaclass. This means we can save the attribute
 * dictionary for each rule instance using an array, which is much more compact than a hashtable.
 *
 * <p>Rule classes whose names start with "$" are considered "abstract"; since they are not valid
 * identifiers, they cannot be named in the build language. However, they are useful for grouping
 * related attributes which are inherited.
 *
 * <p>The exact values in this class are important. In particular:
 *
 * <ul>
 *   <li>Changing an attribute from MANDATORY to OPTIONAL creates the potential for null-pointer
 *       exceptions in code that expects a value.
 *   <li>Attributes whose names are preceded by a "$" or a ":" are "hidden", and cannot be redefined
 *       in a BUILD file. They are a useful way of adding a special dependency. By convention,
 *       attributes starting with "$" are implicit dependencies, and those starting with a ":" are
 *       late-bound implicit dependencies, i.e. dependencies that can only be resolved when the
 *       configuration is known.
 *   <li>Attributes should not be introduced into the hierarchy higher then necessary.
 *   <li>The 'deps' and 'data' attributes are treated specially by the code that builds the runfiles
 *       tree. All targets appearing in these attributes appears beneath the ".runfiles" tree; in
 *       addition, "deps" may have rule-specific semantics.
 * </ul>
 *
 * TODO(bazel-team): Consider breaking up this class in more manageable subclasses.
 */
// Non-final only for mocking in tests. Do not subclass!
@Immutable
@AutoCodec
public class RuleClass {

  /**
   * Maximum attributes per RuleClass. Current value was chosen high enough to be considered a
   * non-breaking change for reasonable use. It was also chosen to be low enough to give significant
   * headroom before hitting {@link AttributeContainer}'s limits.
   */
  private static final int MAX_ATTRIBUTES = 150;

  @AutoCodec
  static final Function<? super Rule, Map<String, Label>> NO_EXTERNAL_BINDINGS =
      Functions.<Map<String, Label>>constant(ImmutableMap.<String, Label>of());

  @AutoCodec
  static final Function<? super Rule, Set<String>> NO_OPTION_REFERENCE =
      Functions.<Set<String>>constant(ImmutableSet.<String>of());

  public static final PathFragment THIRD_PARTY_PREFIX = PathFragment.create("third_party");
  public static final PathFragment EXPERIMENTAL_PREFIX = PathFragment.create("experimental");
  public static final String EXEC_COMPATIBLE_WITH_ATTR = "exec_compatible_with";
  public static final String EXEC_PROPERTIES = "exec_properties";
  /*
   * The attribute that declares the set of license labels which apply to this target.
   */
  public static final String APPLICABLE_LICENSES_ATTR = "applicable_licenses";

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

  /** A factory or builder class for rule implementations. */
  public interface ConfiguredTargetFactory<
      TConfiguredTarget, TContext, TActionConflictException extends Throwable> {
    /**
     * Returns a fully initialized configured target instance using the given context.
     *
     * @throws RuleErrorException if configured target creation could not be completed due to rule
     *     errors
     * @throws TActionConflictException if there were conflicts during action registration
     */
    TConfiguredTarget create(TContext ruleContext)
        throws InterruptedException, RuleErrorException, TActionConflictException;

    /**
     * Exception indicating that configured target creation could not be completed. General error
     * messaging should be done via {@link RuleErrorConsumer}; this exception only interrupts
     * configured target creation in cases where it can no longer continue.
     */
    final class RuleErrorException extends Exception {
      RuleErrorException() {
        super();
      }

      RuleErrorException(String message) {
        super(message);
      }

      RuleErrorException(Throwable cause) {
        super(cause);
      }

      RuleErrorException(String message, Throwable cause) {
        super(message, cause);
      }
    }
  }

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
   * Name of the attribute that stores all {@link
   * com.google.devtools.build.lib.rules.config.ConfigRuleClasses} labels this rule references (i.e.
   * select() keys). This is specially populated in {@link #populateRuleAttributeValues}.
   *
   * <p>This isn't technically necessary for builds: select() keys are evaluated in {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction#getConfigConditions} instead of
   * normal dependency resolution because they're needed to determine other dependencies. So there's
   * no intrinsic reason why we need an extra attribute to store them.
   *
   * <p>There are three reasons why we still create this attribute:
   *
   * <ol>
   *   <li>Collecting them once in {@link #populateRuleAttributeValues} instead of multiple times in
   *       ConfiguredTargetFunction saves extra looping over the rule's attributes.
   *   <li>Query's dependency resolution has no equivalent of {@link
   *       com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction#getConfigConditions} and
   *       we need to make sure its coverage remains complete.
   *   <li>Manual configuration trimming uses the normal dependency resolution process to work
   *       correctly and config_setting keys are subject to this trimming.
   * </ol>
   *
   * <p>It should be possible to clean up these issues if we decide we don't want an artificial
   * attribute dependency. But care has to be taken to do that safely.
   */
  public static final String CONFIG_SETTING_DEPS_ATTRIBUTE = "$config_dependencies";

  /**
   * A support class to make it easier to create {@code RuleClass} instances.
   * This class follows the 'fluent builder' pattern.
   *
   * <p>The {@link #addAttribute} method will throw an exception if an attribute
   * of that name already exists. Use {@link #overrideAttribute} in that case.
   */
  public static final class Builder {
    private static final Pattern RULE_NAME_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

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
                "Mandatory attribute '%s' in test rule class has incorrect type (expected %s).",
                attribute.getName(), attribute.getType());
          }
        }
      },

      /**
       * Placeholder rules are only instantiated when packages which refer to non-native rule
       * classes are deserialized. At this time, non-native rule classes can't be serialized. To
       * prevent crashes on deserialization, when a package containing a rule with a non-native rule
       * class is deserialized, the rule is assigned a placeholder rule class. This is compatible
       * with our limited set of package serialization use cases.
       *
       * Placeholder rule class names obey the rule for identifiers.
       */
      PLACEHOLDER {
        @Override
        public void checkName(String name) {
          Preconditions.checkArgument(RULE_NAME_PATTERN.matcher(name).matches(), name);
        }

        @Override
        public void checkAttributes(Map<String, Attribute> attributes) {
          // No required attributes; this rule class cannot have the wrong set of attributes now
          // because, if it did, the rule class would have failed to build before the package
          // referring to it was serialized.
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

    /** A predicate that filters rule classes based on their names. */
    @AutoCodec
    public static class RuleClassNamePredicate {

      private static final RuleClassNamePredicate UNSPECIFIED_INSTANCE =
          new RuleClassNamePredicate(ImmutableSet.of(), PredicateType.UNSPECIFIED, null);

      private final ImmutableSet<String> ruleClassNames;

      private final PredicateType predicateType;

      private final Predicate<String> ruleClassNamePredicate;
      private final Predicate<RuleClass> ruleClassPredicate;
      // if non-null, used ONLY for checking overlap
      @Nullable private final Set<?> overlappable;

      @VisibleForSerialization
      enum PredicateType {
        ONLY,
        All_EXCEPT,
        UNSPECIFIED
      }

      @VisibleForSerialization
      RuleClassNamePredicate(
          ImmutableSet<String> ruleClassNames, PredicateType predicateType, Set<?> overlappable) {
        this.ruleClassNames = ruleClassNames;
        this.predicateType = predicateType;
        this.overlappable = overlappable;

        switch (predicateType) {
          case All_EXCEPT:
            Predicate<String> containing = only(ruleClassNames).asPredicateOfRuleClassName();
            ruleClassNamePredicate =
                new DescribedPredicate<>(
                    Predicates.not(containing), "all but " + containing.toString());
            ruleClassPredicate =
                new DescribedPredicate<>(
                    Predicates.compose(ruleClassNamePredicate, RuleClass::getName),
                    ruleClassNamePredicate.toString());
            break;
          case ONLY:
            ruleClassNamePredicate =
                new DescribedPredicate<>(
                    Predicates.in(ruleClassNames), StringUtil.joinEnglishList(ruleClassNames));
            ruleClassPredicate =
                new DescribedPredicate<>(
                    Predicates.compose(ruleClassNamePredicate, RuleClass::getName),
                    ruleClassNamePredicate.toString());
            break;
          case UNSPECIFIED:
            ruleClassNamePredicate = Predicates.alwaysTrue();
            ruleClassPredicate = Predicates.alwaysTrue();
            break;
          default:
            // This shouldn't happen normally since the constructor is private and within this file.
            throw new IllegalArgumentException(
                "Predicate type was not specified when constructing a RuleClassNamePredicate.");
        }
      }

      public static RuleClassNamePredicate only(Iterable<String> ruleClassNamesAsIterable) {
        ImmutableSet<String> ruleClassNames = ImmutableSet.copyOf(ruleClassNamesAsIterable);
        return new RuleClassNamePredicate(ruleClassNames, PredicateType.ONLY, ruleClassNames);
      }

      public static RuleClassNamePredicate only(String... ruleClasses) {
        return only(Arrays.asList(ruleClasses));
      }

      public static RuleClassNamePredicate allExcept(String... ruleClasses) {
        ImmutableSet<String> ruleClassNames = ImmutableSet.copyOf(ruleClasses);
        Preconditions.checkState(!ruleClassNames.isEmpty(), "Use unspecified() instead");
        return new RuleClassNamePredicate(ruleClassNames, PredicateType.All_EXCEPT, null);
      }

      /**
       * This is a special sentinel value which represents a "default" {@link
       * RuleClassNamePredicate} which is unspecified. Note that a call to its {@link
       * RuleClassNamePredicate#asPredicateOfRuleClass} produces {@code
       * Predicates.<RuleClass>alwaysTrue()}, which is a sentinel value for other parts of bazel.
       */
      public static RuleClassNamePredicate unspecified() {
        return UNSPECIFIED_INSTANCE;
      }

      public final Predicate<String> asPredicateOfRuleClassName() {
        return ruleClassNamePredicate;
      }

      public final Predicate<RuleClass> asPredicateOfRuleClass() {
        return ruleClassPredicate;
      }

      /**
       * Determines whether two {@code RuleClassNamePredicate}s should be considered incompatible as
       * rule class predicate and rule class warning predicate.
       *
       * <p>Specifically, if both list sets of explicit rule class names to permit, those two sets
       * must be disjoint, so the restriction only applies when both predicates have been created by
       * {@link #only}.
       */
      boolean consideredOverlapping(RuleClassNamePredicate that) {
        return this.overlappable != null
            && that.overlappable != null
            && !Collections.disjoint(this.overlappable, that.overlappable);
      }

      @Override
      public int hashCode() {
        return Objects.hash(ruleClassNames, predicateType);
      }

      @Override
      public boolean equals(Object obj) {
        // NOTE: Specifically not checking equality of ruleClassPredicate.
        // By construction, if the name predicates are equals, the rule class predicates are, too.
        return obj instanceof RuleClassNamePredicate
            && ruleClassNames.equals(((RuleClassNamePredicate) obj).ruleClassNames)
            && predicateType.equals(((RuleClassNamePredicate) obj).predicateType);
      }

      @Override
      public String toString() {
        return ruleClassNamePredicate.toString();
      }

      /** A pass-through predicate, except that an explicit {@link #toString()} is provided. */
      private static class DescribedPredicate<T> implements Predicate<T> {
        private final Predicate<T> delegate; // the actual predicate
        private final String description;

        private DescribedPredicate(Predicate<T> delegate, String description) {
          this.delegate = delegate;
          this.description = description;
        }

        @Override
        public boolean apply(T input) {
          return delegate.apply(input);
        }

        @Override
        public int hashCode() {
          return delegate.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
          return obj instanceof DescribedPredicate
              && delegate.equals(((DescribedPredicate<?>) obj).delegate);
        }

        @Override
        public String toString() {
          return description;
        }
      }
    }

    /**
     * Name of default attribute implicitly added to all Skylark RuleClasses that are {@code
     * build_setting}s.
     */
    public static final String SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME = "build_setting_default";

    public static final String BUILD_SETTING_DEFAULT_NONCONFIGURABLE =
        "Build setting defaults are referenced during analysis.";

    /** List of required attributes for normal rules, name and type. */
    public static final ImmutableList<Attribute> REQUIRED_ATTRIBUTES_FOR_NORMAL_RULES =
        ImmutableList.of(attr("tags", Type.STRING_LIST).build());

    /** List of required attributes for test rules, name and type. */
    public static final ImmutableList<Attribute> REQUIRED_ATTRIBUTES_FOR_TESTS =
        ImmutableList.of(
            attr("tags", Type.STRING_LIST).build(),
            attr("size", Type.STRING).build(),
            attr("timeout", Type.STRING).build(),
            attr("flaky", Type.BOOLEAN).build(),
            attr("shard_count", Type.INTEGER).build(),
            attr("local", Type.BOOLEAN).build());

    private String name;
    private final RuleClassType type;
    private final boolean skylark;
    private boolean skylarkTestable = false;
    private boolean documented;
    private boolean publicByDefault = false;
    private boolean binaryOutput = true;
    private boolean workspaceOnly = false;
    private boolean isExecutableSkylark = false;
    private boolean isAnalysisTest = false;
    private boolean hasAnalysisTestTransition = false;
    private boolean hasFunctionTransitionWhitelist = false;
    private boolean hasStarlarkRuleTransition = false;
    private boolean ignoreLicenses = false;
    private ImplicitOutputsFunction implicitOutputsFunction = ImplicitOutputsFunction.NONE;
    private TransitionFactory<Rule> transitionFactory;
    private ConfiguredTargetFactory<?, ?, ?> configuredTargetFactory = null;
    private PredicateWithMessage<Rule> validityPredicate =
        PredicatesWithMessage.<Rule>alwaysTrue();
    private Predicate<String> preferredDependencyPredicate = Predicates.alwaysFalse();
    private AdvertisedProviderSet.Builder advertisedProviders = AdvertisedProviderSet.builder();
    private StarlarkFunction configuredTargetFunction = null;
    private BuildSetting buildSetting = null;
    private Function<? super Rule, Map<String, Label>> externalBindingsFunction =
        NO_EXTERNAL_BINDINGS;
    private Function<? super Rule, ? extends Set<String>> optionReferenceFunction =
        NO_OPTION_REFERENCE;
    /** This field and the next are null iff the rule is native. */
    @Nullable private Label ruleDefinitionEnvironmentLabel;

    @Nullable private String ruleDefinitionEnvironmentHashCode = null;
    private ConfigurationFragmentPolicy.Builder configurationFragmentPolicy =
        new ConfigurationFragmentPolicy.Builder();

    private boolean supportsConstraintChecking = true;

    /**
     * The policy on whether Bazel should enforce that third_party rules declare <code>licenses().
     * </code>. This is only intended for the migration of <a
     * href="https://github.com/bazelbuild/bazel/issues/7444">GitHub #7444</a>. Our final end state
     * is to have no license-related logic whatsoever. But that's going to take some time.
     */
    public enum ThirdPartyLicenseExistencePolicy {
      /**
       * Always do this check, overriding whatever {@link
       * StarlarkSemanticsOptions#incompatibleDisableThirdPartyLicenseChecking} says.
       */
      ALWAYS_CHECK,

      /**
       * Never do this check, overriding whatever {@link
       * StarlarkSemanticsOptions#incompatibleDisableThirdPartyLicenseChecking} says.
       */
      NEVER_CHECK,

      /**
       * Do whatever {@link StarlarkSemanticsOptions#incompatibleDisableThirdPartyLicenseChecking}
       * says.
       */
      USER_CONTROLLABLE
    }

    private ThirdPartyLicenseExistencePolicy thirdPartyLicenseExistencePolicy;

    private final Map<String, Attribute> attributes = new LinkedHashMap<>();
    private final Set<Label> requiredToolchains = new HashSet<>();
    private boolean useToolchainResolution = true;
    private Set<Label> executionPlatformConstraints = new HashSet<>();
    private OutputFile.Kind outputFileKind = OutputFile.Kind.FILE;

    /**
     * Constructs a new {@code RuleClassBuilder} using all attributes from all
     * parent rule classes. An attribute cannot exist in more than one parent.
     *
     * <p>The rule type affects the allowed names and the required
     * attributes (see {@link RuleClassType}).
     *
     * @throws IllegalArgumentException if an attribute with the same name exists
     * in more than one parent
     */
    public Builder(String name, RuleClassType type, boolean skylark, RuleClass... parents) {
      this.name = name;
      this.skylark = skylark;
      this.type = type;
      Preconditions.checkState(skylark || type != RuleClassType.PLACEHOLDER, name);
      this.documented = type != RuleClassType.ABSTRACT;
      for (RuleClass parent : parents) {
        if (parent.getValidityPredicate() != PredicatesWithMessage.<Rule>alwaysTrue()) {
          setValidityPredicate(parent.getValidityPredicate());
        }
        if (parent.preferredDependencyPredicate != Predicates.<String>alwaysFalse()) {
          setPreferredDependencyPredicate(parent.preferredDependencyPredicate);
        }
        configurationFragmentPolicy
            .includeConfigurationFragmentsFrom(parent.getConfigurationFragmentPolicy());
        configurationFragmentPolicy.setMissingFragmentPolicy(
            parent.getConfigurationFragmentPolicy().getMissingFragmentPolicy());
        supportsConstraintChecking = parent.supportsConstraintChecking;

        addRequiredToolchains(parent.getRequiredToolchains());
        useToolchainResolution = parent.useToolchainResolution;
        addExecutionPlatformConstraints(parent.getExecutionPlatformConstraints());

        for (Attribute attribute : parent.getAttributes()) {
          String attrName = attribute.getName();
          Preconditions.checkArgument(
              !attributes.containsKey(attrName) || attributes.get(attrName).equals(attribute),
              "Attribute %s is inherited multiple times in %s ruleclass",
              attrName,
              name);
          attributes.put(attrName, attribute);
        }

        advertisedProviders.addParent(parent.getAdvertisedProviders());
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
      // For built-ins, name == key
      return build(name, name);
    }

    /** Same as {@link #build} except with setting the name and key parameters. */
    public RuleClass build(String name, String key) {
      Preconditions.checkArgument(this.name.isEmpty() || this.name.equals(name));
      type.checkName(name);

      Preconditions.checkArgument(
          attributes.size() <= MAX_ATTRIBUTES,
          "Rule class %s declared too many attributes (%s > %s)",
          name,
          attributes.size(),
          MAX_ATTRIBUTES);

      type.checkAttributes(attributes);
      Preconditions.checkState(
          (type == RuleClassType.ABSTRACT)
              == (configuredTargetFactory == null && configuredTargetFunction == null),
          "Bad combo for %s: %s %s %s",
          name,
          type,
          configuredTargetFactory,
          configuredTargetFunction);
      if (!workspaceOnly) {
        if (skylark) {
          assertSkylarkRuleClassHasImplementationFunction();
          assertSkylarkRuleClassHasEnvironmentLabel();
        }
        Preconditions.checkState(externalBindingsFunction == NO_EXTERNAL_BINDINGS);
      }
      if (type == RuleClassType.PLACEHOLDER) {
        Preconditions.checkNotNull(ruleDefinitionEnvironmentHashCode, this.name);
      }

      if (buildSetting != null) {
        Type<?> type = buildSetting.getType();
        Attribute.Builder<?> attrBuilder =
            attr(SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, type)
                .nonconfigurable(BUILD_SETTING_DEFAULT_NONCONFIGURABLE)
                .mandatory();
        if (BuildType.isLabelType(type)) {
          attrBuilder.allowedFileTypes(FileTypeSet.ANY_FILE);
          attrBuilder.allowedRuleClasses(ANY_RULE);
        }
        this.add(attrBuilder);

        // Build setting rules should opt out of toolchain resolution, since they form part of the
        // configuration.
        this.useToolchainResolution(false);
      }

      return new RuleClass(
          name,
          key,
          type,
          skylark,
          skylarkTestable,
          documented,
          publicByDefault,
          binaryOutput,
          workspaceOnly,
          isExecutableSkylark,
          isAnalysisTest,
          hasAnalysisTestTransition,
          hasFunctionTransitionWhitelist,
          ignoreLicenses,
          implicitOutputsFunction,
          transitionFactory,
          configuredTargetFactory,
          validityPredicate,
          preferredDependencyPredicate,
          advertisedProviders.build(),
          configuredTargetFunction,
          externalBindingsFunction,
          optionReferenceFunction,
          ruleDefinitionEnvironmentLabel,
          ruleDefinitionEnvironmentHashCode,
          configurationFragmentPolicy.build(),
          supportsConstraintChecking,
          thirdPartyLicenseExistencePolicy,
          requiredToolchains,
          useToolchainResolution,
          executionPlatformConstraints,
          outputFileKind,
          attributes.values(),
          buildSetting);
    }

    private void assertSkylarkRuleClassHasImplementationFunction() {
      Preconditions.checkState(
          (type == RuleClassType.NORMAL || type == RuleClassType.TEST)
              == (configuredTargetFunction != null),
          "%s %s",
          type,
          configuredTargetFunction);
    }

    private void assertSkylarkRuleClassHasEnvironmentLabel() {
      Preconditions.checkState(
          (type == RuleClassType.NORMAL
                  || type == RuleClassType.TEST
                  || type == RuleClassType.PLACEHOLDER)
              == (ruleDefinitionEnvironmentLabel != null),
          "Concrete Starlark rule classes can't have null labels: %s %s",
          ruleDefinitionEnvironmentLabel,
          type);
    }

    /**
     * Declares that the implementation of the associated rule class requires the given fragments to
     * be present in this rule's host and target configurations.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(Class<?>... configurationFragments) {
      configurationFragmentPolicy.requiresConfigurationFragments(
          ImmutableSet.<Class<?>>copyOf(configurationFragments));
      return this;
    }

    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in the given configuration that isn't the rule's configuration but
     * is also readable by the rule.
     *
     * <p>You probably don't want to use this, because rules generally shouldn't read configurations
     * other than their own. If you want to declare host config fragments, see
     * {@link com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder}.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragments(ConfigurationTransition transition,
        Class<?>... configurationFragments) {
      configurationFragmentPolicy.requiresConfigurationFragments(
          transition,
          ImmutableSet.<Class<?>>copyOf(configurationFragments));
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
     */
    /**
     * Declares that the implementation of the associated rule class requires the given
     * fragments to be present in the given configuration that isn't the rule's configuration but
     * is also readable by the rule.
     *
     * <p>In contrast to {@link #requiresConfigurationFragments(ConfigurationTransition, Class...)},
     * this method takes Skylark module names of fragments instead of their classes.
     * *
     * <p>You probably don't want to use this, because rules generally shouldn't read configurations
     * other than their own. If you want to declare host config fragments, see
     * {@link com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder}.
     *
     * <p>The value is inherited by subclasses.
     */
    public Builder requiresConfigurationFragmentsBySkylarkModuleName(
        ConfigurationTransition transition, Collection<String> configurationFragmentNames) {
      configurationFragmentPolicy.requiresConfigurationFragmentsBySkylarkModuleName(transition,
          configurationFragmentNames);
      return this;
    }

    public Builder setSkylarkTestable() {
      Preconditions.checkState(skylark, "Cannot set skylarkTestable on a non-Starlark rule");
      skylarkTestable = true;
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

    /**
     * Applies the given transition to all incoming edges for this rule class.
     *
     * <p>This cannot be a {@link SplitTransition} because that requires coordination with the
     * rule's parent: use {@link Attribute.Builder#cfg(TransitionFactory)} on the parent to declare
     * splits.
     *
     * <p>If you need the transition to depend on the rule it's being applied to, use {@link
     * #cfg(TransitionFactory)}.
     */
    public Builder cfg(PatchTransition transition) {
      return cfg((TransitionFactory<Rule>) (unused) -> (transition));
    }

    /**
     * Applies the given transition factory to all incoming edges for this rule class.
     *
     * <p>Unlike {@link #cfg(PatchTransition)}, the factory can examine the rule when deciding what
     * transition to use.
     */
    public Builder cfg(TransitionFactory<Rule> transitionFactory) {
      Preconditions.checkState(type != RuleClassType.ABSTRACT,
          "Setting not inherited property (cfg) of abstract rule class '%s'", name);
      Preconditions.checkState(this.transitionFactory == null,
          "Property cfg has already been set");
      Preconditions.checkNotNull(transitionFactory);
      Preconditions.checkArgument(!transitionFactory.isSplit());
      this.transitionFactory = transitionFactory;
      return this;
    }

    public void setHasStarlarkRuleTransition() {
      hasStarlarkRuleTransition = true;
    }

    public boolean hasStarlarkRuleTransition() {
      return hasStarlarkRuleTransition;
    }

    public Builder factory(ConfiguredTargetFactory<?, ?, ?> factory) {
      this.configuredTargetFactory = factory;
      return this;
    }

    public Builder setThirdPartyLicenseExistencePolicy(ThirdPartyLicenseExistencePolicy policy) {
      this.thirdPartyLicenseExistencePolicy = policy;
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
      for (Class<?> provider : providers) {
        advertisedProviders.addNative(provider);
      }
      return this;
    }

    public Builder advertiseSkylarkProvider(SkylarkProviderIdentifier... skylarkProviders) {
      for (SkylarkProviderIdentifier skylarkProviderIdentifier : skylarkProviders) {
        advertisedProviders.addSkylark(skylarkProviderIdentifier);
      }
      return this;
    }

    /**
     * Set if the rule can have any provider. This is true for "alias" rules like
     * <code>bind</code> .
     */
    public Builder canHaveAnyProvider() {
      advertisedProviders.canHaveAnyProvider();
      return this;
    }

    public Builder addAttribute(Attribute attribute) {
      Preconditions.checkState(!attributes.containsKey(attribute.getName()),
          "An attribute with the name '%s' already exists.", attribute.getName());
      attributes.put(attribute.getName(), attribute);
      return this;
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
     * Builds provided attribute and attaches it to this rule class.
     *
     * <p>Typically rule classes should only declare a handful of attributes - this expectation is
     * enforced when the instance is built.
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
     *
     * @throws IllegalArgumentException if the attribute overrides an existing attribute (will be
     * legal in the future).
     */
    public void addOrOverrideAttribute(Attribute attribute) {
      String name = attribute.getName();
      // Attributes may be overridden in the future.
      Preconditions.checkArgument(!attributes.containsKey(name),
          "There is already a built-in attribute '%s' which cannot be overridden", name);
      addAttribute(attribute);
    }

    /** True if the rule class contains an attribute named {@code name}. */
    public boolean contains(String name) {
      return attributes.containsKey(name);
    }

    /** Sets the rule implementation function. Meant for Skylark usage. */
    public Builder setConfiguredTargetFunction(StarlarkFunction func) {
      this.configuredTargetFunction = func;
      return this;
    }

    public Builder setBuildSetting(BuildSetting buildSetting) {
      this.buildSetting = buildSetting;
      return this;
    }

    public Builder setExternalBindingsFunction(Function<? super Rule, Map<String, Label>> func) {
      this.externalBindingsFunction = func;
      return this;
    }

    /** Sets the rule definition environment label and hash code. Meant for Skylark usage. */
    public Builder setRuleDefinitionEnvironmentLabelAndHashCode(Label label, String hashCode) {
      this.ruleDefinitionEnvironmentLabel = Preconditions.checkNotNull(label, this.name);
      this.ruleDefinitionEnvironmentHashCode = Preconditions.checkNotNull(hashCode, this.name);
      return this;
    }

    public Label getRuleDefinitionEnvironmentLabel() {
      return this.ruleDefinitionEnvironmentLabel;
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
    public <TYPE> Builder setExecutableSkylark() {
      this.isExecutableSkylark = true;
      return this;
    }

    /** This rule class is marked as an analysis test. */
    public Builder setIsAnalysisTest() {
      this.isAnalysisTest = true;
      return this;
    }

    public boolean isAnalysisTest() {
      return this.isAnalysisTest;
    }

    /**
     * This rule class has at least one attribute with an analysis test transition. (A
     * starlark-defined transition using analysis_test_transition()).
     */
    public Builder setHasAnalysisTestTransition() {
      this.hasAnalysisTestTransition = true;
      return this;
    }

    /**
     * This rule class has the _whitelist_function_transition attribute.  Intended only for Skylark
     * rules.
     */
    public <TYPE> Builder setHasFunctionTransitionWhitelist() {
      this.hasFunctionTransitionWhitelist = true;
      return this;
    }

    /**
     * This rule class never declares a license regardless of what the rule's or package's <code>
     * licenses</code> attribute says.
     */
    // TODO(b/130286108): remove the licenses attribute completely from such rules.
    public Builder setIgnoreLicenses() {
      this.ignoreLicenses = true;
      return this;
    }

    public RuleClassType getType() {
      return this.type;
    }

    /**
     * Sets the kind of output files this rule creates.
     * DO NOT USE! This only exists to support the non-open-sourced {@code fileset} rule.
     * {@see OutputFile.Kind}.
     */
    public Builder setOutputFileKind(OutputFile.Kind outputFileKind) {
      this.outputFileKind = Preconditions.checkNotNull(outputFileKind);
      return this;
    }



    /**
     * Declares that instances of this rule are compatible with the specified environments,
     * in addition to the defaults declared by their environment groups. This can be overridden
     * by rule-specific declarations. See
     * {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics} for details.
     */
    public <TYPE> Builder compatibleWith(Label... environments) {
      add(
          attr(DEFAULT_COMPATIBLE_ENVIRONMENT_ATTR, LABEL_LIST)
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
      add(
          attr(DEFAULT_RESTRICTED_ENVIRONMENT_ATTR, LABEL_LIST)
              .value(environments));
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
      attributes.remove(RuleClass.COMPATIBLE_ENVIRONMENT_ATTR);
      attributes.remove(RuleClass.RESTRICTED_ENVIRONMENT_ATTR);
      return this;
    }

    /**
     * Causes rules of this type to implicitly reference the configuration fragments associated with
     * the options its attributes reference.
     *
     * <p>This is only intended for use by {@code config_setting} - other rules should not use this!
     */
    public Builder setOptionReferenceFunctionForConfigSettingOnly(
        Function<? super Rule, ? extends Set<String>> optionReferenceFunction) {
      this.optionReferenceFunction = Preconditions.checkNotNull(optionReferenceFunction);
      return this;
    }

    /**
     * Causes rules of this type to require the specified toolchains be available via toolchain
     * resolution when a target is configured.
     */
    public Builder addRequiredToolchains(Iterable<Label> toolchainLabels) {
      Iterables.addAll(this.requiredToolchains, toolchainLabels);
      return this;
    }

    /**
     * Causes rules of this type to require the specified toolchains be available via toolchain
     * resolution when a target is configured.
     */
    public Builder addRequiredToolchains(Label... toolchainLabels) {
      return this.addRequiredToolchains(Lists.newArrayList(toolchainLabels));
    }

    /**
     * Causes rules to use toolchain resolution to determine the execution platform and toolchains.
     * Rules that are part of configuring toolchains and platforms should set this to {@code false}.
     */
    public Builder useToolchainResolution(boolean flag) {
      this.useToolchainResolution = flag;
      return this;
    }

    /**
     * Adds additional execution platform constraints that apply for all targets from this rule.
     *
     * <p>Please note that this value is inherited by child rules.
     */
    public Builder addExecutionPlatformConstraints(Label... constraints) {
      return this.addExecutionPlatformConstraints(Lists.newArrayList(constraints));
    }

    /**
     * Adds additional execution platform constraints that apply for all targets from this rule.
     *
     * <p>Please note that this value is inherited by child rules.
     */
    public Builder addExecutionPlatformConstraints(Iterable<Label> constraints) {
      Iterables.addAll(this.executionPlatformConstraints, constraints);
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

  private final String key; // Just the name for native, label + name for skylark

  /**
   * The kind of target represented by this RuleClass (e.g. "cc_library rule").
   * Note: Even though there is partial duplication with the {@link RuleClass#name} field,
   * we want to store this as a separate field instead of generating it on demand in order to
   * avoid string duplication.
   */
  private final String targetKind;

  private final RuleClassType type;
  private final boolean isSkylark;
  private final boolean skylarkTestable;
  private final boolean documented;
  private final boolean publicByDefault;
  private final boolean binaryOutput;
  private final boolean workspaceOnly;
  private final boolean isExecutableSkylark;
  private final boolean isAnalysisTest;
  private final boolean hasAnalysisTestTransition;
  private final boolean hasFunctionTransitionWhitelist;
  private final boolean ignoreLicenses;
  private final boolean hasAspects;

  /**
   * A (unordered) mapping from attribute names to small integers indexing into
   * the {@code attributes} array.
   */
  private final Map<String, Integer> attributeIndex;

  /**
   *  All attributes of this rule class (including inherited ones) ordered by
   *  attributeIndex value.
   */
  private final ImmutableList<Attribute> attributes;

  /** Names of the non-configurable attributes of this rule class. */
  private final ImmutableList<String> nonConfigurableAttributes;

  /**
   * The set of implicit outputs generated by a rule, expressed as a function
   * of that rule.
   */
  private final ImplicitOutputsFunction implicitOutputsFunction;

  /**
   * A factory which will produce a configuration transition that should be applied on any edge of
   * the configured target graph that leads into a target of this rule class.
   */
  private final TransitionFactory<Rule> transitionFactory;

  /** The factory that creates configured targets from this rule. */
  private final ConfiguredTargetFactory<?, ?, ?> configuredTargetFactory;

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
  private final AdvertisedProviderSet advertisedProviders;

  /**
   * The Skylark rule implementation of this RuleClass. Null for non Skylark executable RuleClasses.
   */
  @Nullable private final StarlarkFunction configuredTargetFunction;

  /**
   * The BuildSetting associated with this rule. Null for all RuleClasses except Skylark-defined
   * rules that pass {@code build_setting} to their {@code rule()} declaration.
   */
  @Nullable private final BuildSetting buildSetting;

  /**
   * Returns the extra bindings a workspace function adds to the WORKSPACE file.
   */
  private final Function<? super Rule, Map<String, Label>> externalBindingsFunction;

  /**
   * Returns the options referenced by this rule's attributes.
   */
  private final Function<? super Rule, ? extends Set<String>> optionReferenceFunction;

  /**
   * The Skylark rule definition environment's label and hash code of this RuleClass. Null for non
   * Skylark executable RuleClasses.
   */
  @Nullable private final Label ruleDefinitionEnvironmentLabel;

  @Nullable private final String ruleDefinitionEnvironmentHashCode;
  private final OutputFile.Kind outputFileKind;

  /**
   * The set of configuration fragments which are legal for this rule's implementation to access.
   */
  private final ConfigurationFragmentPolicy configurationFragmentPolicy;

  /**
   * Determines whether instances of this rule should be checked for constraint compatibility
   * with their dependencies and the rules that depend on them. This should be true for
   * everything except for rules that are intrinsically incompatible with the constraint system.
   */
  private final boolean supportsConstraintChecking;

  private final ThirdPartyLicenseExistencePolicy thirdPartyLicenseExistencePolicy;

  private final ImmutableSet<Label> requiredToolchains;
  private final boolean useToolchainResolution;
  private final ImmutableSet<Label> executionPlatformConstraints;

  /**
   * Constructs an instance of RuleClass whose name is 'name', attributes are 'attributes'. The
   * {@code srcsAllowedFiles} determines which types of files are allowed as parameters to the
   * "srcs" attribute; rules are always allowed. For the "deps" attribute, there are four cases:
   *
   * <ul>
   *   <li>if the parameter is a file, it is allowed if its file type is given in {@code
   *       depsAllowedFiles},
   *   <li>if the parameter is a rule and the rule class is accepted by {@code depsAllowedRules},
   *       then it is allowed,
   *   <li>if the parameter is a rule and the rule class is not accepted by {@code
   *       depsAllowedRules}, but accepted by {@code depsAllowedRulesWithWarning}, then it is
   *       allowed, but triggers a warning;
   *   <li>all other parameters trigger an error.
   * </ul>
   *
   * <p>The {@code depsAllowedRules} predicate should have a {@code toString} method which returns a
   * plain English enumeration of the allowed rule class names, if it does not allow all rule
   * classes.
   */
  @VisibleForTesting
  RuleClass(
      String name,
      String key,
      RuleClassType type,
      boolean isSkylark,
      boolean skylarkTestable,
      boolean documented,
      boolean publicByDefault,
      boolean binaryOutput,
      boolean workspaceOnly,
      boolean isExecutableSkylark,
      boolean isAnalysisTest,
      boolean hasAnalysisTestTransition,
      boolean hasFunctionTransitionWhitelist,
      boolean ignoreLicenses,
      ImplicitOutputsFunction implicitOutputsFunction,
      TransitionFactory<Rule> transitionFactory,
      ConfiguredTargetFactory<?, ?, ?> configuredTargetFactory,
      PredicateWithMessage<Rule> validityPredicate,
      Predicate<String> preferredDependencyPredicate,
      AdvertisedProviderSet advertisedProviders,
      @Nullable StarlarkFunction configuredTargetFunction,
      Function<? super Rule, Map<String, Label>> externalBindingsFunction,
      Function<? super Rule, ? extends Set<String>> optionReferenceFunction,
      @Nullable Label ruleDefinitionEnvironmentLabel,
      String ruleDefinitionEnvironmentHashCode,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      boolean supportsConstraintChecking,
      ThirdPartyLicenseExistencePolicy thirdPartyLicenseExistencePolicy,
      Set<Label> requiredToolchains,
      boolean useToolchainResolution,
      Set<Label> executionPlatformConstraints,
      OutputFile.Kind outputFileKind,
      Collection<Attribute> attributes,
      @Nullable BuildSetting buildSetting) {
    this.name = name;
    this.key = key;
    this.type = type;
    this.isSkylark = isSkylark;
    this.targetKind = name + Rule.targetKindSuffix();
    this.skylarkTestable = skylarkTestable;
    this.documented = documented;
    this.publicByDefault = publicByDefault;
    this.binaryOutput = binaryOutput;
    this.implicitOutputsFunction = implicitOutputsFunction;
    this.transitionFactory = transitionFactory;
    this.configuredTargetFactory = configuredTargetFactory;
    this.validityPredicate = validityPredicate;
    this.preferredDependencyPredicate = preferredDependencyPredicate;
    this.advertisedProviders = advertisedProviders;
    this.configuredTargetFunction = configuredTargetFunction;
    this.externalBindingsFunction = externalBindingsFunction;
    this.optionReferenceFunction = optionReferenceFunction;
    this.ruleDefinitionEnvironmentLabel = ruleDefinitionEnvironmentLabel;
    this.ruleDefinitionEnvironmentHashCode = ruleDefinitionEnvironmentHashCode;
    this.outputFileKind = outputFileKind;
    validateNoClashInPublicNames(attributes);
    this.attributes = ImmutableList.copyOf(attributes);
    this.workspaceOnly = workspaceOnly;
    this.isExecutableSkylark = isExecutableSkylark;
    this.isAnalysisTest = isAnalysisTest;
    this.hasAnalysisTestTransition = hasAnalysisTestTransition;
    this.hasFunctionTransitionWhitelist = hasFunctionTransitionWhitelist;
    this.ignoreLicenses = ignoreLicenses;
    this.configurationFragmentPolicy = configurationFragmentPolicy;
    this.supportsConstraintChecking = supportsConstraintChecking;
    this.thirdPartyLicenseExistencePolicy = thirdPartyLicenseExistencePolicy;
    this.requiredToolchains = ImmutableSet.copyOf(requiredToolchains);
    this.useToolchainResolution = useToolchainResolution;
    this.executionPlatformConstraints = ImmutableSet.copyOf(executionPlatformConstraints);
    this.buildSetting = buildSetting;

    // Create the index and collect non-configurable attributes.
    int index = 0;
    attributeIndex = Maps.newHashMapWithExpectedSize(attributes.size());
    boolean computedHasAspects = false;
    ImmutableList.Builder<String> nonConfigurableAttributesBuilder = ImmutableList.builder();
    for (Attribute attribute : attributes) {
      computedHasAspects |= attribute.hasAspects();
      attributeIndex.put(attribute.getName(), index++);
      if (!attribute.isConfigurable()) {
        nonConfigurableAttributesBuilder.add(attribute.getName());
      }
    }
    this.hasAspects = computedHasAspects;
    this.nonConfigurableAttributes = nonConfigurableAttributesBuilder.build();
  }

  private void validateNoClashInPublicNames(Iterable<Attribute> attributes) {
    Map<String, Attribute> publicToPrivateNames = new HashMap<>();
    for (Attribute attribute : attributes) {
      String publicName = attribute.getPublicName();
      if (publicToPrivateNames.containsKey(publicName)) {
        throw new IllegalStateException(
            String.format(
                "Rule %s: Attributes %s and %s have an identical public name: %s",
                name,
                attribute.getName(),
                publicToPrivateNames.get(publicName).getName(),
                publicName));
      }
      publicToPrivateNames.put(publicName, attribute);
    }
  }

  /**
   * Returns the default function for determining the set of implicit outputs generated by a given
   * rule. If not otherwise specified, this will be the implementation used by {@link Rule}s
   * created with this {@link RuleClass}.
   *
   * <p>Do not use this value to calculate implicit outputs for a rule, instead use
   * {@link Rule#getImplicitOutputsFunction()}.
   *
   * <p>An implicit output is an OutputFile that automatically comes into existence when a rule of
   * this class is declared, and whose name is derived from the name of the rule.
   *
   * <p>Implicit outputs are a widely-relied upon. All ".so", and "_deploy.jar" targets referenced
   * in BUILD files are examples.
   */
  @VisibleForTesting
  public ImplicitOutputsFunction getDefaultImplicitOutputsFunction() {
    return implicitOutputsFunction;
  }

  public TransitionFactory<Rule> getTransitionFactory() {
    return transitionFactory;
  }

  @SuppressWarnings("unchecked")
  public <CT, RC, ACE extends Throwable>
      ConfiguredTargetFactory<CT, RC, ACE> getConfiguredTargetFactory() {
    return (ConfiguredTargetFactory<CT, RC, ACE>) configuredTargetFactory;
  }

  /**
   * Returns the class of rule that this RuleClass represents (e.g. "cc_library").
   */
  public String getName() {
    return name;
  }

  /** Returns the type of rule that this RuleClass represents. Only for use during serialization. */
  public RuleClassType getRuleClassType() {
    return type;
  }

  /** Returns a unique key. Used for profiling purposes. */
  public String getKey() {
    return key;
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
    return attributes.get(attrIndex);
  }

  /**
   * Returns the attribute whose name is 'attrName'; fails with NullPointerException if not found.
   */
  public Attribute getAttributeByName(String attrName) {
    Integer attrIndex = Preconditions.checkNotNull(getAttributeIndex(attrName),
        "Attribute %s does not exist", attrName);
    return attributes.get(attrIndex);
  }

  /**
   * Returns the attribute whose name is {@code attrName}, or null if not
   * found.
   */
  Attribute getAttributeByNameMaybe(String attrName) {
    Integer i = getAttributeIndex(attrName);
    return i == null ? null : attributes.get(i);
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
    return attributes;
  }

  /** Returns set of non-configurable attribute names defined for this class of rule. */
  public List<String> getNonConfigurableAttributes() {
    return nonConfigurableAttributes;
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
   **/
  public AdvertisedProviderSet getAdvertisedProviders() {
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
   * Returns this rule's policy for configuration fragment access.
   */
  public ConfigurationFragmentPolicy getConfigurationFragmentPolicy() {
    return configurationFragmentPolicy;
  }

  /**
   * Returns true if rules of this type can be used with the constraint enforcement system.
   */
  public boolean supportsConstraintChecking() {
    return supportsConstraintChecking;
  }

  public boolean hasAspects() {
    return hasAspects;
  }

  /**
   * Creates a new {@link Rule} {@code r} where {@code r.getPackage()} is the {@link Package}
   * associated with {@code pkgBuilder}.
   *
   * <p>The created {@link Rule} will be populated with attribute values from {@code
   * attributeValues} or the default attribute values associated with this {@link RuleClass} and
   * {@code pkgBuilder}.
   *
   * <p>The created {@link Rule} will also be populated with output files. These output files will
   * have been collected from the explicitly provided values of type {@link BuildType#OUTPUT} and
   * {@link BuildType#OUTPUT_LIST} as well as from the implicit outputs determined by this {@link
   * RuleClass} and the values in {@code attributeValues}.
   *
   * <p>This performs several validity checks. Invalid output file labels result in a thrown {@link
   * LabelSyntaxException}. Computed default attributes that fail during precomputation result in a
   * {@link CannotPrecomputeDefaultsException}. All other errors are reported on {@code
   * eventHandler}.
   */
  <T> Rule createRule(
      Package.Builder pkgBuilder,
      Label ruleLabel,
      AttributeValues<T> attributeValues,
      EventHandler eventHandler,
      Location location,
      AttributeContainer attributeContainer,
      boolean checkThirdPartyRulesHaveLicenses)
      throws LabelSyntaxException, InterruptedException, CannotPrecomputeDefaultsException {
    Rule rule = pkgBuilder.createRule(ruleLabel, this, location, attributeContainer);
    populateRuleAttributeValues(rule, pkgBuilder, attributeValues, eventHandler);
    checkAspectAllowedValues(rule, eventHandler);
    rule.populateOutputFiles(eventHandler, pkgBuilder);
    checkForDuplicateLabels(rule, eventHandler);

    boolean actuallyCheckLicense;
    if (thirdPartyLicenseExistencePolicy == ThirdPartyLicenseExistencePolicy.ALWAYS_CHECK) {
      actuallyCheckLicense = true;
    } else if (thirdPartyLicenseExistencePolicy == ThirdPartyLicenseExistencePolicy.NEVER_CHECK) {
      actuallyCheckLicense = false;
    } else {
      actuallyCheckLicense = checkThirdPartyRulesHaveLicenses;
    }

    if (actuallyCheckLicense) {
      checkThirdPartyRuleHasLicense(rule, pkgBuilder, eventHandler);
    }
    checkForValidSizeAndTimeoutValues(rule, eventHandler);
    rule.checkValidityPredicate(eventHandler);
    rule.checkForNullLabels();
    return rule;
  }

  /**
   * Same as {@link #createRule}, except without some internal sanity checks.
   *
   * <p>Don't call this function unless you know what you're doing.
   */
  <T> Rule createRuleUnchecked(
      Package.Builder pkgBuilder,
      Label ruleLabel,
      AttributeValues<T> attributeValues,
      Location location,
      AttributeContainer attributeContainer,
      ImplicitOutputsFunction implicitOutputsFunction)
      throws InterruptedException, CannotPrecomputeDefaultsException {
    Rule rule = pkgBuilder.createRule(
        ruleLabel,
        this,
        location,
        attributeContainer,
        implicitOutputsFunction);
    populateRuleAttributeValues(rule, pkgBuilder, attributeValues, NullEventHandler.INSTANCE);
    rule.populateOutputFilesUnchecked(NullEventHandler.INSTANCE, pkgBuilder);
    return rule;
  }

  /**
   * Populates the attributes table of the new {@link Rule} with the values in the {@code
   * attributeValues} map and with default values provided by this {@link RuleClass} and the {@code
   * pkgBuilder}.
   *
   * <p>Errors are reported on {@code eventHandler}.
   */
  private <T> void populateRuleAttributeValues(
      Rule rule,
      Package.Builder pkgBuilder,
      AttributeValues<T> attributeValues,
      EventHandler eventHandler)
      throws InterruptedException, CannotPrecomputeDefaultsException {
    BitSet definedAttrIndices =
        populateDefinedRuleAttributeValues(
            rule,
            pkgBuilder.getRepositoryMapping(),
            attributeValues,
            pkgBuilder.getListInterner(),
            eventHandler);
    populateDefaultRuleAttributeValues(rule, pkgBuilder, definedAttrIndices, eventHandler);
    // Now that all attributes are bound to values, collect and store configurable attribute keys.
    populateConfigDependenciesAttribute(rule);
  }

  /**
   * Populates the attributes table of the new {@link Rule} with the values in the {@code
   * attributeValues} map.
   *
   * <p>Handles the special cases of the attribute named {@code "name"} and attributes with value
   * {@link Starlark#NONE}.
   *
   * <p>Returns a bitset {@code b} where {@code b.get(i)} is {@code true} if this method set a value
   * for the attribute with index {@code i} in this {@link RuleClass}. Errors are reported on {@code
   * eventHandler}.
   */
  private <T> BitSet populateDefinedRuleAttributeValues(
      Rule rule,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      AttributeValues<T> attributeValues,
      Interner<ImmutableList<?>> listInterner,
      EventHandler eventHandler) {
    BitSet definedAttrIndices = new BitSet();
    for (T attributeAccessor : attributeValues.getAttributeAccessors()) {
      String attributeName = attributeValues.getName(attributeAccessor);
      Object attributeValue = attributeValues.getValue(attributeAccessor);
      // Ignore all None values.
      if (attributeValue == Starlark.NONE) {
        continue;
      }

      // Check that the attribute's name belongs to a valid attribute for this rule class.
      Integer attrIndex = getAttributeIndex(attributeName);
      if (attrIndex == null) {
        rule.reportError(
            String.format(
                "%s: no such attribute '%s' in '%s' rule", rule.getLabel(), attributeName, name),
            eventHandler);
        continue;
      }
      Attribute attr = getAttribute(attrIndex);

      if (attributeName.equals("licenses") && ignoreLicenses) {
        setRuleAttributeValue(rule, eventHandler, attr, License.NO_LICENSE, /*explicit=*/ false);
        definedAttrIndices.set(attrIndex);
        continue;
      }

      // Convert the build-lang value to a native value, if necessary.
      Object nativeAttributeValue;
      if (attributeValues.valuesAreBuildLanguageTyped()) {
        try {
          nativeAttributeValue =
              convertFromBuildLangType(rule, attr, attributeValue, repositoryMapping, listInterner);
        } catch (ConversionException e) {
          rule.reportError(String.format("%s: %s", rule.getLabel(), e.getMessage()), eventHandler);
          continue;
        }
      } else {
        nativeAttributeValue = attributeValue;
      }

      boolean explicit = attributeValues.isExplicitlySpecified(attributeAccessor);
      setRuleAttributeValue(rule, eventHandler, attr, nativeAttributeValue, explicit);
      definedAttrIndices.set(attrIndex);
    }
    return definedAttrIndices;
  }

  /**
   * Populates the attributes table of the new {@link Rule} with default values provided by this
   * {@link RuleClass} and the {@code pkgBuilder}. This will only provide values for attributes that
   * haven't already been populated, using {@code definedAttrIndices} to determine whether an
   * attribute was populated.
   *
   * <p>Errors are reported on {@code eventHandler}.
   */
  private void populateDefaultRuleAttributeValues(
      Rule rule, Package.Builder pkgBuilder, BitSet definedAttrIndices, EventHandler eventHandler)
      throws InterruptedException, CannotPrecomputeDefaultsException {
    // Set defaults; ensure that every mandatory attribute has a value. Use the default if none
    // is specified.
    List<Attribute> attrsWithComputedDefaults = new ArrayList<>();
    int numAttributes = getAttributeCount();
    for (int attrIndex = 0; attrIndex < numAttributes; ++attrIndex) {
      if (definedAttrIndices.get(attrIndex)) {
        continue;
      }
      Attribute attr = getAttribute(attrIndex);
      if (attr.isMandatory()) {
        rule.reportError(
            String.format(
                "%s: missing value for mandatory attribute '%s' in '%s' rule",
                rule.getLabel(),
                attr.getName(),
                name),
            eventHandler);
      }

      if (attr.getName().equals("licenses") && ignoreLicenses) {
        rule.setAttributeValue(attr, License.NO_LICENSE, /*explicit=*/ false);
      } else if (attr.hasComputedDefault()) {
        // Note that it is necessary to set all non-computed default values before calling
        // Attribute#getDefaultValue for computed default attributes. Computed default attributes
        // may have a condition predicate (i.e. the predicate returned by Attribute#getCondition)
        // that depends on non-computed default attribute values, and that condition predicate is
        // evaluated by the call to Attribute#getDefaultValue.
        attrsWithComputedDefaults.add(attr);
      } else if (attr.isLateBound()) {
        rule.setAttributeValue(attr, attr.getLateBoundDefault(), /*explicit=*/ false);
      } else {
        Object defaultValue = getAttributeNoncomputedDefaultValue(attr, pkgBuilder);
        rule.setAttributeValue(attr, defaultValue, /*explicit=*/ false);
        checkAllowedValues(rule, attr, eventHandler);
      }
    }

    // Set computed default attribute values now that all other (i.e. non-computed) default values
    // have been set.
    for (Attribute attr : attrsWithComputedDefaults) {
      // If Attribute#hasComputedDefault was true above, Attribute#getDefaultValue returns the
      // computed default function object or a Skylark computed default template. Note that we
      // cannot determine the exact value of the computed default function here because it may
      // depend on other attribute values that are configurable (i.e. they came from select({..})
      // expressions in the build language, and they require configuration data from the analysis
      // phase to be resolved). Instead, we're setting the attribute value to a reference to the
      // computed default function, or if #getDefaultValue is a Skylark computed default
      // template, setting the attribute value to a reference to the SkylarkComputedDefault
      // returned from SkylarkComputedDefaultTemplate#computePossibleValues.
      //
      // SkylarkComputedDefaultTemplate#computePossibleValues pre-computes all possible values the
      // function may evaluate to, and records them in a lookup table. By calling it here, with an
      // EventHandler, any errors that might occur during the function's evaluation can
      // be discovered and propagated here.
      Object valueToSet;
      Object defaultValue = attr.getDefaultValue(rule);
      if (defaultValue instanceof SkylarkComputedDefaultTemplate) {
        SkylarkComputedDefaultTemplate template = (SkylarkComputedDefaultTemplate) defaultValue;
        valueToSet = template.computePossibleValues(attr, rule, eventHandler);
      } else if (defaultValue instanceof ComputedDefault) {
        // Compute all possible values to verify that the ComputedDefault is well-defined. This was
        // previously done implicitly as part of visiting all labels to check for null-ness in
        // Rule.checkForNullLabels, but that was changed to skip non-label attributes to improve
        // performance.
        ((ComputedDefault) defaultValue).getPossibleValues(attr.getType(), rule);
        valueToSet = defaultValue;
      } else {
        valueToSet = defaultValue;
      }
      rule.setAttributeValue(attr, valueToSet, /*explicit=*/ false);
    }
  }

  /**
   * Collects all labels used as keys for configurable attributes and places them into
   * the special implicit attribute that tracks them.
   */
  private static void populateConfigDependenciesAttribute(Rule rule) {
    RawAttributeMapper attributes = RawAttributeMapper.of(rule);
    Attribute configDepsAttribute =
        attributes.getAttributeDefinition(CONFIG_SETTING_DEPS_ATTRIBUTE);
    if (configDepsAttribute == null) {
      return;
    }

    LinkedHashSet<Label> configLabels = new LinkedHashSet<>();
    for (Attribute attr : rule.getAttributes()) {
      SelectorList<?> selectorList = attributes.getSelectorList(attr.getName(), attr.getType());
      if (selectorList != null) {
        configLabels.addAll(selectorList.getKeyLabels());
      }
    }

    rule.setAttributeValue(configDepsAttribute, ImmutableList.copyOf(configLabels),
        /*explicit=*/false);
  }

  public void checkAttributesNonEmpty(
      RuleErrorConsumer ruleErrorConsumer, AttributeMap attributes) {
    for (String attributeName : attributes.getAttributeNames()) {
      Attribute attr = attributes.getAttributeDefinition(attributeName);
      if (!attr.isNonEmpty()) {
        continue;
      }
      Object attributeValue = attributes.get(attributeName, attr.getType());

      boolean isEmpty = false;
      if (attributeValue instanceof Sequence) {
        isEmpty = ((Sequence<?>) attributeValue).isEmpty();
      } else if (attributeValue instanceof List<?>) {
        isEmpty = ((List<?>) attributeValue).isEmpty();
      } else if (attributeValue instanceof Map<?, ?>) {
        isEmpty = ((Map<?, ?>) attributeValue).isEmpty();
      }

      if (isEmpty) {
        ruleErrorConsumer.attributeError(attr.getName(), "attribute must be non empty");
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
      if (attribute.getType() == BuildType.LABEL_LIST) {
        checkForDuplicateLabels(rule, attribute, eventHandler);
      }
    }
  }

  /**
   * Reports an error against the specified rule if it's beneath third_party
   * but does not have a declared license.
   */
  private static void checkThirdPartyRuleHasLicense(Rule rule,
      Package.Builder pkgBuilder, EventHandler eventHandler) {
    if (rule.getRuleClassObject().ignoreLicenses()) {
      // A package license is sufficient; ignore rules that don't include it.
      return;
    }
    if (isThirdPartyPackage(rule.getLabel().getPackageIdentifier())) {
      License license = rule.getLicense();
      if (license == null) {
        license = pkgBuilder.getDefaultLicense();
      }
      if (!license.isSpecified()) {
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
    Set<Label> duplicates = AggregatingAttributeMapper.of(rule).checkForDuplicateLabels(attribute);
    for (Label label : duplicates) {
      rule.reportError(
          String.format("Label '%s' is duplicated in the '%s' attribute of rule '%s'",
          label, attribute.getName(), rule.getName()), eventHandler);
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
      Package.Builder pkgBuilder) {
    // TODO(b/149505729): Determine the right semantics for someone trying to define their own
    // attribute named applicable_licenses.
    if (attr.getName().equals("applicable_licenses")) {
      return pkgBuilder.getDefaultApplicableLicenses();
    }
    // Starlark rules may define their own "licenses" attributes with different types -
    // we shouldn't trigger the special "licenses" on those cases.
    if (attr.getName().equals("licenses") && attr.getType() == BuildType.LICENSE) {
      return pkgBuilder.getDefaultLicense();
    }
    if (attr.getName().equals("distribs")) {
      return pkgBuilder.getDefaultDistribs();
    }
    return attr.getDefaultValue(null);
  }

  /**
   * Sets the value of attribute {@code attr} in {@code rule} to the native value {@code
   * nativeAttrVal}, and sets the value's explicitness to {@code explicit}.
   *
   * <p>Handles the special case of the "visibility" attribute by also setting the rule's
   * visibility with {@link Rule#setVisibility}.
   *
   * <p>Checks that {@code nativeAttrVal} is an allowed value via {@link #checkAllowedValues}.
   */
  private static void setRuleAttributeValue(
      Rule rule,
      EventHandler eventHandler,
      Attribute attr,
      Object nativeAttrVal,
      boolean explicit) {
    if (attr.getName().equals("visibility")) {
      @SuppressWarnings("unchecked")
      List<Label> attrList = (List<Label>) nativeAttrVal;
      if (!attrList.isEmpty()
          && ConstantRuleVisibility.LEGACY_PUBLIC_LABEL.equals(attrList.get(0))) {
        rule.reportError(
            rule.getLabel() + ": //visibility:legacy_public only allowed in package declaration",
            eventHandler);
      }
      try {
        rule.setVisibility(PackageUtils.getVisibility(rule.getLabel(), attrList));
      } catch (EvalException e) {
         rule.reportError(rule.getLabel() + " " + e.getMessage(), eventHandler);
      }
    }
    rule.setAttributeValue(attr, nativeAttrVal, explicit);
    checkAllowedValues(rule, attr, eventHandler);
  }

  /**
   * Converts the build-language-typed {@code buildLangValue} to a native value via {@link
   * BuildType#selectableConvert}. Canonicalizes the value's order if it is a {@link List} type and
   * {@code attr.isOrderIndependent()} returns {@code true}.
   *
   * <p>Throws {@link ConversionException} if the conversion fails, or if {@code buildLangValue} is
   * a selector expression but {@code attr.isConfigurable()} is {@code false}.
   */
  private static Object convertFromBuildLangType(
      Rule rule,
      Attribute attr,
      Object buildLangValue,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      Interner<ImmutableList<?>> listInterner)
      throws ConversionException {
    LabelConversionContext context = new LabelConversionContext(rule.getLabel(), repositoryMapping);
    Object converted =
        BuildType.selectableConvert(
            attr.getType(),
            buildLangValue,
            new AttributeConversionContext(attr.getName(), rule.getRuleClass()),
            context);

    if ((converted instanceof SelectorList<?>) && !attr.isConfigurable()) {
      throw new ConversionException(
          String.format("attribute \"%s\" is not configurable", attr.getName()));
    }

    if (converted instanceof List<?>) {
      if (attr.isOrderIndependent()) {
        @SuppressWarnings("unchecked")
        List<? extends Comparable<?>> list = (List<? extends Comparable<?>>) converted;
        converted = Ordering.natural().sortedCopy(list);
      }
      // It's common for multiple rule instances in the same package to have the same value for some
      // attributes. As a concrete example, consider a package having several 'java_test' instances,
      // each with the same exact 'tags' attribute value.
      converted = listInterner.intern(ImmutableList.copyOf((List<?>) converted));
    }

    return converted;
  }

  /**
   * Provides a {@link #toString()} description of the attribute being converted for
   * {@link BuildType#selectableConvert}. This is preferred over a raw string to avoid uselessly
   * constructing strings which are never used. A separate class instead of inline to avoid
   * accidental memory leaks.
   */
  private static class AttributeConversionContext {
    private final String attrName;
    private final String ruleClass;

    AttributeConversionContext(String attrName, String ruleClass) {
      this.attrName = attrName;
      this.ruleClass = ruleClass;
    }

    @Override
    public String toString() {
      return "attribute '" + attrName + "' in '" + ruleClass + "' rule";
    }
  }


  /**
   * Verifies that the rule has a valid value for the attribute according to its allowed values.
   *
   * <p>If the value for the given attribute on the given rule is invalid, an error will be recorded
   * in the given EventHandler.
   *
   * <p>If the rule is configurable, all of its potential values are evaluated, and errors for each
   * of the invalid values are reported.
   */
  private static void checkAllowedValues(
      Rule rule, Attribute attribute, EventHandler eventHandler) {
    if (attribute.checkAllowedValues()) {
      PredicateWithMessage<Object> allowedValues = attribute.getAllowedValues();
      Iterable<?> values =
          AggregatingAttributeMapper.of(rule).visitAttribute(
              attribute.getName(), attribute.getType());
      for (Object value : values) {
        if (!allowedValues.apply(value)) {
          rule.reportError(
              String.format(
                  "%s: invalid value in '%s' attribute: %s",
                  rule.getLabel(),
                  attribute.getName(),
                  allowedValues.getErrorReason(value)),
              eventHandler);
        }
      }
    }
  }

  private static void checkAspectAllowedValues(
      Rule rule, EventHandler eventHandler) {
    if (rule.hasAspects()) {
      for (Attribute attrOfRule : rule.getAttributes()) {
        for (Aspect aspect : attrOfRule.getAspects(rule)) {
          for (Attribute attrOfAspect : aspect.getDefinition().getAttributes().values()) {
            // By this point the AspectDefinition has been created and values assigned.
            if (attrOfAspect.checkAllowedValues()) {
              PredicateWithMessage<Object> allowedValues = attrOfAspect.getAllowedValues();
              Object value = attrOfAspect.getDefaultValue(rule);
              if (!allowedValues.apply(value)) {
                if (RawAttributeMapper.of(rule).isConfigurable(attrOfAspect.getName())) {
                  rule.reportError(
                      String.format(
                          "%s: attribute '%s' has a select() and aspect %s also declares "
                              + "'%s'. Aspect attributes don't currently support select().",
                          rule.getLabel(),
                          attrOfAspect.getName(),
                          aspect.getDefinition().getName(),
                          rule.getLabel()),
                      eventHandler);
                } else {
                  rule.reportError(
                      String.format(
                          "%s: invalid value in '%s' attribute: %s",
                          rule.getLabel(),
                          attrOfAspect.getName(),
                          allowedValues.getErrorReason(value)),
                      eventHandler);
                }
              }
            }
          }
        }
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

  /** Returns this RuleClass's custom Skylark rule implementation. */
  @Nullable
  public StarlarkFunction getConfiguredTargetFunction() {
    return configuredTargetFunction;
  }

  @Nullable
  public BuildSetting getBuildSetting() {
    return buildSetting;
  }

  /**
   * Returns a function that computes the external bindings a repository function contributes to
   * the WORKSPACE file.
   */
  public Function<? super Rule, Map<String, Label>> getExternalBindingsFunction() {
    return externalBindingsFunction;
  }

  /**
   * Returns a function that computes the options referenced by a rule.
   */
  public Function<? super Rule, ? extends Set<String>> getOptionReferenceFunction() {
    return optionReferenceFunction;
  }

  /**
   * For Skylark rule classes, returns this RuleClass's rule definition environment's label, which
   * is never null. Is null for native rules' RuleClass objects.
   */
  @Nullable
  public Label getRuleDefinitionEnvironmentLabel() {
    return ruleDefinitionEnvironmentLabel;
  }

  /**
   * Returns the hash code for the RuleClass's rule definition environment. Will be null for native
   * rules' RuleClass objects.
   */
  @Nullable
  public String getRuleDefinitionEnvironmentHashCode() {
    return ruleDefinitionEnvironmentHashCode;
  }

  /** Returns true if this RuleClass is a Skylark-defined RuleClass. */
  public boolean isSkylark() {
    return isSkylark;
  }

  /**
   * Returns true if this RuleClass is Skylark-defined and is subject to analysis-time
   * tests.
   */
  public boolean isSkylarkTestable() {
    return skylarkTestable;
  }

  /**
   * Returns true if this rule class outputs a default executable for every rule.
   */
  public boolean isExecutableSkylark() {
    return isExecutableSkylark;
  }

  /** Returns true if this rule class is an analysis test (set by analysis_test = true). */
  public boolean isAnalysisTest() {
    return isAnalysisTest;
  }

  /**
   * Returns true if this rule class has at least one attribute with an analysis test transition. (A
   * starlark-defined transition using analysis_test_transition()).
   */
  public boolean hasAnalysisTestTransition() {
    return hasAnalysisTestTransition;
  }

  /**
   * Returns true if this rule class has the _whitelist_function_transition attribute.
   */
  public boolean hasFunctionTransitionWhitelist() {
    return hasFunctionTransitionWhitelist;
  }

  /**
   * If true, no rule of this class ever declares a license regardless of what the rule's or
   * package's <code>licenses</code> attribute says.
   *
   * <p>This is useful for rule types that don't make sense for license checking.
   */
  public boolean ignoreLicenses() {
    return ignoreLicenses;
  }

  public ImmutableSet<Label> getRequiredToolchains() {
    return requiredToolchains;
  }

  public boolean useToolchainResolution() {
    return useToolchainResolution;
  }

  public ImmutableSet<Label> getExecutionPlatformConstraints() {
    return executionPlatformConstraints;
  }

  public OutputFile.Kind  getOutputFileKind() {
    return outputFileKind;
  }

  public static boolean isThirdPartyPackage(PackageIdentifier packageIdentifier) {
    if (!packageIdentifier.getRepository().isMain()) {
      return false;
    }

    if (!packageIdentifier.getPackageFragment().startsWith(THIRD_PARTY_PREFIX)) {
      return false;
    }

    if (packageIdentifier.getPackageFragment().segmentCount() <= 1) {
      return false;
    }

    return true;
  }
}
