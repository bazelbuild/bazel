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
package com.google.devtools.common.options;

import static java.util.stream.Collectors.joining;

import com.google.common.base.Verify;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.flogger.GoogleLogger;
import com.google.common.flogger.LazyArgs;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.AllowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.DisallowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.FlagPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.FlagPolicy.OperationCase;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.UseDefault;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser.OptionDescription;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * Enforces the {@link FlagPolicy}s (from an {@link InvocationPolicy} proto) on an {@link
 * OptionsParser} by validating and changing the flag values in the given {@link OptionsParser}.
 *
 * <p>"Flag" and "Option" are used interchangeably in this file.
 */
public final class InvocationPolicyEnforcer {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final String INVOCATION_POLICY_SOURCE = "Invocation policy";
  @Nullable private final InvocationPolicy invocationPolicy;
  private final Level loglevel;

  /**
   * Creates an InvocationPolicyEnforcer that enforces the given policy.
   *
   * @param invocationPolicy the policy to enforce. A null policy means this enforcer will do
   *     nothing in calls to enforce().
   */
  public InvocationPolicyEnforcer(@Nullable InvocationPolicy invocationPolicy) {
    this(invocationPolicy, Level.FINE);
  }

  /**
   * Creates an InvocationPolicyEnforcer that enforces the given policy.
   *
   * @param invocationPolicy the policy to enforce. A null policy means this enforcer will do
   *     nothing in calls to enforce().
   * @param loglevel the level at which to log informational statements. Warnings and errors will
   *     still be logged at the appropriate level.
   */
  public InvocationPolicyEnforcer(@Nullable InvocationPolicy invocationPolicy, Level loglevel) {
    this.invocationPolicy = invocationPolicy;
    this.loglevel = loglevel;
  }

  private static final class FlagPolicyWithContext {
    private final FlagPolicy policy;
    private final OptionDescription description;
    private final OptionInstanceOrigin origin;

    public FlagPolicyWithContext(
        FlagPolicy policy, OptionDescription description, OptionInstanceOrigin origin) {
      this.policy = policy;
      this.description = description;
      this.origin = origin;
    }
  }

  public InvocationPolicy getInvocationPolicy() {
    return invocationPolicy;
  }

  /**
   * Applies this OptionsPolicyEnforcer's policy to the given OptionsParser for all blaze commands.
   *
   * @param parser The OptionsParser to enforce policy on.
   * @throws OptionsParsingException if any flag policy is invalid.
   */
  public void enforce(OptionsParser parser) throws OptionsParsingException {
    enforce(parser, null);
  }

  /**
   * Applies this OptionsPolicyEnforcer's policy to the given OptionsParser.
   *
   * @param parser The OptionsParser to enforce policy on.
   * @param command The current blaze command, for flag policies that apply to only specific
   *     commands. Such policies will be enforced only if they contain this command or a command
   *     they inherit from
   * @throws OptionsParsingException if any flag policy is invalid.
   */
  public void enforce(OptionsParser parser, @Nullable String command)
      throws OptionsParsingException {
    if (invocationPolicy == null || invocationPolicy.getFlagPoliciesCount() == 0) {
      return;
    }

    // The effective policy returned is expanded, filtered for applicable commands, and cleaned of
    // redundancies and conflicts.
    List<FlagPolicyWithContext> effectivePolicies =
        getEffectivePolicies(invocationPolicy, parser, command, loglevel);

    for (FlagPolicyWithContext flagPolicy : effectivePolicies) {
      String flagName = flagPolicy.policy.getFlagName();

      OptionValueDescription valueDescription;
      try {
        valueDescription = parser.getOptionValueDescription(flagName);
      } catch (IllegalArgumentException e) {
        // This flag doesn't exist. We are deliberately lenient if the flag policy has a flag
        // we don't know about. This is for better future proofing so that as new flags are added,
        // new policies can use the new flags without worrying about older versions of Bazel.
        logger.at(loglevel).log(
            "Flag '%s' specified by invocation policy does not exist", flagName);
        continue;
      }

      // getOptionDescription() will return null if the option does not exist, however
      // getOptionValueDescription() above would have thrown an IllegalArgumentException if that
      // were the case.
      Verify.verifyNotNull(flagPolicy.description);

      switch (flagPolicy.policy.getOperationCase()) {
        case SET_VALUE:
          applySetValueOperation(parser, flagPolicy, valueDescription, loglevel);
          break;

        case USE_DEFAULT:
          applyUseDefaultOperation(
              parser, "UseDefault", flagPolicy.description.getOptionDefinition(), loglevel);
          break;

        case ALLOW_VALUES:
          AllowValues allowValues = flagPolicy.policy.getAllowValues();
          FilterValueOperation.AllowValueOperation allowValueOperation =
              new FilterValueOperation.AllowValueOperation(loglevel);
          allowValueOperation.apply(
              parser,
              flagPolicy.origin,
              allowValues.getAllowedValuesList(),
              allowValues.hasNewValue() ? allowValues.getNewValue() : null,
              allowValues.hasUseDefault(),
              valueDescription,
              flagPolicy.description);
          break;

        case DISALLOW_VALUES:
          DisallowValues disallowValues = flagPolicy.policy.getDisallowValues();
          FilterValueOperation.DisallowValueOperation disallowValueOperation =
              new FilterValueOperation.DisallowValueOperation(loglevel);
          disallowValueOperation.apply(
              parser,
              flagPolicy.origin,
              disallowValues.getDisallowedValuesList(),
              disallowValues.hasNewValue() ? disallowValues.getNewValue() : null,
              disallowValues.hasUseDefault(),
              valueDescription,
              flagPolicy.description);
          break;

        case OPERATION_NOT_SET:
          throw new PolicyOperationNotSetException(flagName);

        default:
          logger.atWarning().log(
              "Unknown operation '%s' from invocation policy for flag '%s'",
              flagPolicy.policy.getOperationCase(), flagName);
          break;
      }
    }
  }

  private static class PolicyOperationNotSetException extends OptionsParsingException {
    PolicyOperationNotSetException(String flagName) {
      super(String.format("Flag policy for flag '%s' does not " + "have an operation", flagName));
    }
  }

  private static boolean policyApplies(FlagPolicy policy, ImmutableSet<String> applicableCommands) {
    // Skip the flag policy if it doesn't apply to this command. If the commands list is empty,
    // then the policy applies to all commands.
    if (policy.getCommandsList().isEmpty() || applicableCommands.isEmpty()) {
      return true;
    }

    return !Collections.disjoint(policy.getCommandsList(), applicableCommands);
  }

  /** Returns the expanded and filtered policy that would be enforced for the given command. */
  public static InvocationPolicy getEffectiveInvocationPolicy(
      InvocationPolicy invocationPolicy, OptionsParser parser, String command, Level loglevel)
      throws OptionsParsingException {
    ImmutableList<FlagPolicyWithContext> effectivePolicies =
        getEffectivePolicies(invocationPolicy, parser, command, loglevel);

    InvocationPolicy.Builder builder = InvocationPolicy.newBuilder();
    for (FlagPolicyWithContext policyWithContext : effectivePolicies) {
      builder.addFlagPolicies(policyWithContext.policy);
    }
    return builder.build();
  }

  /**
   * Takes the provided policy and processes it to the form that can be used on the user options.
   *
   * <p>Expands any policies on expansion flags.
   */
  private static ImmutableList<FlagPolicyWithContext> getEffectivePolicies(
      InvocationPolicy invocationPolicy, OptionsParser parser, String command, Level loglevel)
      throws OptionsParsingException {
    if (invocationPolicy == null) {
      return ImmutableList.of();
    }

    ImmutableSet<String> commandAndParentCommands =
        command == null
            ? ImmutableSet.of()
            : CommandNameCache.CommandNameCacheInstance.INSTANCE.get(command);

    // Expand all policies to transfer policies on expansion flags to policies on the child flags.
    List<FlagPolicyWithContext> expandedPolicies = new ArrayList<>();
    OptionPriority nextPriority =
        OptionPriority.lowestOptionPriorityAtCategory(PriorityCategory.INVOCATION_POLICY);
    for (FlagPolicy policy : invocationPolicy.getFlagPoliciesList()) {
      // Explicitly disallow --config in invocation policy.
      if (policy.getFlagName().equals("config")) {
        throw new OptionsParsingException(
            "Invocation policy is applied after --config expansion, changing config values now "
                + "would have no effect and is disallowed to prevent confusion. Please remove the "
                + "following policy : "
                + policy);
      }

      // These policies are high-level, before expansion, and so are not the implicitDependents or
      // expansions of any other flag, other than in an obtuse sense from --invocation_policy.
      OptionPriority currentPriority = nextPriority;
      OptionInstanceOrigin origin =
          new OptionInstanceOrigin(currentPriority, INVOCATION_POLICY_SOURCE, null, null);
      nextPriority = OptionPriority.nextOptionPriority(currentPriority);
      if (!policyApplies(policy, commandAndParentCommands)) {
        // Only keep and expand policies that are applicable to the current command.
        continue;
      }

      OptionDescription optionDescription = parser.getOptionDescription(policy.getFlagName());
      if (optionDescription == null) {
        // InvocationPolicy ignores policy on non-existing flags by design, for version
        // compatibility.
        logger.at(loglevel).log(
            "Flag '%s' specified by invocation policy does not exist, and will be ignored",
            policy.getFlagName());
        continue;
      }
      FlagPolicyWithContext policyWithContext =
          new FlagPolicyWithContext(policy, optionDescription, origin);
      List<FlagPolicyWithContext> policies = expandPolicy(policyWithContext, parser, loglevel);
      expandedPolicies.addAll(policies);
    }

    // Only keep that last policy for each flag.
    Map<String, FlagPolicyWithContext> effectivePolicy = new HashMap<>();
    for (FlagPolicyWithContext expandedPolicy : expandedPolicies) {
      String flagName = expandedPolicy.policy.getFlagName();
      effectivePolicy.put(flagName, expandedPolicy);
    }

    return ImmutableList.copyOf(effectivePolicy.values());
  }

  private static void throwAllowValuesOnExpansionFlagException(String flagName)
      throws OptionsParsingException {
    throw new OptionsParsingException(
        String.format("Allow_Values on expansion flags like %s is not allowed.", flagName));
  }

  private static void throwDisallowValuesOnExpansionFlagException(String flagName)
      throws OptionsParsingException {
    throw new OptionsParsingException(
        String.format("Disallow_Values on expansion flags like %s is not allowed.", flagName));
  }

  /**
   * Expand a single policy. If the policy is not about an expansion flag, this will simply return a
   * list with a single element, oneself. If the policy is for an expansion flag, the policy will
   * get split into multiple policies applying to each flag the original flag expands to.
   *
   * <p>None of the flagPolicies returned should be on expansion flags.
   */
  private static ImmutableList<FlagPolicyWithContext> expandPolicy(
      FlagPolicyWithContext originalPolicy, OptionsParser parser, Level loglevel)
      throws OptionsParsingException {
    ImmutableList.Builder<FlagPolicyWithContext> expandedPolicies = ImmutableList.builder();

    boolean isExpansion = originalPolicy.description.isExpansion();
    ImmutableList<ParsedOptionDescription> subflags =
        parser.getExpansionValueDescriptions(
            originalPolicy.description.getOptionDefinition(), originalPolicy.origin);

    // If we have nothing to expand to, no need to do any further work.
    if (subflags.isEmpty()) {
      return ImmutableList.of(originalPolicy);
    }

    // Log the expansion. This is only really useful for understanding the invocation policy itself.
    logger.at(loglevel).log(
        "Expanding %s on option %s to its %s: %s.",
        originalPolicy.policy.getOperationCase(),
        originalPolicy.policy.getFlagName(),
        isExpansion ? "expansions" : "implied flags",
        LazyArgs.lazy(
            () ->
                subflags.stream()
                    .map(f -> "--" + f.getOptionDefinition().getOptionName())
                    .collect(joining("; "))));

    // Repeated flags are special, and could set multiple times in an expansion, with the user
    // expecting both values to be valid. Collect these separately.
    Multimap<OptionDescription, ParsedOptionDescription> repeatableSubflagsInSetValues =
        ArrayListMultimap.create();

    // Create a flag policy for the child that looks like the parent's policy "transferred" to its
    // child. Note that this only makes sense for SetValue, when setting an expansion flag, or
    // UseDefault, when preventing it from being set.
    for (ParsedOptionDescription currentSubflag : subflags) {
      OptionDescription subflagOptionDescription =
          parser.getOptionDescription(currentSubflag.getOptionDefinition().getOptionName());

      if (currentSubflag.getOptionDefinition().allowsMultiple()
          && originalPolicy.policy.getOperationCase().equals(OperationCase.SET_VALUE)) {
        repeatableSubflagsInSetValues.put(subflagOptionDescription, currentSubflag);
      } else {
        FlagPolicyWithContext subflagAsPolicy =
            getSingleValueSubflagAsPolicy(
                subflagOptionDescription, currentSubflag, originalPolicy, isExpansion);
        // In case any of the expanded flags are themselves expansions, recurse.
        expandedPolicies.addAll(expandPolicy(subflagAsPolicy, parser, loglevel));
      }
    }

    // If there are any repeatable flag SetValues, deal with them together now.
    // Note that expansion flags have no value, and so cannot have multiple values either.
    // Skipping the recursion above is fine.
    for (OptionDescription repeatableFlag : repeatableSubflagsInSetValues.keySet()) {
      int numValues = repeatableSubflagsInSetValues.get(repeatableFlag).size();
      ArrayList<String> newValues = new ArrayList<>(numValues);
      ArrayList<OptionInstanceOrigin> origins = new ArrayList<>(numValues);
      for (ParsedOptionDescription setValue : repeatableSubflagsInSetValues.get(repeatableFlag)) {
        newValues.add(setValue.getUnconvertedValue());
        origins.add(setValue.getOrigin());
      }
      // These options come from expanding a single policy, so they have effectively the same
      // priority. They could have come from different expansions or implicit requirements in the
      // recursive resolving of the option list, so just pick the first one. Do collapse the source
      // strings though, in case there are different sources.
      OptionInstanceOrigin arbitraryFirstOptionOrigin = origins.get(0);
      OptionInstanceOrigin originOfSubflags =
          new OptionInstanceOrigin(
              arbitraryFirstOptionOrigin.getPriority(),
              origins.stream()
                  .map(OptionInstanceOrigin::getSource)
                  .distinct()
                  .collect(joining(", ")),
              arbitraryFirstOptionOrigin.getImplicitDependent(),
              arbitraryFirstOptionOrigin.getExpandedFrom());
      expandedPolicies.add(
          getSetValueSubflagAsPolicy(repeatableFlag, newValues, originOfSubflags, originalPolicy));
    }

    // Don't add the original policy if it was an expansion flag, which have no value, but do add
    // it if there was either no expansion or if it was a valued flag with implicit requirements.
    if (!isExpansion) {
      expandedPolicies.add(originalPolicy);
    }

    return expandedPolicies.build();
  }

  /**
   * Expand a SetValue flag policy on a repeatable flag. SetValue operations are the only flag
   * policies that set the flag, and so interact with repeatable flags, flags that can be set
   * multiple times, in subtle ways.
   *
   * @param subflagDesc, the description of the flag the SetValue'd expansion flag expands to.
   * @param subflagValue, the values that the SetValue'd expansion flag expands to for this flag.
   * @param originalPolicy, the original policy on the expansion flag.
   * @return the flag policy for the subflag given, this will be part of the expanded form of the
   *     SetValue policy on the original flag.
   */
  private static FlagPolicyWithContext getSetValueSubflagAsPolicy(
      OptionDescription subflagDesc,
      List<String> subflagValue,
      OptionInstanceOrigin subflagOrigin,
      FlagPolicyWithContext originalPolicy) {
    // Some checks.
    OptionDefinition subflag = subflagDesc.getOptionDefinition();
    Verify.verify(originalPolicy.policy.getOperationCase().equals(OperationCase.SET_VALUE));
    if (!subflag.allowsMultiple()) {
      Verify.verify(subflagValue.size() <= 1);
    }

    // Flag value from the expansion, overridability from the original policy, unless the flag is
    // repeatable, in which case we care about appendability, not overridability.
    SetValue.Builder setValueExpansion = SetValue.newBuilder();
    for (String value : subflagValue) {
      setValueExpansion.addFlagValue(value);
    }
    if (subflag.allowsMultiple()) {
      setValueExpansion.setAppend(originalPolicy.policy.getSetValue().getOverridable());
    } else {
      setValueExpansion.setOverridable(originalPolicy.policy.getSetValue().getOverridable());
    }

    // Commands from the original policy, flag name of the expansion
    return new FlagPolicyWithContext(
        FlagPolicy.newBuilder()
            .addAllCommands(originalPolicy.policy.getCommandsList())
            .setFlagName(subflag.getOptionName())
            .setSetValue(setValueExpansion)
            .build(),
        subflagDesc,
        subflagOrigin);
  }

  /**
   * For an expansion flag in an invocation policy, each flag it expands to must be given a
   * corresponding policy.
   */
  private static FlagPolicyWithContext getSingleValueSubflagAsPolicy(
      OptionDescription subflagContext,
      ParsedOptionDescription currentSubflag,
      FlagPolicyWithContext originalPolicy,
      boolean isExpansion)
      throws OptionsParsingException {
    FlagPolicyWithContext subflagAsPolicy = null;
    switch (originalPolicy.policy.getOperationCase()) {
      case SET_VALUE:
        if (currentSubflag.getOptionDefinition().allowsMultiple()) {
          throw new AssertionError(
              "SetValue subflags with allowMultiple should have been dealt with separately and "
                  + "accumulated into a single FlagPolicy.");
        }
        // Accept null originalValueStrings, they are expected when the subflag is also an expansion
        // flag.
        List<String> subflagValue;
        if (currentSubflag.getUnconvertedValue() == null) {
          subflagValue = ImmutableList.of();
        } else {
          subflagValue = ImmutableList.of(currentSubflag.getUnconvertedValue());
        }
        subflagAsPolicy =
            getSetValueSubflagAsPolicy(
                subflagContext, subflagValue, currentSubflag.getOrigin(), originalPolicy);
        break;

      case USE_DEFAULT:
        // Commands from the original policy, flag name of the expansion
        subflagAsPolicy =
            new FlagPolicyWithContext(
                FlagPolicy.newBuilder()
                    .addAllCommands(originalPolicy.policy.getCommandsList())
                    .setFlagName(currentSubflag.getOptionDefinition().getOptionName())
                    .setUseDefault(UseDefault.getDefaultInstance())
                    .build(),
                subflagContext,
                currentSubflag.getOrigin());
        break;

      case ALLOW_VALUES:
        if (isExpansion) {
          throwAllowValuesOnExpansionFlagException(originalPolicy.policy.getFlagName());
        }
        // If this flag is an implicitRequirement, and some values for the parent flag are
        // allowed, nothing needs to happen on the implicitRequirement that is set for all
        // values of the flag.
        break;

      case DISALLOW_VALUES:
        if (isExpansion) {
          throwDisallowValuesOnExpansionFlagException(originalPolicy.policy.getFlagName());
        }
        // If this flag is an implicitRequirement, and some values for the parent flag are
        // disallowed, that implies that all others are allowed, so nothing needs to happen
        // on the implicitRequirement that is set for all values of the parent flag.
        break;

      case OPERATION_NOT_SET:
        throw new PolicyOperationNotSetException(originalPolicy.policy.getFlagName());

      default:
        return null;
    }
    return subflagAsPolicy;
  }

  private static void applySetValueOperation(
      OptionsParser parser,
      FlagPolicyWithContext flagPolicy,
      OptionValueDescription valueDescription,
      Level loglevel)
      throws OptionsParsingException {
    SetValue setValue = flagPolicy.policy.getSetValue();
    OptionDefinition optionDefinition = flagPolicy.description.getOptionDefinition();

    // SetValue.flag_value must have at least 1 value.
    if (setValue.getFlagValueCount() == 0) {
      throw new OptionsParsingException(
          String.format(
              "SetValue operation from invocation policy for %s does not have a value",
              optionDefinition));
    }

    // Flag must allow multiple values if multiple values are specified by the policy.
    if (setValue.getFlagValueCount() > 1
        && !flagPolicy.description.getOptionDefinition().allowsMultiple()) {
      throw new OptionsParsingException(
          String.format(
              "SetValue operation from invocation policy sets multiple values for %s which "
                  + "does not allow multiple values",
              optionDefinition));
    }

    if (setValue.getOverridable() && valueDescription != null) {
      // The user set the value for the flag but the flag policy is overridable, so keep the user's
      // value.
      logger.at(loglevel).log(
          "Keeping value '%s' from source '%s' for %s because the invocation policy specifying "
              + "the value(s) '%s' is overridable",
          valueDescription.getValue(),
          valueDescription.getSourceString(),
          optionDefinition,
          setValue.getFlagValueList());
    } else {

      if (!setValue.getAppend()) {
        // Clear the value in case the flag is a repeated flag so that values don't accumulate.
        parser.clearValue(flagPolicy.description.getOptionDefinition());
      }

      // Set all the flag values from the policy.
      for (String flagValue : setValue.getFlagValueList()) {
        if (valueDescription == null) {
          logger.at(loglevel).log(
              "Setting value for %s from invocation policy to '%s', overriding the default value "
                  + "'%s'",
              optionDefinition, flagValue, optionDefinition.getDefaultValue());
        } else {
          logger.at(loglevel).log(
              "Setting value for %s from invocation policy to '%s', overriding value '%s' from "
                  + "'%s'",
              optionDefinition,
              flagValue,
              valueDescription.getValue(),
              valueDescription.getSourceString());
        }

        parser.addOptionValueAtSpecificPriority(flagPolicy.origin, optionDefinition, flagValue);
      }
    }
  }

  private static void applyUseDefaultOperation(
      OptionsParser parser, String policyType, OptionDefinition option, Level loglevel)
      throws OptionsParsingException {
    OptionValueDescription clearedValueDescription = parser.clearValue(option);
    if (clearedValueDescription != null) {
      // Log the removed value.
      String clearedFlagName = clearedValueDescription.getOptionDefinition().getOptionName();
      Object clearedFlagDefaultValue =
          clearedValueDescription.getOptionDefinition().getDefaultValue();
      logger.at(loglevel).log(
          "Using default value '%s' for flag '%s' as specified by %s invocation policy, "
              + "overriding original value '%s' from '%s'",
          clearedFlagDefaultValue,
          clearedFlagName,
          policyType,
          clearedValueDescription.getValue(),
          clearedValueDescription.getSourceString());
    }
  }

  /** Checks the user's flag values against a filtering function. */
  private abstract static class FilterValueOperation {

    private static final class AllowValueOperation extends FilterValueOperation {
      AllowValueOperation(Level loglevel) {
        super("Allow", loglevel);
      }

      @Override
      boolean isFlagValueAllowed(Set<Object> convertedPolicyValues, Object value) {
        return convertedPolicyValues.contains(value);
      }
    }

    private static final class DisallowValueOperation extends FilterValueOperation {
      DisallowValueOperation(Level loglevel) {
        super("Disalllow", loglevel);
      }

      @Override
      boolean isFlagValueAllowed(Set<Object> convertedPolicyValues, Object value) {
        // In a disallow operation, the values that the flag policy specifies are not allowed,
        // so the value is allowed if the set of policy values does not contain the current
        // flag value.
        return !convertedPolicyValues.contains(value);
      }
    }

    private final String policyType;
    private final Level loglevel;

    FilterValueOperation(String policyType, Level loglevel) {
      this.policyType = policyType;
      this.loglevel = loglevel;
    }

    /**
     * Determines if the given value is allowed.
     *
     * @param convertedPolicyValues The values given from the FlagPolicy, converted to real objects.
     * @param value The user value of the flag.
     * @return True if the value should be allowed, false if it should not.
     */
    abstract boolean isFlagValueAllowed(Set<Object> convertedPolicyValues, Object value);

    void apply(
        OptionsParser parser,
        OptionInstanceOrigin origin,
        List<String> policyValues,
        String newValue,
        boolean useDefault,
        OptionValueDescription valueDescription,
        OptionDescription optionDescription)
        throws OptionsParsingException {
      OptionDefinition optionDefinition = optionDescription.getOptionDefinition();
      // Convert all the allowed values from strings to real objects using the options'
      // converters so that they can be checked for equality using real .equals() instead
      // of string comparison. For example, "--foo=0", "--foo=false", "--nofoo", and "-f-"
      // (if the option has an abbreviation) are all equal for boolean flags. Plus converters
      // can be arbitrarily complex.
      Set<Object> convertedPolicyValues = new HashSet<>();
      for (String value : policyValues) {
        Object convertedValue = optionDefinition.getConverter().convert(value);
        // Some converters return lists, and if the flag is a repeatable flag, the items in the
        // list from the converter should be added, and not the list itself. Otherwise the items
        // from invocation policy will be compared to lists, which will never work.
        // See OptionsParserImpl.ParsedOptionEntry.addValue.
        if (optionDefinition.allowsMultiple() && convertedValue instanceof List<?>) {
          convertedPolicyValues.addAll((List<?>) convertedValue);
        } else {
          convertedPolicyValues.add(optionDefinition.getConverter().convert(value));
        }
      }

      // Check that if the default value of the flag is disallowed by the policy, that the policy
      // does not also set use_default. Otherwise the default value would still be set if the
      // user uses a disallowed value. This doesn't apply to repeatable flags since the default
      // value for repeatable flags is always the empty list. It also doesn't apply to flags that
      // are null by default, since these flags' default value is not parsed by the converter, so
      // there is no guarantee that there exists an accepted user-input value that would also set
      // the value to NULL. In these cases, we assume that "unset" is a distinct value that is
      // always allowed.
      if (!optionDescription.getOptionDefinition().allowsMultiple()
          && !optionDescription.getOptionDefinition().isSpecialNullDefault()) {
        boolean defaultValueAllowed =
            isFlagValueAllowed(
                convertedPolicyValues, optionDescription.getOptionDefinition().getDefaultValue());
        if (!defaultValueAllowed && useDefault) {
          throw new OptionsParsingException(
              String.format(
                  "%sValues policy disallows the default value '%s' for %s but also specifies to "
                      + "use the default value",
                  policyType, optionDefinition.getDefaultValue(), optionDefinition));
        }
      }

      if (valueDescription == null) {
        // Nothing has set the value yet, so check that the default value from the flag's
        // definition is allowed. The else case below (i.e. valueDescription is not null) checks for
        // the flag allowing multiple values, however, flags that allow multiple values cannot have
        // default values, and their value is always the empty list if they haven't been specified,
        // which is why new_default_value is not a repeated field.
        checkDefaultValue(
            parser, origin, optionDescription, policyValues, newValue, convertedPolicyValues);
      } else {
        checkUserValue(
            parser,
            origin,
            optionDescription,
            valueDescription,
            policyValues,
            newValue,
            useDefault,
            convertedPolicyValues);
      }
    }

    void checkDefaultValue(
        OptionsParser parser,
        OptionInstanceOrigin origin,
        OptionDescription optionDescription,
        List<String> policyValues,
        String newValue,
        Set<Object> convertedPolicyValues)
        throws OptionsParsingException {

      OptionDefinition optionDefinition = optionDescription.getOptionDefinition();
      if (optionDefinition.isSpecialNullDefault()) {
        // Do nothing, the unset value by definition cannot be set. In option filtering operations,
        // the value is being filtered, but the value that is `no value` passes any filter.
        // Otherwise, there is no way to "usedefault" on one of these options that has no value by
        // default.
      } else if (!isFlagValueAllowed(convertedPolicyValues, optionDefinition.getDefaultValue())) {
        if (newValue != null) {
          // Use the default value from the policy, since the original default is not allowed
          logger.at(loglevel).log(
              "Overriding default value '%s' for %s with value '%s' specified by invocation "
                  + "policy. %sed values are: %s",
              optionDefinition.getDefaultValue(),
              optionDefinition,
              newValue,
              policyType,
              policyValues);
          parser.clearValue(optionDefinition);
          parser.addOptionValueAtSpecificPriority(origin, optionDefinition, newValue);
        } else {
          // The operation disallows the default value, but doesn't supply a new value.
          throw new OptionsParsingException(
              String.format(
                  "Default flag value '%s' for %s is not allowed by invocation policy, but "
                      + "the policy does not provide a new value. %sed values are: %s",
                  optionDescription.getOptionDefinition().getDefaultValue(),
                  optionDefinition,
                  policyType,
                  policyValues));
        }
      }
    }

    void checkUserValue(
        OptionsParser parser,
        OptionInstanceOrigin origin,
        OptionDescription optionDescription,
        OptionValueDescription valueDescription,
        List<String> policyValues,
        String newValue,
        boolean useDefault,
        Set<Object> convertedPolicyValues)
        throws OptionsParsingException {
      OptionDefinition option = optionDescription.getOptionDefinition();
      if (optionDescription.getOptionDefinition().allowsMultiple()) {
        // allowMultiple requires that the type of the option be List<T>, so cast from Object
        // to List<?>.
        List<?> optionValues = (List<?>) valueDescription.getValue();
        for (Object value : optionValues) {
          if (!isFlagValueAllowed(convertedPolicyValues, value)) {
            if (useDefault) {
              applyUseDefaultOperation(parser, policyType + "Values", option, loglevel);
            } else {
              throw new OptionsParsingException(
                  String.format(
                      "Flag value '%s' for %s is not allowed by invocation policy. %sed values "
                          + "are: %s",
                      value, option, policyType, policyValues));
            }
          }
        }

      } else {

        if (!isFlagValueAllowed(convertedPolicyValues, valueDescription.getValue())) {
          if (newValue != null) {
            logger.at(loglevel).log(
                "Overriding disallowed value '%s' for %s with value '%s' "
                    + "specified by invocation policy. %sed values are: %s",
                valueDescription.getValue(), option, newValue, policyType, policyValues);
            parser.clearValue(option);
            parser.addOptionValueAtSpecificPriority(origin, option, newValue);
          } else if (useDefault) {
            applyUseDefaultOperation(parser, policyType + "Values", option, loglevel);
          } else {
            throw new OptionsParsingException(
                String.format(
                    "Flag value '%s' for %s is not allowed by invocation policy and the "
                        + "policy does not specify a new value. %sed values are: %s",
                    valueDescription.getValue(), option, policyType, policyValues));
          }
        }
      }
    }
  }
}
