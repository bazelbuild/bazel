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
package com.google.devtools.build.lib.flags;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.AllowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.DisallowValues;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.FlagPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.SetValue;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParser.OptionDescription;
import com.google.devtools.common.options.OptionsParser.OptionValueDescription;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Enforces the {@link FlagPolicy}s (from an {@link InvocationPolicy} proto) on an
 * {@link OptionsParser} by validating and changing the flag values in the given
 * {@link OptionsParser}.
 *
 * <p>"Flag" and "Option" are used interchangeably in this file.
 */
public final class InvocationPolicyEnforcer {

  private static final Logger log = Logger.getLogger(InvocationPolicyEnforcer.class.getName());

  private static final Function<Object, String> INVOCATION_POLICY_SOURCE =
      Functions.constant("Invocation policy");

  @Nullable private final InvocationPolicy invocationPolicy;

  /**
   * Creates an InvocationPolicyEnforcer that enforces the given policy.
   *
   * @param invocationPolicy the policy to enforce. A null policy means this enforcer will do
   *     nothing in calls to enforce().
   */
  public InvocationPolicyEnforcer(@Nullable InvocationPolicy invocationPolicy) {
    this.invocationPolicy = invocationPolicy;
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

    ImmutableSet<String> commandAndParentCommands =
        command == null
            ? ImmutableSet.<String>of()
            : CommandNameCache.CommandNameCacheInstance.INSTANCE.get(command);
    for (FlagPolicy flagPolicy : invocationPolicy.getFlagPoliciesList()) {
      String flagName = flagPolicy.getFlagName();

      // Skip the flag policy if it doesn't apply to this command. If the commands list is empty,
      // then the policy applies to all commands.
      if (!flagPolicy.getCommandsList().isEmpty() && !commandAndParentCommands.isEmpty()) {
        boolean flagApplies = false;
        for (String policyCommand : flagPolicy.getCommandsList()) {
          if (commandAndParentCommands.contains(policyCommand)) {
            flagApplies = true;
            break;
          }
        }
        if (!flagApplies) {
          continue;
        }
      }

      OptionValueDescription valueDescription;
      try {
        valueDescription = parser.getOptionValueDescription(flagName);
      } catch (IllegalArgumentException e) {
        // This flag doesn't exist. We are deliberately lenient if the flag policy has a flag
        // we don't know about. This is for better future proofing so that as new flags are added,
        // new policies can use the new flags without worrying about older versions of Bazel.
        log.info(
            String.format("Flag '%s' specified by invocation policy does not exist", flagName));
        continue;
      }

      OptionDescription optionDescription = parser.getOptionDescription(flagName);
      // getOptionDescription() will return null if the option does not exist, however
      // getOptionValueDescription() above would have thrown an IllegalArgumentException if that
      // were the case.
      Verify.verifyNotNull(optionDescription);

      switch (flagPolicy.getOperationCase()) {
        case SET_VALUE:
          applySetValueOperation(parser, flagPolicy, flagName, valueDescription, optionDescription);
          break;

        case USE_DEFAULT:
          applyUseDefaultOperation(parser, "UseDefault", flagName);
          break;

        case ALLOW_VALUES:
          AllowValues allowValues = flagPolicy.getAllowValues();
          FilterValueOperation.ALLOW_VALUE_OPERATION.apply(
              parser,
              allowValues.getAllowedValuesList(),
              allowValues.hasNewValue() ? allowValues.getNewValue() : null,
              allowValues.hasUseDefault(),
              flagName,
              valueDescription,
              optionDescription);
          break;

        case DISALLOW_VALUES:
          DisallowValues disallowValues = flagPolicy.getDisallowValues();
          FilterValueOperation.DISALLOW_VALUE_OPERATION.apply(
              parser,
              disallowValues.getDisallowedValuesList(),
              disallowValues.hasNewValue() ? disallowValues.getNewValue() : null,
              disallowValues.hasUseDefault(),
              flagName,
              valueDescription,
              optionDescription);
          break;

        case OPERATION_NOT_SET:
          throw new OptionsParsingException(
              String.format("Flag policy for flag '%s' does not " + "have an operation", flagName));

        default:
          log.warning(
              String.format(
                  "Unknown operation '%s' from invocation policy for flag '%s'",
                  flagPolicy.getOperationCase(),
                  flagName));
          break;
      }
    }
  }

  private static void logInApplySetValueOperation(String formattingString, Object... objects) {
    // Finding the caller here is relatively expensive and shows up in profiling, so provide it
    // manually.
    log.logp(
        Level.INFO,
        "InvocationPolicyEnforcer",
        "applySetValueOperation",
        String.format(formattingString, objects));
  }

  private static void applySetValueOperation(
      OptionsParser parser,
      FlagPolicy flagPolicy,
      String flagName,
      OptionValueDescription valueDescription,
      OptionDescription optionDescription)
      throws OptionsParsingException {

    SetValue setValue = flagPolicy.getSetValue();

    // SetValue.flag_value must have at least 1 value.
    if (setValue.getFlagValueCount() == 0) {
      throw new OptionsParsingException(
          String.format(
              "SetValue operation from invocation policy for flag '%s' does not have a value",
              flagName));
    }

    // Flag must allow multiple values if multiple values are specified by the policy.
    if (setValue.getFlagValueCount() > 1 && !optionDescription.getAllowMultiple()) {
      throw new OptionsParsingException(
          String.format(
              "SetValue operation from invocation policy sets multiple values for flag '%s' which "
                  + "does not allow multiple values",
              flagName));
    }

    if (setValue.getOverridable() && valueDescription != null) {
      // The user set the value for the flag but the flag policy is overridable, so keep the user's
      // value.
      logInApplySetValueOperation(
          "Keeping value '%s' from source '%s' for flag '%s' "
              + "because the invocation policy specifying the value(s) '%s' is overridable",
          valueDescription.getValue(),
          valueDescription.getSource(),
          flagName,
          setValue.getFlagValueList());
    } else {

      if (!setValue.getAppend()) {
        // Clear the value in case the flag is a repeated flag so that values don't accumulate.
        parser.clearValue(flagName);
      }

      // Set all the flag values from the policy.
      for (String flagValue : setValue.getFlagValueList()) {
        if (valueDescription == null) {
          logInApplySetValueOperation(
              "Setting value for flag '%s' from invocation "
                  + "policy to '%s', overriding the default value '%s'",
              flagName, flagValue, optionDescription.getDefaultValue());
        } else {
          logInApplySetValueOperation(
              "Setting value for flag '%s' from invocation "
                  + "policy to '%s', overriding value '%s' from '%s'",
              flagName, flagValue, valueDescription.getValue(), valueDescription.getSource());
        }
        setFlagValue(parser, flagName, flagValue);
      }
    }
  }

  private static void applyUseDefaultOperation(
      OptionsParser parser, String policyType, String flagName) throws OptionsParsingException {

    Map<String, OptionValueDescription> clearedValues = parser.clearValue(flagName);
    for (Entry<String, OptionValueDescription> clearedValue : clearedValues.entrySet()) {

      OptionValueDescription clearedValueDescription = clearedValue.getValue();
      String clearedFlagName = clearedValue.getKey();
      String originalValue = clearedValueDescription.getValue().toString();
      String source = clearedValueDescription.getSource();

      Object clearedFlagDefaultValue =
          parser.getOptionDescription(clearedFlagName).getDefaultValue();

      log.info(
          String.format(
              "Using default value '%s' for flag '%s' as "
                  + "specified by %s invocation policy, overriding original value '%s' from '%s'",
              clearedFlagDefaultValue,
              clearedFlagName,
              policyType,
              originalValue,
              source));
    }
  }

  /**
   * Checks the user's flag values against a filtering function.
   */
  private abstract static class FilterValueOperation {

    private static final FilterValueOperation ALLOW_VALUE_OPERATION =
        new FilterValueOperation("Allow") {
          @Override
          boolean isFlagValueAllowed(Set<Object> convertedPolicyValues, Object value) {
            return convertedPolicyValues.contains(value);
          }
        };

    private static final FilterValueOperation DISALLOW_VALUE_OPERATION =
        new FilterValueOperation("Disallow") {
          @Override
          boolean isFlagValueAllowed(Set<Object> convertedPolicyValues, Object value) {
            // In a disallow operation, the values that the flag policy specifies are not allowed,
            // so the value is allowed if the set of policy values does not contain the current
            // flag value.
            return !convertedPolicyValues.contains(value);
          }
        };

    private final String policyType;

    FilterValueOperation(String policyType) {
      this.policyType = policyType;
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
        List<String> policyValues,
        String newValue,
        boolean useDefault,
        String flagName,
        OptionValueDescription valueDescription,
        OptionDescription optionDescription)
        throws OptionsParsingException {

      // Convert all the allowed values from strings to real objects using the options'
      // converters so that they can be checked for equality using real .equals() instead
      // of string comparison. For example, "--foo=0", "--foo=false", "--nofoo", and "-f-"
      // (if the option has an abbreviation) are all equal for boolean flags. Plus converters
      // can be arbitrarily complex.
      Set<Object> convertedPolicyValues = Sets.newHashSet();
      for (String value : policyValues) {
        Object convertedValue = optionDescription.getConverter().convert(value);
        // Some converters return lists, and if the flag is a repeatable flag, the items in the
        // list from the converter should be added, and not the list itself. Otherwise the items
        // from invocation policy will be compared to lists, which will never work.
        // See OptionsParserImpl.ParsedOptionEntry.addValue.
        if (optionDescription.getAllowMultiple() && convertedValue instanceof List<?>) {
          convertedPolicyValues.addAll((List<?>) convertedValue);
        } else {
          convertedPolicyValues.add(optionDescription.getConverter().convert(value));
        }
      }

      // Check that if the default value of the flag is disallowed by the policy, that the policy
      // does not also set use_default. Otherwise the default value would will still be set if the
      // user uses a disallowed value. This doesn't apply to repeatable flags since the default
      // value for repeatable flags is always the empty list.
      if (!optionDescription.getAllowMultiple()) {

        boolean defaultValueAllowed =
            isFlagValueAllowed(convertedPolicyValues, optionDescription.getDefaultValue());
        if (!defaultValueAllowed && useDefault) {
          throw new OptionsParsingException(
              String.format(
                  "%sValues policy disallows the default value '%s' for flag '%s' but also "
                      + "specifies to use the default value",
                  policyType,
                  optionDescription.getDefaultValue(),
                  flagName));
        }
      }

      if (valueDescription == null) {
        // Nothing has set the value yet, so check that the default value from the flag's
        // definition is allowed. The else case below (i.e. valueDescription is not null) checks for
        // the flag allowing multiple values, however, flags that allow multiple values cannot have
        // default values, and their value is always the empty list if they haven't been specified,
        // which is why new_default_value is not a repeated field.
        checkDefaultValue(
            parser,
            policyValues,
            newValue,
            flagName,
            optionDescription,
            convertedPolicyValues);
      } else {
        checkUserValue(
            parser,
            policyValues,
            newValue,
            useDefault,
            flagName,
            valueDescription,
            optionDescription,
            convertedPolicyValues);
      }
    }

    void checkDefaultValue(
        OptionsParser parser,
        List<String> policyValues,
        String newValue,
        String flagName,
        OptionDescription optionDescription,
        Set<Object> convertedPolicyValues)
        throws OptionsParsingException {

      if (!isFlagValueAllowed(convertedPolicyValues, optionDescription.getDefaultValue())) {
        if (newValue != null) {
          // Use the default value from the policy.
          log.info(
              String.format(
                  "Overriding default value '%s' for flag '%s' with value '%s' "
                      + "specified by invocation policy. %sed values are: %s",
                  optionDescription.getDefaultValue(),
                  flagName,
                  newValue,
                  policyType,
                  policyValues));
          parser.clearValue(flagName);
          setFlagValue(parser, flagName, newValue);
        } else {
          // The operation disallows the default value, but doesn't supply a new value.
          throw new OptionsParsingException(
              String.format(
                  "Default flag value '%s' for flag '%s' is not allowed by invocation policy, but "
                      + "the policy does not provide a new value. "
                      + "%sed values are: %s",
                  optionDescription.getDefaultValue(),
                  flagName,
                  policyType,
                  policyValues));
        }
      }
    }

    void checkUserValue(
        OptionsParser parser,
        List<String> policyValues,
        String newValue,
        boolean useDefault,
        String flagName,
        OptionValueDescription valueDescription,
        OptionDescription optionDescription,
        Set<Object> convertedPolicyValues)
        throws OptionsParsingException {
      
      if (optionDescription.getAllowMultiple()) {
        // allowMultiple requires that the type of the option be List<T>, so cast from Object
        // to List<?>.
        List<?> optionValues = (List<?>) valueDescription.getValue();
        for (Object value : optionValues) {
          if (!isFlagValueAllowed(convertedPolicyValues, value)) {
            if (useDefault) {
              applyUseDefaultOperation(parser, policyType + "Values", flagName);
            } else {
              throw new OptionsParsingException(
                  String.format(
                      "Flag value '%s' for flag '%s' is not allowed by invocation policy. "
                          + "%sed values are: %s",
                      value,
                      flagName,
                      policyType,
                      policyValues));
            }
          }
        }

      } else {

        if (!isFlagValueAllowed(convertedPolicyValues, valueDescription.getValue())) {
          if (newValue != null) {
            log.info(
                String.format(
                    "Overriding disallowed value '%s' for flag '%s' with value '%s' "
                        + "specified by invocation policy. %sed values are: %s",
                    valueDescription.getValue(),
                    flagName,
                    newValue,
                    policyType,
                    policyValues));
            parser.clearValue(flagName);
            setFlagValue(parser, flagName, newValue);
          } else if (useDefault) {
            applyUseDefaultOperation(parser, policyType + "Values", flagName);
          } else {
            throw new OptionsParsingException(
                String.format(
                    "Flag value '%s' for flag '%s' is not allowed by invocation policy and the "
                        + "policy does not specify a new value. %sed values are: %s",
                    valueDescription.getValue(),
                    flagName,
                    policyType,
                    policyValues));
          }
        }
      }
    }
  }

  private static void setFlagValue(OptionsParser parser, String flagName, String flagValue)
      throws OptionsParsingException {

    parser.parseWithSourceFunction(
        OptionPriority.INVOCATION_POLICY,
        INVOCATION_POLICY_SOURCE,
        Arrays.asList(String.format("--%s=%s", flagName, flagValue)));
  }
}
