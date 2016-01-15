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
package com.google.devtools.build.lib.runtime;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.common.io.BaseEncoding;
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
import com.google.devtools.common.options.OptionsProvider;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.TextFormat;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Given an OptionsParser and a InvocationPolicy proto, enforces the FlagPolicies on an
 * OptionsParser.
 *
 * <p>"Flag" and "Option" are used interchangeably in this file.
 */
public final class InvocationPolicyEnforcer {

  /**
   * Creates an {@link InvocationPolicyEnforcer} with the invocation policy obtained from the given
   * {@link OptionsProvider}. This uses the provider only to obtain the policy from the
   * --invocation_policy flag and does not enforce any policy on the flags in the provider.
   *
   * @param startupOptionsProvider an options provider which provides a BlazeServerStartupOptions
   *     options class
   *
   * @throws OptionsParsingException if the value of --invocation_policy is invalid
   */
  public static InvocationPolicyEnforcer create(OptionsProvider startupOptionsProvider)
      throws OptionsParsingException {

    BlazeServerStartupOptions blazeServerStartupOptions =
        startupOptionsProvider.getOptions(BlazeServerStartupOptions.class);
    return new InvocationPolicyEnforcer(parsePolicy(blazeServerStartupOptions.invocationPolicy));
  }

  public static InvocationPolicyEnforcer create(String invocationPolicy)
      throws OptionsParsingException {

    return new InvocationPolicyEnforcer(parsePolicy(invocationPolicy));
  }

  /**
   * Parses the given InvocationPolicy string, which may be a base64-encoded binary-serialized
   * InvocationPolicy message, or a text formatted InvocationPolicy message. Note that the
   * text format is not backwards compatible as the binary format is, and the option to
   * provide a text formatted proto is provided only for debugging.
   *
   * @throws OptionsParsingException if the value of --invocation_policy is invalid
   */
  private static InvocationPolicy parsePolicy(String policy) throws OptionsParsingException {
    if (policy == null || policy.isEmpty()) {
      return null;
    }

    try {
      try {
        // First try decoding the policy as a base64 encoded binary proto.
        return InvocationPolicy.parseFrom(
            BaseEncoding.base64().decode(CharMatcher.WHITESPACE.removeFrom(policy)));
      } catch (IllegalArgumentException e) {
        // If the flag value can't be decoded from base64, try decoding the policy as a text
        // formated proto.
        InvocationPolicy.Builder builder = InvocationPolicy.newBuilder();
        TextFormat.merge(policy, builder);
        return builder.build();
      }
    } catch (InvalidProtocolBufferException | TextFormat.ParseException e) {
      throw new OptionsParsingException("Malformed value of --invocation_policy: " + policy, e);
    }
  }

  private static final Logger LOG = Logger.getLogger(InvocationPolicyEnforcer.class.getName());
  
  @Nullable
  private final InvocationPolicy invocationPolicy;

  public InvocationPolicyEnforcer(@Nullable InvocationPolicy invocationPolicy) {
    this.invocationPolicy = invocationPolicy;
  }

  /**
   * Applies this OptionsPolicyEnforcer's policy to the given OptionsParser.
   *
   * @param parser The OptionsParser to enforce policy on.
   * @param command The command to which the options in the OptionsParser apply.
   * @throws OptionsParsingException
   */
  public void enforce(OptionsParser parser, String command) throws OptionsParsingException {
    if (invocationPolicy == null) {
      return;
    }

    if (invocationPolicy.getFlagPoliciesCount() == 0) {
      LOG.warning("InvocationPolicy contains no flag policies.");
    }

    Function<Object, String> sourceFunction = Functions.constant("Invocation policy");

    for (FlagPolicy flagPolicy : invocationPolicy.getFlagPoliciesList()) {
      String flagName = flagPolicy.getFlagName();

      // Skip the flag policy if it doesn't apply to this command.
      if (!flagPolicy.getCommandsList().isEmpty()
          && !flagPolicy.getCommandsList().contains(command)) {
        LOG.info(String.format("Skipping flag policy for flag '%s' because it "
            + "applies only to commands %s and the current command is '%s'",
            flagName, flagPolicy.getCommandsList(), command));
        continue;
      }

      OptionValueDescription valueDescription;
      try {
        valueDescription = parser.getOptionValueDescription(flagName);
      } catch (IllegalArgumentException e) {
        // This flag doesn't exist. We are deliberately lenient if the flag policy has a flag
        // we don't know about. This is for better future proofing so that as new flags are added,
        // new policies can use the new flags without worrying about older versions of Bazel. 
        LOG.info(String.format(
            "Flag '%s' specified by invocation policy does not exist", flagName));
        continue;
      }

      OptionDescription optionDescription = parser.getOptionDescription(flagName);
      // getOptionDescription() will return null if the option does not exist, however
      // getOptionValueDescription() above would have thrown an IllegalArgumentException if that
      // were the case.
      Verify.verifyNotNull(optionDescription);

      switch (flagPolicy.getOperationCase()) {
        case SET_VALUE:
          applySetValueOperation(parser, sourceFunction, flagPolicy, flagName,
              valueDescription, optionDescription);
          break;

        case USE_DEFAULT:
          applyUseDefaultOperation(parser, flagName);
          break;

        case ALLOW_VALUES:
          applyAllowValuesOperation(parser, sourceFunction, flagPolicy,
              flagName, valueDescription, optionDescription);
          break;

        case DISALLOW_VALUES:
          applyDisallowValuesOperation(parser, sourceFunction, flagPolicy,
              flagName, valueDescription, optionDescription);
          break;

        case OPERATION_NOT_SET:
          throw new OptionsParsingException(String.format("Flag policy for flag '%s' does not "
              + "have an operation", flagName));

        default:
          LOG.warning(String.format("Unknown operation '%s' from invocation policy for flag '%s'",
              flagPolicy.getOperationCase(), flagName));
          break;
      }
    }
  }

  private static void applySetValueOperation(
      OptionsParser parser,
      Function<Object, String> sourceFunction,
      FlagPolicy flagPolicy,
      String flagName,
      OptionValueDescription valueDescription,
      OptionDescription optionDescription) throws OptionsParsingException {

    SetValue setValue = flagPolicy.getSetValue();

    // SetValue.flag_value must have at least 1 value.
    if (setValue.getFlagValueCount() == 0) {
      throw new OptionsParsingException(String.format(
          "SetValue operation from invocation policy for flag '%s' does not have a value",
          flagName));
    }
  
    // Flag must allow multiple values if multiple values are specified by the policy.
    if (setValue.getFlagValueCount() > 1 && !optionDescription.getAllowMultiple()) {
      throw new OptionsParsingException(String.format(
          "SetValue operation from invocation policy sets multiple values for flag '%s' which "
          + "does not allow multiple values", flagName));
    }
  
    if (setValue.getOverridable() && valueDescription != null) {
      // The user set the value for the flag but the flag policy is overridable, so keep the user's
      // value.
      LOG.info(String.format("Keeping value '%s' from source '%s' for flag '%s' "
          + "because the invocation policy specifying the value(s) '%s' is overridable",
          valueDescription.getValue(), valueDescription.getSource(), flagName,
          setValue.getFlagValueList()));
    } else {
  
      // Clear the value in case the flag is a repeated flag (so that values don't accumulate), and
      // in case the flag is an expansion flag or has implicit flags (so that the additional flags
      // also get cleared).
      parser.clearValue(flagName);

      // Set all the flag values from the policy.
      for (String flagValue : setValue.getFlagValueList()) {
        if (valueDescription == null) {
          LOG.info(String.format("Setting value for flag '%s' from invocation "
              + "policy to '%s', overriding the default value '%s'", flagName, flagValue,
              optionDescription.getDefaultValue()));
        } else {
          LOG.info(String.format("Setting value for flag '%s' from invocation "
              + "policy to '%s', overriding value '%s' from '%s'", flagName, flagValue,
              valueDescription.getValue(), valueDescription.getSource()));
        }
        setFlagValue(parser, flagName, flagValue, sourceFunction);
      }
    }
  }

  private static void applyUseDefaultOperation(OptionsParser parser, String flagName) {

    Map<String, OptionValueDescription> clearedValues = parser.clearValue(flagName);
    for (Entry<String, OptionValueDescription> clearedValue : clearedValues.entrySet()) {
  
      OptionValueDescription clearedValueDesc = clearedValue.getValue();
      String clearedFlagName = clearedValue.getKey();
      String originalValue = clearedValueDesc.getValue().toString();
      String source = clearedValueDesc.getSource();
  
      OptionDescription clearedFlagDesc = parser.getOptionDescription(clearedFlagName);
      Object clearedFlagdefaultValue = clearedFlagDesc.getDefaultValue();
  
      LOG.info(String.format("Using default value '%s' for flag '%s' as "
          + "specified by invocation policy, overriding original value '%s' from '%s'",
          clearedFlagdefaultValue, clearedFlagName, originalValue, source));
    }
  }

  private static void applyAllowValuesOperation(
      OptionsParser parser,
      Function<Object, String> sourceFunction,
      FlagPolicy flagPolicy,
      String flagName,
      OptionValueDescription valueDescription,
      OptionDescription optionDescription) throws OptionsParsingException {

    AllowValues allowValues = flagPolicy.getAllowValues();
    applyAllowDisallowValueOperation(
        parser,
        sourceFunction,
        /*allowValues=*/ true,
        allowValues.getAllowedValuesList(),
        allowValues.hasNewDefaultValue() ? allowValues.getNewDefaultValue() : null,
        flagName,
        valueDescription,
        optionDescription);
  }
  
  private static void applyDisallowValuesOperation(
      OptionsParser parser,
      Function<Object, String> sourceFunction,
      FlagPolicy flagPolicy,
      String flagName,
      OptionValueDescription valueDescription,
      OptionDescription optionDescription) throws OptionsParsingException {

    DisallowValues disallowValues = flagPolicy.getDisallowValues();
    applyAllowDisallowValueOperation(
        parser,
        sourceFunction,
        /*allowValues=*/ false,
        disallowValues.getDisallowedValuesList(),
        disallowValues.hasNewDefaultValue() ? disallowValues.getNewDefaultValue() : null,
        flagName,
        valueDescription,
        optionDescription);
  }

  /**
   * Shared logic between AllowValues and DisallowValues operations.
   *
   * @param parser
   * @param sourceFunction
   * @param allowValues True if this is an AllowValues operation, false if DisallowValues
   * @param policyValues The list of allowed or disallowed values
   * @param newDefaultValue The new default to use if the default value for the flag is now allowed
   *   (i.e. not in the list of allowed values or in the list of disallowed values).
   * @param flagName
   * @param valueDescription
   * @param optionDescription
   *
   * @throws OptionsParsingException
   */
  private static void applyAllowDisallowValueOperation(
      OptionsParser parser,
      Function<Object, String> sourceFunction,
      boolean allowValues,
      List<String> policyValues,
      String newDefaultValue,
      String flagName,
      OptionValueDescription valueDescription,
      OptionDescription optionDescription) throws OptionsParsingException {

    // For error reporting.
    String policyType = allowValues ? "Allow" : "Disallow";
    
    // Convert all the allowed values from strings to real object using the option's
    // converter so that they can be checked for equality using real .equals() instead
    // of string comparison. For example, "--foo=0", "--foo=false", "--nofoo", and "-f-"
    // (if the option has an abbreviation) are all equal for boolean flags. Plus converters
    // can be arbitrarily complex.
    Set<Object> convertedPolicyValues = Sets.newHashSet();
    for (String value : policyValues) {
      convertedPolicyValues.add(optionDescription.getConverter().convert(value));
    }

    if (valueDescription == null) {
      // Nothing has set the value yet, so check that the default value from the flag's
      // definition is allowed. The else case below (i.e. valueDescription is not null) checks for
      // the flag allowing multiple values, however, flags that allow multiple values cannot have
      // default values, and their value is always the empty list if they haven't been specified,
      // which is why new_default_value is not a repeated field.
      //
      // This is xor'ed with allowValues because if the policy is to allow these values,
      // then we want to apply the new default (or throw an error) if the default value of the flag
      // is not in the set of allowed values. If the policy is to disallow these values
      // (allowValues is false), then we want to apply the new default (or throw an error) if
      // the default value of the flag is in the set of disallowed values. This works out to xor.
      if (allowValues ^ convertedPolicyValues.contains(optionDescription.getDefaultValue())) {
        if (newDefaultValue != null) {
          // Use the default value from the policy.
          LOG.info(String.format("Overriding default value '%s' for flag '%s' with "
              + "new default value '%s' specified by invocation policy. %sed values are: %s",
              optionDescription.getDefaultValue(), flagName, newDefaultValue,
              policyType, Joiner.on(", ").join(policyValues)));
          parser.clearValue(flagName);
          setFlagValue(parser, flagName, newDefaultValue, sourceFunction);
        } else {
          // The operation disallows the default value, but doesn't supply its own default.
          throw new OptionsParsingException(String.format(
              "Default flag value '%s' for flag '%s' is not allowed by invocation policy, but "
              + "the policy does not provide a new default value. "
              + "%sed values are: %s", optionDescription.getDefaultValue(), flagName,
              policyType, Joiner.on(", ").join(policyValues)));
        }
      }
    } else {
      // Check that the flag's value is allowed.
      List<?> values;
      if (optionDescription.getAllowMultiple()) {
        // allowMultiple requires that the type of the option be List<T>.
        values = (List<?>) valueDescription.getValue();
      } else {
        values = ImmutableList.of(valueDescription.getValue());
      }

      for (Object value : values) {
        // See above about the xor.
        if (allowValues ^ convertedPolicyValues.contains(value)) {
          throw new OptionsParsingException(String.format(
              "Flag value '%s' for flag '%s' is not allowed by invocation policy. "
              + "%sed values are: %s", value, flagName, policyType,
              Joiner.on(", ").join(policyValues)));
        }
      }
    }
  }

  private static void setFlagValue(
      OptionsParser parser,
      String flagName,
      String flagValue,
      Function<? super String, String> sourceFunction) throws OptionsParsingException {
 
    parser.parseWithSourceFunction(OptionPriority.INVOCATION_POLICY, sourceFunction,
        Arrays.asList(String.format("--%s=%s", flagName, flagValue)));
  }
}
