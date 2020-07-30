// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;

/**
 * Converts the {@code --apple_bitcode} command line option to a pair containing an optional
 * platform type and the bitcode mode to apply to builds targeting that platform.
 */
public final class AppleBitcodeConverter
    implements Converter<Map.Entry<ApplePlatform.PlatformType, AppleBitcodeMode>> {
  /** Used to convert Bitcode mode strings to their enum value. */
  private static final AppleBitcodeMode.Converter MODE_CONVERTER = new AppleBitcodeMode.Converter();

  /** Used to convert Apple platform type strings to their enum value. */
  private static final AppleCommandLineOptions.PlatformTypeConverter PLATFORM_TYPE_CONVERTER =
      new AppleCommandLineOptions.PlatformTypeConverter();

  private static final String TYPE_DESCRIPTION =
      String.format(
          "'mode' or 'platform=mode', where 'mode' is %s, and 'platform' is %s",
          MODE_CONVERTER.getTypeDescription(), PLATFORM_TYPE_CONVERTER.getTypeDescription());

  @VisibleForTesting
  public static final String INVALID_APPLE_BITCODE_OPTION_FORMAT =
      "Apple Bitcode mode must be in the form " + TYPE_DESCRIPTION;

  @Override
  public Map.Entry<ApplePlatform.PlatformType, AppleBitcodeMode> convert(String input)
      throws OptionsParsingException {
    ApplePlatform.PlatformType platformType;
    AppleBitcodeMode mode;

    int pos = input.indexOf('=');
    if (pos < 0) {
      // If there was no '=', then parse it as a Bitcode mode and apply it to all platforms (by
      // using a null key in the entry).
      platformType = null;
      mode = convertAppleBitcodeMode(input);
    } else {
      // If there was a '=', then parse the platform type from the left side, the Bitcode mode from
      // the right side, and apply it to just that platform.
      String platformTypeName = input.substring(0, pos);
      String modeName = input.substring(pos + 1);

      platformType = convertPlatformType(platformTypeName);
      mode = convertAppleBitcodeMode(modeName);
    }

    return Maps.immutableEntry(platformType, mode);
  }

  @Override
  public String getTypeDescription() {
    return TYPE_DESCRIPTION;
  }

  /**
   * Returns the {@code AppleBitcodeMode} value that is equivalent to the given string.
   *
   * @throws OptionsParsingException if the string was not a valid Apple Bitcode mode
   */
  private static AppleBitcodeMode convertAppleBitcodeMode(String input)
      throws OptionsParsingException {
    try {
      return MODE_CONVERTER.convert(input);
    } catch (OptionsParsingException e) {
      throw new OptionsParsingException(INVALID_APPLE_BITCODE_OPTION_FORMAT, e);
    }
  }

  /**
   * Returns the {@code ApplePlatform.PlatformType} value that is equivalent to the given string.
   *
   * @throws OptionsParsingException if any of the strings was not a valid Apple platform type
   */
  private static ApplePlatform.PlatformType convertPlatformType(String input)
      throws OptionsParsingException {
    try {
      return PLATFORM_TYPE_CONVERTER.convert(input);
    } catch (OptionsParsingException e) {
      throw new OptionsParsingException(INVALID_APPLE_BITCODE_OPTION_FORMAT, e);
    }
  }
}
