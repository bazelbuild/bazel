// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.options;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import java.util.List;
import java.util.regex.Pattern;

/** Options for remote execution and distributed caching that shared between Bazel and Blaze. */
public class CommonRemoteOptions extends OptionsBase {
  @Option(
      name = "experimental_remote_download_regex",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Force Bazel to download the artifacts that match the given regexp. To be used in"
              + " conjunction with Build without the Bytes (or the internal equivalent) to allow"
              + " the client to request certain artifacts that might be needed locally (e.g. IDE"
              + " support)")
  public List<String> remoteDownloadRegex;

  /** Returns the specified duration. Assumes seconds if unitless. */
  public static class RemoteDurationConverter extends Converter.Contextless<Duration> {

    private static final Pattern UNITLESS_REGEX = Pattern.compile("^[0-9]+$");

    @Override
    public Duration convert(String input) throws OptionsParsingException {
      if (UNITLESS_REGEX.matcher(input).matches()) {
        input += "s";
      }
      return new Converters.DurationConverter().convert(input, /* conversionContext= */ null);
    }

    @Override
    public String getTypeDescription() {
      return "An immutable length of time.";
    }
  }
}
