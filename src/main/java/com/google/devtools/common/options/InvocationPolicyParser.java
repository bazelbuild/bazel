// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.CharMatcher;
import com.google.common.base.Strings;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.TextFormat;

/**
 * Parses the given InvocationPolicy string, which may be a base64-encoded binary-serialized
 * InvocationPolicy message, or a text formatted InvocationPolicy message. Note that the text
 * format is not backwards compatible as the binary format is.
 */
public class InvocationPolicyParser {
  /**
   * Parses InvocationPolicy in either of the accepted formats. Returns an empty policy if no policy
   * is provided.
   *
   * @throws com.google.devtools.common.options.OptionsParsingException if the value of
   *     --invocation_policy is invalid.
   */
  public static InvocationPolicy parsePolicy(String policy) throws OptionsParsingException {
    if (Strings.isNullOrEmpty(policy)) {
      return InvocationPolicy.getDefaultInstance();
    }

    try {
      try {
        // First try decoding the policy as a base64 encoded binary proto.
        return InvocationPolicy.parseFrom(
            BaseEncoding.base64().decode(CharMatcher.whitespace().removeFrom(policy)));
      } catch (IllegalArgumentException e) {
        // If the flag value can't be decoded from base64, try decoding the policy as a text
        // formatted proto.
        return TextFormat.parse(policy, InvocationPolicy.class);
      }
    } catch (InvalidProtocolBufferException | TextFormat.ParseException e) {
      throw new OptionsParsingException("Malformed value of --invocation_policy: " + policy, e);
    }
  }
}
