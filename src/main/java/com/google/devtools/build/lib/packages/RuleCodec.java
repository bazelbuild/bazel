// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;

/**
 * Codec for {@link Rule} that throws. We expect never to serialize Rule except for in PackageCodec,
 * which has custom logic.
 */
public class RuleCodec implements ObjectCodec<Rule> {

  @VisibleForTesting
  static final String SERIALIZATION_ERROR_TEMPLATE =
      "Rule serialization is not permitted outside of PackageCodec, but attempted to serialize "
          + "Rule %s.";

  private static final String DESERIALIZATION_ERROR_TEMPLATE =
      "Rule deserialization is not permitted outside of PackageCodec, but attempted to deserialize "
          + "a rule";

  @Override
  public Class<? extends Rule> getEncodedClass() {
    return Rule.class;
  }

  @Override
  public void serialize(SerializationContext context, Rule obj, CodedOutputStream codedOut)
      throws SerializationException {
    throw new SerializationException(String.format(SERIALIZATION_ERROR_TEMPLATE, obj));
  }

  @Override
  public Rule deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException {
    throw new SerializationException(DESERIALIZATION_ERROR_TEMPLATE);
  }
}
