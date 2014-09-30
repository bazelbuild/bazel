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

import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.Type.FILESET_ENTRY_LIST;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.INTEGER_LIST;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST_DICT;
import static com.google.devtools.build.lib.packages.Type.LICENSE;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.Type.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.OUTPUT;
import static com.google.devtools.build.lib.packages.Type.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT;
import static com.google.devtools.build.lib.packages.Type.STRING_DICT_UNARY;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST_DICT;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;

import java.util.Map;

/**
 * Shared code used in proto buffer output for rules and rule classes.
 */
public class ProtoUtils {
  /**
   * This map contains all attribute types that are recognized by the protocol
   * output formatter.
   */
  private static final Map<Type<?>, Discriminator> TYPE_MAP
      = new ImmutableMap.Builder<Type<?>, Discriminator>()
          .put(INTEGER, Discriminator.INTEGER)
          .put(DISTRIBUTIONS, Discriminator.DISTRIBUTION_SET)
          .put(LABEL, Discriminator.LABEL)
          // NODEP_LABEL attributes are not really strings. This is implemented
          // this way for the sake of backward compatibility.
          .put(NODEP_LABEL, Discriminator.STRING)
          .put(LABEL_LIST, Discriminator.LABEL_LIST)
          .put(NODEP_LABEL_LIST, Discriminator.STRING_LIST)
          .put(STRING, Discriminator.STRING)
          .put(STRING_LIST, Discriminator.STRING_LIST)
          .put(OUTPUT, Discriminator.OUTPUT)
          .put(OUTPUT_LIST, Discriminator.OUTPUT_LIST)
          .put(LICENSE, Discriminator.LICENSE)
          .put(STRING_DICT, Discriminator.STRING_DICT)
          .put(FILESET_ENTRY_LIST, Discriminator.FILESET_ENTRY_LIST)
          .put(LABEL_LIST_DICT, Discriminator.LABEL_LIST_DICT)
          .put(STRING_LIST_DICT, Discriminator.STRING_LIST_DICT)
          .put(BOOLEAN, Discriminator.BOOLEAN)
          .put(TRISTATE, Discriminator.TRISTATE)
          .put(INTEGER_LIST, Discriminator.INTEGER_LIST)
          .put(STRING_DICT_UNARY, Discriminator.STRING_DICT_UNARY)
          .build();

  /**
   * Returns the appropriate Attribute.Discriminator value from an internal attribute type.
   */
  public static Discriminator getDiscriminatorFromType(Type<?> type) {
    Preconditions.checkArgument(TYPE_MAP.containsKey(type));
    return TYPE_MAP.get(type);
  }
}
