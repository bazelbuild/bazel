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

package com.google.devtools.build.lib.bazel.rules.java;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BuildInfo;
import com.google.devtools.build.lib.rules.java.BuildInfoPropertiesTranslator;
import com.google.devtools.build.lib.rules.java.GenericBuildInfoPropertiesTranslator;
import com.google.devtools.build.lib.rules.java.JavaBuildInfoFactory;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * BuildInfoFactory for Java.
 */
public class BazelJavaBuildInfoFactory extends JavaBuildInfoFactory {
  @AutoCodec @AutoCodec.VisibleForSerialization
  static final GenericBuildInfoPropertiesTranslator VOLATILE_KEYS =
      new GenericBuildInfoPropertiesTranslator(
          ImmutableMap.<String, String>builder()
              .put("build.time", "%BUILD_TIME%")
              .put("build.timestamp.as.int", "%BUILD_TIMESTAMP%")
              .put("build.timestamp", "%BUILD_TIMESTAMP%")
              .build());

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final GenericBuildInfoPropertiesTranslator NONVOLATILE_KEYS =
      new GenericBuildInfoPropertiesTranslator(
          ImmutableMap.of("build.label", "%" + BuildInfo.BUILD_EMBED_LABEL + "|%"));

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final GenericBuildInfoPropertiesTranslator REDACTED_KEYS =
      new GenericBuildInfoPropertiesTranslator(
          ImmutableMap.<String, String>builder()
              .put("build.time", "Thu Jan 01 00:00:00 1970 (0)")
              .put("build.timestamp.as.int", "0")
              .put("build.timestamp", "Thu Jan 01 00:00:00 1970 (0)")
              .build());

  @Override
  protected BuildInfoPropertiesTranslator createVolatileTranslator() {
    return VOLATILE_KEYS;
  }

  @Override
  protected BuildInfoPropertiesTranslator createNonVolatileTranslator() {
    return NONVOLATILE_KEYS;
  }

  @Override
  protected BuildInfoPropertiesTranslator createRedactedTranslator() {
    return REDACTED_KEYS;
  }
}
