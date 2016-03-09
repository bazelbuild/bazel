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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;

import java.io.IOException;
import java.io.InputStream;

/**
 * A class to parse a {@link JavaToolchainData} from the result of blaze query. It is used by
 * {@link com.google.devtools.build.java.bazel.BazelJavaCompiler} to get default options.
 */
public class JavaToolchainDataParser {

  /**
   * Parse a {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult} as
   * returned by a bazel query and look for the list of target containing a {@code java_toolchain}
   * rule. These rules are then parsed into {@link JavaToolchainData}'s and returned as map with the
   * name of the target as key and the {@link JavaToolchainData} as value.
   */
  public static ImmutableMap<String, JavaToolchainData> parse(QueryResult queryResult) {
    ImmutableMap.Builder<String, JavaToolchainData> builder = ImmutableMap.builder();
    for (Build.Target target : queryResult.getTargetList()) {
      Build.Rule rule = target.getRule();
      if (target.hasRule() && rule.getRuleClass().equals("java_toolchain")) {
        builder.put(rule.getName(), parseBuildRuleProto(rule));
      }
    }
    return builder.build();
  }

  /**
   * Parse a text serialization of a
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult}. See
   * {@link #parse(QueryResult)} for the details of the result.
   *
   * @throws ParseException when the provided string does not corresponds to a bazel query output.
   */
  public static ImmutableMap<String, JavaToolchainData> parse(String queryResult)
      throws ParseException {
    QueryResult.Builder builder = QueryResult.newBuilder();
    TextFormat.merge(queryResult, builder);
    return parse(builder.build());
  }
  /**
   * Parse a {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult} from
   * an input stream. See {@link #parse(QueryResult)} for the details of the result.
   */
  public static ImmutableMap<String, JavaToolchainData> parse(InputStream queryResult)
      throws IOException {
    return parse(QueryResult.newBuilder().mergeFrom(queryResult).build());
  }

  private static JavaToolchainData parseBuildRuleProto(Build.Rule rule) {
    String source = "";
    String target = "";
    ImmutableList<String> bootclasspath = ImmutableList.of();
    ImmutableList<String> extclasspath = ImmutableList.of();
    String encoding = "";
    ImmutableList<String> xlint = ImmutableList.of();
    ImmutableList<String> misc = ImmutableList.of();
    ImmutableList<String> jvmOpts = ImmutableList.of();
    for (Build.Attribute attribute : rule.getAttributeList()) {
      switch (attribute.getName()) {
        case "source_version":
          source = attribute.getStringValue();
          break;
        case "target_version":
          target = attribute.getStringValue();
          break;
        case "bootclasspath":
          bootclasspath = ImmutableList.copyOf(attribute.getStringListValueList());
          break;
        case "extclasspath":
          extclasspath = ImmutableList.copyOf(attribute.getStringListValueList());
          break;
        case "encoding":
          encoding = attribute.getStringValue();
          break;
        case "xlint":
          xlint = ImmutableList.copyOf(attribute.getStringListValueList());
          break;
        case "misc":
          misc = ImmutableList.copyOf(attribute.getStringListValueList());
          break;
        case "jvm_opts":
          jvmOpts = ImmutableList.copyOf(attribute.getStringListValueList());
          break;
      }
    }
    return new JavaToolchainData(
        source, target, bootclasspath, extclasspath, encoding, xlint, misc, jvmOpts);
  }
}
