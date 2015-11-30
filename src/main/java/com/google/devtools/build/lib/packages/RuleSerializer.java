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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.query2.proto.proto2api.Build;

/** Serialize a {@link Rule} as its protobuf representation. */
public class RuleSerializer {

  public static Build.Rule.Builder serializeRule(Rule rule) {
    Build.Rule.Builder builder = Build.Rule.newBuilder();
    builder.setName(rule.getLabel().getName());
    builder.setRuleClass(rule.getRuleClass());
    builder.setPublicByDefault(rule.getRuleClassObject().isPublicByDefault());
    for (Attribute attribute : rule.getAttributes()) {
      builder.addAttribute(
          AttributeSerializer.getAttributeProto(
              attribute,
              AttributeSerializer.getAttributeValues(rule, attribute),
              rule.isAttributeValueExplicitlySpecified(attribute),
              /*includeGlobs=*/ true));
    }
    return builder;
  }
}

