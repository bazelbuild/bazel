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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.util.Preconditions;

/** Serialize a {@link Rule} as its protobuf representation. */
public class RuleSerializer {

  // Skylark doesn't support defining rule classes with ComputedDefault attributes. Therefore, the
  // only ComputedDefault attributes we expect to see for Skylark-defined rule classes are
  // those declared in those rule classes' natively defined base rule classes, which are:
  //
  // 1. The "timeout" attribute in SkylarkRuleClassFunctions.testBaseRule
  // 2. The "deprecation" attribute in BaseRuleClasses.commonCoreAndSkylarkAttributes
  // 3. The "testonly" attribute in BaseRuleClasses.commonCoreAndSkylarkAttributes
  private static final ImmutableSet<String> SKYLARK_RULE_CLASS_COMPUTED_DEFAULT_ATTRIBUTES =
      ImmutableSet.of("timeout", "deprecation", "testonly");

  public static Build.Rule.Builder serializeRule(Rule rule) {
    Build.Rule.Builder builder = Build.Rule.newBuilder();
    builder.setName(rule.getLabel().getName());
    builder.setRuleClass(rule.getRuleClass());
    builder.setPublicByDefault(rule.getRuleClassObject().isPublicByDefault());
    RawAttributeMapper rawAttributeMapper = RawAttributeMapper.of(rule);
    for (Attribute attr : rule.getAttributes()) {
      Object rawAttributeValue = rawAttributeMapper.getRawAttributeValue(rule, attr);

      Object valueToSerialize;
      if (rawAttributeValue instanceof ComputedDefault) {
        if (rule.getRuleClassObject().isSkylark()) {
          // If the rule class is Skylark-defined (i.e. rule.getRuleClassObject().isSkylark() is
          // true), and the attribute has a ComputedDefault value, we must serialize it. The
          // Skylark-defined ComputedDefault function won't be available after deserialization due
          // to Skylark's non-serializability. Fortunately (from the perspective of rule
          // serialization), Skylark doesn't support defining rule classes with ComputedDefault
          // attributes, and so the only ComputedDefault attributes we need to worry about for
          // Skylark-defined rule classes are those declared in those rule classes' natively
          // defined base rule classes.
          //
          // See the comment for SKYLARK_RULE_CLASS_COMPUTED_DEFAULT_ATTRIBUTES for the locations
          // of these expected attributes.
          //
          // The RawAttributeMapper#get method, inherited from AbstractAttributeMapper, evaluates
          // the ComputedDefault function, so we use that, after verifying the attribute's name is
          // expected.
          Preconditions.checkState(
              SKYLARK_RULE_CLASS_COMPUTED_DEFAULT_ATTRIBUTES.contains(attr.getName()),
              "Unexpected ComputedDefault value for %s in %s",
              attr,
              rule);
          valueToSerialize = rawAttributeMapper.get(attr.getName(), attr.getType());
        } else {
          // If the rule class is native (i.e. not Skylark-defined), we can skip serialization of
          // attributes with ComputedDefault values. The native rule class can provide the same
          // ComputedDefault value for the attribute after deserialization.
          //
          // TODO(mschaller): While the native rule class *could* provide it, it doesn't yet. Make
          // it so! For now, we fall back to flattening the set of all possible values, computed
          // using AggregatingAttributeMapper.
          Iterable<Object> possibleValues =
              AggregatingAttributeMapper.of(rule).getPossibleAttributeValues(rule, attr);
          valueToSerialize =
              AggregatingAttributeMapper.flattenAttributeValues(attr.getType(), possibleValues);
        }
      } else {
        valueToSerialize = rawAttributeValue;
      }

      builder.addAttribute(
          AttributeSerializer.getAttributeProto(
              attr,
              valueToSerialize,
              rule.isAttributeValueExplicitlySpecified(attr),
              /*includeGlobs=*/ true,
              /*encodeBooleanAndTriStateAsIntegerAndString=*/ false));
    }
    return builder;
  }
}

