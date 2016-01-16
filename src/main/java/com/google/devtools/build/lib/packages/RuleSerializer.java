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
    boolean isSkylark = rule.getRuleClassObject().isSkylark();

    for (Attribute attr : rule.getAttributes()) {
      Object rawAttributeValue = rawAttributeMapper.getRawAttributeValue(rule, attr);
      boolean isExplicit = rule.isAttributeValueExplicitlySpecified(attr);

      if (!isSkylark && !isExplicit) {
        // If the rule class is native (i.e. not Skylark-defined), then we can skip serialization
        // of implicit attribute values. The native rule class can provide the same default value
        // for the attribute after deserialization.
        continue;
      }

      Object valueToSerialize;
      if (isExplicit) {
        valueToSerialize = rawAttributeValue;
      } else if (rawAttributeValue instanceof ComputedDefault) {
        // If the rule class is Skylark-defined (i.e. rule.getRuleClassObject().isSkylark() is
        // true), and the attribute has a ComputedDefault value, then we must serialize what it
        // evaluates to. The Skylark-defined ComputedDefault function won't be available after
        // deserialization due to Skylark's non-serializability.
        valueToSerialize = evaluateSkylarkComputedDefault(rule, rawAttributeMapper, attr);
      } else {
        // If the rule class is Skylark-defined and the attribute value is implicit, then we
        // must serialize it. The Skylark-defined rule class won't be available after
        // deserialization due to Skylark's non-serializability.
        valueToSerialize = rawAttributeValue;
      }

      builder.addAttribute(
          AttributeSerializer.getAttributeProto(
              attr,
              valueToSerialize,
              isExplicit,
              /*includeGlobs=*/ true,
              /*encodeBooleanAndTriStateAsIntegerAndString=*/ false));
    }
    return builder;
  }

  /**
   * Evaluates a {@link ComputedDefault} attribute value for a {@link Rule} with a
   * Skylark-defined {@link RuleClass}.
   *
   * <p>Fortunately (from the perspective of rule serialization), Skylark doesn't support defining
   * rule classes with {@link ComputedDefault} attributes, and so the only {@link
   * ComputedDefault} attributes we need to worry about for Skylark-defined rule classes are
   * declared in those rule classes' natively-defined base rule classes.
   *
   * <p>See the comment for {@link #SKYLARK_RULE_CLASS_COMPUTED_DEFAULT_ATTRIBUTES} for the
   * locations of these expected attributes. None of them have dependencies on other attributes
   * which are configurable, so they can be evaluated here without loss of fidelity.
   *
   * <p>The {@link RawAttributeMapper#get} method, inherited from {@link
   * AbstractAttributeMapper}, evaluates the {@link ComputedDefault} function, so we use that,
   * after verifying the attribute's name is expected.
   */
  private static Object evaluateSkylarkComputedDefault(
      Rule rule, RawAttributeMapper rawAttributeMapper, Attribute attr) {
    Preconditions.checkState(
        SKYLARK_RULE_CLASS_COMPUTED_DEFAULT_ATTRIBUTES.contains(attr.getName()),
        "Unexpected ComputedDefault value for %s in %s",
        attr,
        rule);
    return rawAttributeMapper.get(attr.getName(), attr.getType());
  }
}

