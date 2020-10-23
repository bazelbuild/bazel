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

package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;

/** Where the value of an attribute comes from. */
public enum AttributeValueSource {
  /** Explicitly specified on the rule. */
  RULE,

  /** Package default. */
  PACKAGE,

  /** Rule class default. */
  DEFAULT;

  public static AttributeValueSource forRuleAndAttribute(Rule rule, Attribute attr) {
    if (attr.getName().equals("visibility")) {
      if (rule.isVisibilitySpecified()) {
        return AttributeValueSource.RULE;
      } else if (rule.getPackage().isDefaultVisibilitySet()) {
        return AttributeValueSource.PACKAGE;
      } else {
        return AttributeValueSource.DEFAULT;
      }
    } else {
      return rule.isAttributeValueExplicitlySpecified(attr)
          ? AttributeValueSource.RULE
          : AttributeValueSource.DEFAULT;
    }
  }
}