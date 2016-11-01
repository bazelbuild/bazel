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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;

/**
 * An aspect that transpiles .proto dependencies using the J2ObjC proto plugin
 * and //tools/objc:standalone_protoc. N.B.: This tool has not yet been released into
 * open-source.
 */
public class BazelJ2ObjcProtoAspect extends AbstractJ2ObjcProtoAspect {
  public static final String NAME = "BazelJ2ObjcProtoAspect";

  public BazelJ2ObjcProtoAspect(String toolsRepository) {
    super(toolsRepository);
  }

  @Override
  protected AspectDefinition.Builder addAdditionalAttributes(AspectDefinition.Builder builder) {
    return builder.add(attr("$j2objc_plugin", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                toolsRepository + "//third_party/java/j2objc:proto_plugin")));
  }

  @Override
  protected boolean checkShouldCreateAspect(RuleContext ruleContext) {
    return true;
  }

  @Override
  protected boolean allowServices(RuleContext ruleContext) {
    return true;
  }
}
