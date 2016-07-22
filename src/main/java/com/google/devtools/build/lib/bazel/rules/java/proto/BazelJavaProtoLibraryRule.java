// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.java.proto;

import static com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoAspect.SPEED_PROTO_RUNTIME_ATTR;
import static com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoAspect.SPEED_PROTO_RUNTIME_LABEL;
import static com.google.devtools.build.lib.packages.Aspect.INJECTING_RULE_KIND_PARAMETER_KEY;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;

import javax.annotation.Nullable;

/** Declaration of the {@code java_proto_library} rule. */
public class BazelJavaProtoLibraryRule implements RuleDefinition {

  private static final Function<Rule, AspectParameters> ASPECT_PARAMETERS =
      new Function<Rule, AspectParameters>() {
        @Nullable
        @Override
        public AspectParameters apply(@Nullable Rule rule) {
          return new AspectParameters.Builder()
              .addAttribute(INJECTING_RULE_KIND_PARAMETER_KEY, "java_proto_library")
              .build();
        }
      };

  private final BazelJavaProtoAspect javaProtoAspect;

  public BazelJavaProtoLibraryRule(BazelJavaProtoAspect javaProtoAspect) {
    this.javaProtoAspect = javaProtoAspect;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        // This rule isn't ready for use yet.
        .setUndocumented()
        /* <!-- #BLAZE_RULE(java_proto_library).ATTRIBUTE(deps) -->
        The list of <a href="protocol-buffer.html#proto_library"><code>proto_library</code></a>
        rules to generate Java code for.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(
            attr("deps", LABEL_LIST)
                .allowedRuleClasses("proto_library")
                .allowedFileTypes()
                .aspect(javaProtoAspect, ASPECT_PARAMETERS))
        .add(
            attr(SPEED_PROTO_RUNTIME_ATTR, LABEL)
                .legacyAllowAnyFileType()
                .value(Label.parseAbsoluteUnchecked(SPEED_PROTO_RUNTIME_LABEL)))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_proto_library")
        .factoryClass(BazelJavaProtoLibrary.class)
        .ancestors(BaseRuleClasses.RuleBase.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = java_proto_library, TYPE = LIBRARY, FAMILY = Java) -->

<p>
<code>java_proto_library</code> generates Java code from <code>.proto</code> files.
</p>

<p>
<code>deps</code> must point to <a href="protocol-buffer.html#proto_library"><code>proto_library
</code></a> rules.
</p>

<p>
Example:
</p>

<pre class="code">
java_library(
    name = "lib",
    deps = [":foo"],
)

java_proto_library(
    name = "foo",
    deps = [":bar"],
)

proto_library(
    name = "bar",
)
</pre>


<!-- #END_BLAZE_RULE -->*/
