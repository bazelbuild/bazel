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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoLibrary;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;

/** Declaration of the {@code java_proto_library} rule. */
public class BazelJavaProtoLibraryRule implements RuleDefinition {

  private final BazelJavaProtoAspect javaProtoAspect;

  public BazelJavaProtoLibraryRule(BazelJavaProtoAspect javaProtoAspect) {
    this.javaProtoAspect = javaProtoAspect;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class, ProtoConfiguration.class)
        /* <!-- #BLAZE_RULE(java_proto_library).ATTRIBUTE(deps) -->
        The list of <a href="protocol-buffer.html#proto_library"><code>proto_library</code></a>
        rules to generate Java code for.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(
            attr("deps", LABEL_LIST)
                .allowedRuleClasses("proto_library")
                .allowedFileTypes()
                .aspect(javaProtoAspect))
        .add(attr("strict_deps", BOOLEAN).value(true).undocumented("for migration"))
        .advertiseStarlarkProvider(StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_proto_library")
        .factoryClass(JavaProtoLibrary.class)
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
    deps = [":foo_java_proto"],
)

java_proto_library(
    name = "foo_java_proto",
    deps = [":foo_proto"],
)

proto_library(
    name = "foo_proto",
)
</pre>


<!-- #END_BLAZE_RULE -->*/
