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

package com.google.devtools.build.lib.rules.cpp.proto;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/** Declaration part of cc_proto_library. */
public class CcProtoLibraryRule implements RuleDefinition {

  private final CcProtoAspect ccProtoAspect;

  public CcProtoLibraryRule(CcProtoAspect ccProtoAspect) {
    this.ccProtoAspect = ccProtoAspect;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(cc_proto_library).ATTRIBUTE(deps) -->
        The list of <a href="protocol-buffer.html#proto_library"><code>proto_library</code></a>
        rules to generate C++ code for.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(
            attr("deps", LABEL_LIST)
                .allowedRuleClasses("proto_library")
                .allowedFileTypes()
                .aspect(ccProtoAspect))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_proto_library")
        .factoryClass(CcProtoLibrary.class)
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_proto_library, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>
<code>cc_proto_library</code> generates C++ code from <code>.proto</code> files.
</p>

<p>
<code>deps</code> must point to <a href="protocol-buffer.html#proto_library"><code>proto_library
</code></a> rules.
</p>

<p>
Example:
</p>

<pre class="code">
cc_library(
    name = "lib",
    deps = [":foo_cc_proto"],
)

cc_proto_library(
    name = "foo_cc_proto",
    deps = [":foo_proto"],
)

proto_library(
    name = "foo_proto",
)
</pre>


<!-- #END_BLAZE_RULE -->*/
