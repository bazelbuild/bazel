// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CcImportRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/** Rule definition for the cc_import rule. */
public final class BazelCcImportRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        .add(
            attr(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, LABEL)
                .mandatoryProviders(CcToolchainProvider.PROVIDER.id())
                .value(CppRuleClasses.ccToolchainAttribute(env)))
        .add(
            attr(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL)
                .value(CppRuleClasses.ccToolchainTypeAttribute(env)))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_import")
        .ancestors(BaseRuleClasses.NativeBuildRule.class, CcImportRule.class)
        .factoryClass(BazelCcImport.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_import, TYPE = LIBRARY, FAMILY = C / C++) -->
<p>
<code>cc_import</code> rules allows users to import precompiled C/C++ libraries.
</p>

<p>
The following are the typical use cases: <br/>

1. Linking a static library
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  static_library = "libmylib.a",
  # If alwayslink is turned on,
  # libmylib.a will be forcely linked into any binary that depends on it.
  # alwayslink = 1,
)
</pre>

2. Linking a shared library (Unix)
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  shared_library = "libmylib.so",
)
</pre>

3. Linking a shared library with interface library (Windows)
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  # mylib.lib is a import library for mylib.dll which will be passed to linker
  interface_library = "mylib.lib",
  # mylib.dll will be available for runtime
  shared_library = "mylib.dll",
)
</pre>

4. Linking a shared library with <code>system_provided=True</code> (Windows)
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  # mylib.lib is an import library for mylib.dll which will be passed to linker
  interface_library = "mylib.lib",
  # mylib.dll is provided by system environment, for example it can be found in PATH.
  # This indicates that Bazel is not responsible for making mylib.dll available.
  system_provided = 1,
)
</pre>

5. Linking to static or shared library <br/>

On Unix:
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  static_library = "libmylib.a",
  shared_library = "libmylib.so",
)

# first will link to libmylib.a
cc_binary(
  name = "first",
  srcs = ["first.cc"],
  deps = [":mylib"],
  linkstatic = 1, # default value
)

# second will link to libmylib.so
cc_binary(
  name = "second",
  srcs = ["second.cc"],
  deps = [":mylib"],
  linkstatic = 0,
)
</pre>

On Windows:
<pre class="code">
cc_import(
  name = "mylib",
  hdrs = ["mylib.h"],
  static_library = "libmylib.lib", # A normal static library
  interface_library = "mylib.lib", # An import library for mylib.dll
  shared_library = "mylib.dll",
)

# first will link to libmylib.lib
cc_binary(
  name = "first",
  srcs = ["first.cc"],
  deps = [":mylib"],
  linkstatic = 1, # default value
)

# second will link to mylib.dll through mylib.lib
cc_binary(
  name = "second",
  srcs = ["second.cc"],
  deps = [":mylib"],
  linkstatic = 0,
)
</pre>



</p>

<!-- #END_BLAZE_RULE -->*/
