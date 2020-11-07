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

package com.google.devtools.build.lib.rules.proto;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;

/** Implements {code proto_lang_toolchain}. */
public class ProtoLangToolchainRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder

        /* <!-- #BLAZE_RULE(proto_lang_toolchain).ATTRIBUTE(command_line) -->
        This value will be passed to proto-compiler to generate the code. Only include the parts
        specific to this code-generator/plugin (e.g., do not include -I parameters)
        <ul>
          <li><code>$(OUT)</code> is LANG_proto_library-specific. The rules are expected to define
              how they interpret this variable. For Java, for example, $(OUT) will be replaced with
              the src-jar filename to create.</li>
          <li><code>$(PLUGIN_out)</code> will be substituted to work with a
              `--plugin=protoc-gen-PLUGIN` command line.</li>
        </ul>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("command_line", Type.STRING).mandatory())

        /* <!-- #BLAZE_RULE(proto_lang_toolchain).ATTRIBUTE(plugin) -->
        If provided, will be made available to the action that calls the proto-compiler, and will be
        passed to the proto-compiler:
        <code>--plugin=protoc-gen-PLUGIN=<executable>.</code>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("plugin", LABEL).exec().cfg(HostTransition.createFactory()).allowedFileTypes())

        /* <!-- #BLAZE_RULE(proto_lang_toolchain).ATTRIBUTE(runtime) -->
        A language-specific library that the generated code is compiled against.
        The exact behavior is LANG_proto_library-specific.
        Java, for example, should compile against the runtime.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("runtime", LABEL).allowedFileTypes())

        /* <!-- #BLAZE_RULE(proto_lang_toolchain).ATTRIBUTE(blacklisted_protos) -->
        No code will be generated for files in the <code>srcs</code> attribute of
        <code>blacklisted_protos</code>.
        This is used for .proto files that are already linked into proto runtimes, such as
        <code>any.proto</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("blacklisted_protos", LABEL_LIST)
                .allowedFileTypes()
                .mandatoryProviders(StarlarkProviderIdentifier.forKey(ProtoInfo.PROVIDER.getKey())))
        .requiresConfigurationFragments(ProtoConfiguration.class)
        .advertiseProvider(ProtoLangToolchainProvider.class)
        .removeAttribute("data")
        .removeAttribute("deps")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("proto_lang_toolchain")
        .ancestors(BaseRuleClasses.RuleBase.class)
        .factoryClass(ProtoLangToolchain.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = proto_lang_toolchain, TYPE = LIBRARY, FAMILY = Protocol Buffer) -->

<p>Deprecated. Please <a href="https://github.com/bazelbuild/rules_proto">
   https://github.com/bazelbuild/rules_proto</a> instead.
</p>

<p>
Specifies how a LANG_proto_library rule (e.g., <code>java_proto_library</code>) should invoke the
proto-compiler.
Some LANG_proto_library rules allow specifying which toolchain to use using command-line flags;
consult their documentation.
</p>

<p>Normally you should not write those kind of rules unless you want to
tune your Java compiler.</p>

<p>
There's no compiler. The proto-compiler is taken from the proto_library rule we attach to. It is
passed as a command-line flag to Blaze.
Several features require a proto-compiler to be invoked on the proto_library rule itself.
It's beneficial to enforce the compiler that LANG_proto_library uses is the same as the one
<code>proto_library</code> does.
</p>

<h4>Examples</h4>

<p>A simple example would be:
</p>

<pre class="code">
proto_lang_toolchain(
    name = "javalite_toolchain",
    command_line = "--$(PLUGIN_OUT)=shared,immutable:$(OUT)",
    plugin = ":javalite_plugin",
    runtime = ":protobuf_lite",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
