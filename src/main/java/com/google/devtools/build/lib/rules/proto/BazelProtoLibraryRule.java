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

package com.google.devtools.build.lib.rules.proto;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;

/**
 * Rule definition for the proto_library rule.
 *
 * <p>This rule is implemented in Starlark. This class remains only for doc-gen purposes.
 */
public final class BazelProtoLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {

    return builder
        .requiresConfigurationFragments(ProtoConfiguration.class)
        .setOutputToGenfiles()
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(deps) -->
        The list of other <code>proto_library</code> rules that the target depends upon.
        A <code>proto_library</code> may only depend on other
        <code>proto_library</code> targets.
        It may not depend on language-specific libraries.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(attr("deps", LABEL_LIST).allowedFileTypes())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(srcs) -->
        The list of <code>.proto</code> and <code>.protodevel</code> files that are
        processed to create the target. This is usually a non empty list. One usecase
        where <code>srcs</code> can be empty is an <i>alias-library</i>. This is a
        proto_library rule having one or more other proto_library in <code>deps</code>.
        This pattern can be used to e.g. export a public api under a persistent name.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("srcs", LABEL_LIST)
                .direct_compile_time_input()
                .allowedFileTypes(FileType.of(".proto"), FileType.of(".protodevel")))
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(exports) -->
        List of proto_library targets that can be referenced via "import public" in the proto
        source.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("exports", LABEL_LIST).allowedFileTypes())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(strip_import_prefix) -->
        The prefix to strip from the paths of the .proto files in this rule.

        <p>When set, .proto source files in the <code>srcs</code> attribute of this rule are
        accessible at their path with this prefix cut off.

        <p>If it's a relative path (not starting with a slash), it's taken as a package-relative
        one. If it's an absolute one, it's understood as a repository-relative path.

        <p>The prefix in the <code>import_prefix</code> attribute is added after this prefix is
        stripped.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("strip_import_prefix", STRING))
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(import_prefix) -->
        The prefix to add to the paths of the .proto files in this rule.

        <p>When set, the .proto source files in the <code>srcs</code> attribute of this rule are
        accessible at is the value of this attribute prepended to their repository-relative path.

        <p>The prefix in the <code>strip_import_prefix</code> attribute is removed before this
        prefix is added.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("allow_exports", LABEL).allowedFileTypes())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(allow_exports) -->
        An optional allowlist that prevents proto library to be reexported or used in
        lang_proto_library that is not in one of the listed packages.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("import_prefix", STRING))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("proto_library")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = proto_library, TYPE = LIBRARY, FAMILY = Protocol Buffer) -->

<p>Deprecated. Please use <a href="https://github.com/bazelbuild/rules_proto">
   https://github.com/bazelbuild/rules_proto</a> instead.
</p>

<p>Use <code>proto_library</code> to define libraries of protocol buffers
   which may be used from multiple languages. A <code>proto_library</code> may be listed
   in the <code>deps</code> clause of supported rules, such as <code>java_proto_library</code>.
</p>

<p>When compiled on the command-line, a <code>proto_library</code> creates a file named
   <code>foo-descriptor-set.proto.bin</code>, which is the descriptor set for the
   messages the rule srcs. The file is a serialized <code>FileDescriptorSet</code>, which is
   described in
   <a href="https://developers.google.com/protocol-buffers/docs/techniques#self-description">
   https://developers.google.com/protocol-buffers/docs/techniques#self-description</a>.
</p>

<p>It only contains information about the <code>.proto</code> files directly mentioned by a
<code>proto_library</code> rule; the collection of transitive descriptor sets is available through
the <code>[ProtoInfo].transitive_descriptor_sets</code> Starlark provider.
See documentation in <code>ProtoInfo.java</code>.</p>

<p>Recommended code organization:</p>

<ul>
<li> One <code>proto_library</code> rule per <code>.proto</code> file.
<li> A file named <code>foo.proto</code> will be in a rule named <code>foo_proto</code>, which
   is located in the same package.
<li> A <code>[language]_proto_library</code> that wraps a <code>proto_library</code> named
  <code>foo_proto</code> should be called
   <code>foo_[language]_proto</code>, and be located in the same package.
</ul>

<!-- #END_BLAZE_RULE -->*/
