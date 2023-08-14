// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;
import javax.annotation.Nullable;

/** Definition of the {@code starlark_doc_extract} rule. */
public final class StarlarkDocExtractRule implements RuleDefinition {
  @Override
  @Nullable
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(starlark_doc_extract).ATTRIBUTE(src) -->
        A Starlark file from which to extract documentation.

        <p>Note that this must be a file in the source tree; Bazel cannot <code>load()</code>
        generated files.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(StarlarkDocExtract.SRC_ATTR, LABEL)
                .allowedFileTypes(FileType.of(".bzl"), FileType.of(".scl"))
                .singleArtifact()
                .mandatory())
        /*<!-- #BLAZE_RULE(starlark_doc_extract).ATTRIBUTE(deps) -->
        A list of targets wrapping the Starlark files which are <code>load()</code>-ed by
        <code>src</code>. These targets <em>should</em> under normal usage be
        <a href="https://github.com/bazelbuild/bazel-skylib/blob/main/bzl_library.bzl"><code>bzl_library</code></a>
        targets, but the <code>starlark_doc_extract</code> rule does not enforce that, and accepts
        any target which provides Starlark files in its <code>DefaultInfo</code>.

        <p>Note that the wrapped Starlark files must be files in the source tree; Bazel cannot
        <code>load()</code> generated files.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        // TODO(https://github.com/bazelbuild/bazel/issues/18599): we cannot set
        // mandatoryProviders(StarlarkLibraryInfo) because StarlarkLibraryInfo is defined in
        // bazel_skylib, not natively in Bazel. Ideally, we ought to get rid of the deps attribute
        // altogether; but that requires new dependency machinery for `bazel query` to use the
        // Starlark load graph for collecting the dependencies of starlark_doc_extract's src.
        .override(
            attr(StarlarkDocExtract.DEPS_ATTR, LABEL_LIST)
                .allowedFileTypes(FileType.of(".bzl"), FileType.of(".scl")))
        /*<!-- #BLAZE_RULE(starlark_doc_extract).ATTRIBUTE(symbol_names) -->
        An optional list of qualified names of exported functions, rules, providers, or aspects (or
        structs in which they are nested) for which to extract documentation. Here, a <em>qualified
        name</em> means the name under which an entity is made available to a user of the module,
        including any structs in which the entity is nested for namespacing.

        <p><code>starlark_doc_extract</code> emits documentation for an entity if and only if
        <ol>
          <li>
            each component of the entity's qualified name is public (in other words, the first
            character of each component of the qualified name is alphabetic, not <code>"_"</code>);
            <em>and</em>
          </li>
          <li>
            <ol>
              <li>
                <em>either</em> the <code>symbol_names</code> list is empty (which is the default
                case), <em>or</em>
              </li>
              <li>
                the entity's qualified name, or the qualified name of a struct in which the entity
                is nested, is in the <code>symbol_names</code> list.
              </li>
            </ol>
          </li>
        </ol>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(StarlarkDocExtract.SYMBOL_NAMES_ATTR, STRING_LIST)
                .value(ImmutableList.<String>of()))
        /*<!-- #BLAZE_RULE(starlark_doc_extract).ATTRIBUTE(render_main_repo_name) -->
        If true, render labels in the main repository in emitted documentation with a repo component
        (in other words, <code>//foo:bar.bzl</code> will be emitted as
        <code>@main_repo_name//foo:bar.bzl</code>).
        <p>The name to use for the main repository is obtained from <code>module(name = ...)</code>
        in the main repository's <code>MODULE.bazel</code> file (if Bzlmod is enabled), or from
        <code>workspace(name = ...)</code> in the main repository's <code>WORKSPACE</code> file.
        <p>This attribute should be set to <code>False</code> when generating documentation for
        Starlark files which are intended to be used only within the same repository, and to
        <code>True</code> when generating documentation for Starlark files which are intended to be
        used from other repositories.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(StarlarkDocExtract.RENDER_MAIN_REPO_NAME, BOOLEAN).value(false))
        /*<!-- #BLAZE_RULE(starlark_doc_extract).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code><var>name</var>.binaryproto</code> (the default output): A
            <code>ModuleInfo</code> binary proto.</li>
          <li><code><var>name</var>.textproto</code> (only built if explicitly requested): the text
            proto version of <code><var>name</var>.binaryproto</code>.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            fromFunctions(StarlarkDocExtract.BINARYPROTO_OUT, StarlarkDocExtract.TEXTPROTO_OUT))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        // TODO(b/276733514): add `bazel dump --starlark_doc` command.
        .name("starlark_doc_extract")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(StarlarkDocExtract.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = starlark_doc_extract, FAMILY = General)[GENERIC_RULE] -->

<p><code>starlark_doc_extract()</code> extracts documentation for rules, functions (including
macros), aspects, and providers defined or re-exported in a given <code>.bzl</code> or
<code>.scl</code> file. The output of this rule is a <code>ModuleInfo</code> binary proto as defined
in
<a href="https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/skydoc/rendering/proto/stardoc_output.proto">stardoc_output.proto</a>
in the Bazel source tree.

${IMPLICIT_OUTPUTS}

<p>Warning: the output format of this rule is not guaranteed to be stable. It is intended mainly for
internal use by <a href="https://github.com/bazelbuild/stardoc">Stardoc</a>.

<!-- #END_BLAZE_RULE -->*/
