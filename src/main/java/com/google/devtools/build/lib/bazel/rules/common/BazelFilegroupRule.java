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
package com.google.devtools.build.lib.bazel.rules.common;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.filegroup.Filegroup;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule object implementing "filegroup".
 */
public final class BazelFilegroupRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    // filegroup ignores any filtering set with setSrcsAllowedFiles.
    return builder
        /*<!-- #BLAZE_RULE(filegroup).ATTRIBUTE(srcs) -->
        The list of targets that are members of the file group.
        <p>
          It is common to use the result of a <a href="${link glob}">glob</a> expression for
          the value of the <code>srcs</code> attribute.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("srcs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
        /*<!-- #BLAZE_RULE(filegroup).ATTRIBUTE(output_group) -->
        The output group from which to gather artifacts from sources.  If this attribute is
        specified, artifacts from the specified output group of the dependencies will be exported
        instead of the default output group.
        <p>An "output group" is a category of output artifacts of a target, specified in that
          rule's implementation.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("output_group", STRING))
        /*<!-- #BLAZE_RULE(filegroup).ATTRIBUTE(data) -->
        The list of files needed by this rule at runtime.
        <p>
          Targets named in the <code>data</code> attribute will be added to the
          <code>runfiles</code> of this <code>filegroup</code> rule. When the
          <code>filegroup</code> is referenced in the <code>data</code> attribute of
          another rule its <code>runfiles</code> will be added to the <code>runfiles</code>
          of the depending rule. See the <a href="../build-ref.html#data">data dependencies</a>
          section and <a href="${link common-definitions#common.data}">general documentation of
          <code>data</code></a> for more information about how to depend on and use data files.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("data", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE).dontCheckConstraints())
        .add(attr("output_licenses", LICENSE))
        /*<!-- #BLAZE_RULE(filegroup).ATTRIBUTE(path) -->
        An optional string to set a path to the files in the group, relative to the package path.
        <p>
          This attribute can be used internally by other rules depending on this
          <code>filegroup</code> to find the name of the directory holding the files.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("path", STRING)
                .undocumented(
                    "only used to expose FilegroupPathProvider, which is not currently used"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("filegroup")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(Filegroup.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = filegroup, FAMILY = General)[GENERIC_RULE] -->

<p>
  Use <code>filegroup</code> to give a convenient name to a collection of targets.
  These can then be referenced from other rules.
</p>

<p>
  Using <code>filegroup</code> is encouraged instead of referencing directories directly.
  The latter is unsound since the build system does not have full knowledge of all files
  below the directory, so it may not rebuild when these files change. When combined with
  <a href="${link glob}">glob</a>, <code>filegroup</code> can ensure that all files are
  explicitly known to the build system.
</p>

<h4 id="filegroup_example">Examples</h4>

<p>
  To create a <code>filegroup</code> consisting of two source files, do
</p>
<pre class="code">
filegroup(
    name = "mygroup",
    srcs = [
        "a_file.txt",
        "some/subdirectory/another_file.txt",
    ],
)
</pre>
<p>
  Or, use a <code>glob</code> to grovel a testdata directory:
</p>
<pre class="code">
filegroup(
    name = "exported_testdata",
    srcs = glob([
        "testdata/*.dat",
        "testdata/logs/**&#47;*.log",
    ]),
)
</pre>
<p>
  To make use of these definitions, reference the <code>filegroup</code> with a label from any rule:
</p>
<pre class="code">
cc_library(
    name = "my_library",
    srcs = ["foo.cc"],
    data = [
        "//my_package:exported_testdata",
        "//my_package:mygroup",
    ],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
