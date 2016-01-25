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

package com.google.devtools.build.lib.bazel.rules.workspace;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;

/**
 * Rule definition for the http_archive rule.
 */
public class HttpArchiveRule implements RuleDefinition {

  public static final String NAME = "http_archive";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(http_archive).ATTRIBUTE(url) -->
         A URL referencing an archive file containing a Bazel repository.
         ${SYNOPSIS}

         <p>Archives of type .zip, .jar, .war, .tar.gz or .tgz are supported. There is no support
         for authentication. Redirections are followed.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("url", STRING).mandatory())
        /* <!-- #BLAZE_RULE(http_archive).ATTRIBUTE(sha256) -->
         The expected SHA-256 hash of the file downloaded.
         ${SYNOPSIS}

         <p>This must match the SHA-256 hash of the file downloaded.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("sha256", STRING).mandatory())
        /* <!-- #BLAZE_RULE(http_archive).ATTRIBUTE(type) -->
         The archive type of the downloaded file.
         ${SYNOPSIS}

         <p>By default, the archive type is determined from the file extension of the URL. If the
         file has no extension, you can explicitly specify either "zip", "jar", "tar.gz", or
         "tgz" here.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("type", STRING))
        /* <!-- #BLAZE_RULE(http_archive).ATTRIBUTE(strip_prefix) -->
         A directory prefix to strip from the extracted files.
         ${SYNOPSIS}

         <p>Many archives contain a top-level directory that contains all of the useful files in
         archive. Instead of needing to specify this prefix over and over in the
         <code>build_file</code>, this field can be used to strip it from all of the extracted
         files.</p>

         <p>For example, suppose you are using foo-lib-latest.zip, which contains the directory
         foo-lib-1.2.3/ under which there is a WORKSPACE file and are src/, lib/, and test/
         directories that contain the actual code you wish to build. Specify
         <code>strip_prefix = "foo-lib-1.2.3"</code> to use the foo-lib-1.2.3 directory as your
         top-level directory.</p>

         <p>Note that if there are files outside of this directory, they will be discarded and
         inaccessible (e.g., a top-level license file). This includes files/directories that
         start with the prefix but are not in the directory (e.g., foo-lib-1.2.3.release-notes).
         If the specified prefix does not match a directory in the archive, Bazel will return an
         error.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("strip_prefix", STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(HttpArchiveRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = http_archive, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Downloads a Bazel repository as a compressed archive file,
  decompresses it, and makes its targets available for binding. The
  repository should already contain a BUILD file. If it does not, use
  <a href="#new_http_archive">new_http_archive</a> instead.</p>

<p>It supports Zip-formatted archives (with the .zip extension) and
tarballs (.tar.gz and .tgz extensions).</p>

<h4 id="http_archive_examples">Examples</h4>

<p>Suppose the current repository contains the source code for a chat program, rooted at the
  directory <i>~/chat-app</i>. It needs to depend on an SSL library which is available from
  <i>http://example.com/openssl.zip</i>. This .zip file contains the following directory
  structure:</p>

<pre class="code">
WORKSPACE
src/
  BUILD
  openssl.cc
  openssl.h
</pre>

<p><i>src/BUILD</i> contains the following target definition:</p>

<pre class="code">
cc_library(
    name = "openssl-lib",
    srcs = ["openssl.cc"],
    hdrs = ["openssl.h"],
)
</pre>

<p>Targets in the <i>~/chat-app</i> repository can depend on this target if the following lines are
  added to <i>~/chat-app/WORKSPACE</i>:</p>

<pre class="code">
http_archive(
    name = "my-ssl",
    url = "http://example.com/openssl.zip",
    sha256 = "03a58ac630e59778f328af4bcc4acb4f80208ed4",
)
</pre>

<p>Then targets would specify <code>@my-ssl//src:openssl-lib</code> as a dependency.</p>

<!-- #END_BLAZE_RULE -->*/
