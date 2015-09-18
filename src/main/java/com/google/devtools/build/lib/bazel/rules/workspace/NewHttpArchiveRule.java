// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * Rule definition for the new_http_archive rule.
 */
public class NewHttpArchiveRule implements RuleDefinition {
  public static final String NAME = "new_http_archive";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(url) -->
         A URL referencing an archive file containing a Bazel repository.
         ${SYNOPSIS}

         <p>Archives of type .zip, .jar, .war, .tar.gz or .tgz are supported. There is no support
         for authentication. Redirections are followed.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("url", STRING).mandatory())
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(sha256) -->
         The expected SHA-256 hash of the file downloaded.
         ${SYNOPSIS}

         <p>This must match the SHA-256 hash of the file downloaded.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("sha256", STRING).mandatory())
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(build_file) -->
         The file to use as the BUILD file for this repository.
         ${SYNOPSIS}

         <p>This path is relative to the build's workspace. The file does not need to be named
         BUILD, but can be something like BUILD.new-repo-name to distinguish it
         from the workspace's actual BUILD files.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_file", STRING).mandatory())
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(type) -->
         The archive type of the downloaded file.
         ${SYNOPSIS}

         <p>By default, the archive type is determined from the file extension of the URL. If the
         file has no extension, you can explicitly specify either "zip", "jar", "tar.gz", or
         "tgz" here.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("type", STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(NewHttpArchiveRule.NAME)
        .type(RuleClass.Builder.RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = new_http_archive, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

${ATTRIBUTE_SIGNATURE}

<p>Downloads a compressed archive file, decompresses it, and creates a Bazel repository by
combining the archive with the provided BUILD file.</p>

<p>Only Zip-formatted archives with the .zip extension are supported.</p>

${ATTRIBUTE_DEFINITION}

<h4 id="new_http_archive_examples">Examples</h4>

<p>Suppose the current repository contains the source code for a chat program, rooted at the
  directory <i>~/chat-app</i>. It needs to depend on an SSL library which is available from
  <i>http://example.com/openssl.zip</i>. This .zip file contains the following directory
  structure:</p>

<pre class="code">
src/
  openssl.cc
  openssl.h
</pre>

<p>In the local repository, the user creates a <i>BUILD.ssl</i> file which contains the following
target definition:</p>

<pre class="code">
cc_library(
    name = "openssl-lib",
    srcs = ["src/openssl.cc"],
    hdrs = ["src/openssl.h"],
)
</pre>

<p>Targets in the <i>~/chat-app</i> repository can depend on this target if the following lines are
  added to <i>~/chat-app/WORKSPACE</i>:</p>

<pre class="code">
new_http_archive(
    name = "my-ssl",
    url = "http://example.com/openssl.zip",
    sha256 = "03a58ac630e59778f328af4bcc4acb4f80208ed4",
    build_file = "BUILD.ssl",
)
</pre>

<p>Targets would specify <code>@my-ssl//:openssl-lib</code> as a dependency to depend on this
 jar.</p>

<!-- #END_BLAZE_RULE -->*/
