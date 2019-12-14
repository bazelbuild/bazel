// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.blackbox.tests.workspace;

import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import org.junit.Test;

public class SpyBlackBoxTest extends AbstractBlackBoxTest {

  @Test
  public void testSimpleDeps() throws Exception {
    context().write(WORKSPACE, "workspace(name ='test')",
        "load(\"@bazel_tools//tools/build_defs/repo:http.bzl\", \"http_archive\")\n"

            + "http_archive(\n"
            + "    name = \"bazel_skylib\",\n"
            + "    urls = [\n"
            + "        \"https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz\",\n"
            + "        \"https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz\",\n"
            + "    ],\n"
            + "    sha256 = \"97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44\",\n"
            + ")\n"
            + "load(\"@bazel_skylib//:workspace.bzl\", \"bazel_skylib_workspace\")\n"

            + "bazel_skylib_workspace()");
    context()
        .write(
            "rule.bzl",
            "def _impl(ctx):",
            "  out = ctx.actions.declare_file('does_not_matter')",
            "  ctx.actions.do_nothing(mnemonic = 'UseInput', inputs = ctx.attr.dep.files)",
            "  ctx.actions.write(out, 'Hi')",
            "  return [DefaultInfo(files = depset([out]))]",
            "",
            "debug_rule = rule(",
            "    implementation = _impl,",
            "    attrs = {",
            "        \"dep\": attr.label(allow_single_file = True),",
            "    }",
            ")");
    context().write("BUILD", "load(':rule.bzl', 'debug_rule')",
        "debug_rule(name='debug', dep='@bazel_skylib//:bzl_library.bzl')");

    context().bazel().build("//...");
  }
}
