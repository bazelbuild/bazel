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

package com.google.devtools.build.lib.rules.proto;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.Scratch;

public class ProtoTestHelper {
  private ProtoTestHelper() {
    throw new UnsupportedOperationException();
  }

  public static final String LOAD_PROTO_LIBRARY =
      "load('@rules_proto//proto:defs.bzl', 'proto_library')";
  public static final String LOAD_PROTO_LANG_TOOLCHAIN =
      "load('@rules_proto//proto:defs.bzl', 'proto_lang_toolchain')";

  public static void setupWorkspace(BuildViewTestCase testCase) throws Exception {
    testCase.rewriteWorkspace(
        "local_repository(",
        "    name = 'rules_proto',",
        "    path = 'third_party/rules_proto',",
        ")");

    Scratch scratch = testCase.getScratch();
    scratch.file("third_party/rules_proto/WORKSPACE");
    scratch.file("third_party/rules_proto/proto/BUILD", "licenses(['notice'])");
    scratch.file(
        "third_party/rules_proto/proto/defs.bzl",
        "def _add_tags(kargs):",
        "    if 'tags' in kargs:",
        "        kargs['tags'] += ['__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
        "    else:",
        "        kargs['tags'] = ['__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
        "    return kargs",
        "",
        "def proto_library(**kargs): native.proto_library(**_add_tags(kargs))",
        "def proto_lang_toolchain(**kargs): native.proto_lang_toolchain(**_add_tags(kargs))");
  }
}
