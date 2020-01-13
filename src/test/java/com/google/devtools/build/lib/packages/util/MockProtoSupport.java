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

package com.google.devtools.build.lib.packages.util;

import com.google.devtools.build.lib.rules.proto.ProtoCommon;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/**
 * Creates mock BUILD files required for the proto_library rule.
 */
public final class MockProtoSupport {
  private MockProtoSupport() {
    throw new UnsupportedOperationException();
  }

  /** This is workarround for tests that do not use {@code Scratch} (e.g. aquery-tests). */
  public static final String MIGRATION_TAG =
      String.format("tags = ['%s'],", ProtoCommon.PROTO_RULES_MIGRATION_LABEL);

  /**
   * Setup the support for building proto_library. You additionally need to setup support for each
   * of the languages used in the specific test.
   *
   * <p>Cannot be used for integration tests that actually need to run protoc.
   */
  public static void setup(MockToolsConfig config) throws IOException {
    createNetProto2(config);
    createJavascriptJspb(config);
  }

  /**
   * Create a dummy "net/proto2 compiler and proto APIs for all languages
   * and versions.
   */
  private static void createNetProto2(MockToolsConfig config) throws IOException {
    config.create(
        "net/proto2/compiler/public/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "sh_binary(name='protocol_compiler', srcs=['protocol_compiler.sh'])");

    if (config.isRealFileSystem()) {
      // when using a "real" file system, import the jars and link to ensure compilation
      config.create("java/com/google/io/protocol/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "java_import(name = 'protocol',",
          "            jars = [ 'protocol.jar' ])");
      config.create("java/com/google/io/protocol2/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "java_import(name = 'protocol2',",
          "            jars = [ 'protocol2.jar' ])");

      config.linkTool("net/proto2/compiler/public/release/protocol_compiler_linux",
          "net/proto2/compiler/public/protocol_compiler");
      config.linkTool("javatests/com/google/devtools/build/lib/prepackaged_protocol_deploy.jar",
          "java/com/google/io/protocol/protocol.jar");
      config.linkTool("javatests/com/google/devtools/build/lib/prepackaged_protocol2_deploy.jar",
          "java/com/google/io/protocol2/protocol2.jar");
    } else {
      // for "fake" file systems, provide stub rules. This is different from the "real" filesystem,
      // as it produces the interface jars that the production environment has.
      config.create("java/com/google/io/protocol/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "java_library(name = 'protocol',",
          "             srcs = [ 'Protocol.java' ])");
      config.create("java/com/google/io/protocol/Protocol.java");
      config.create("java/com/google/io/protocol2/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "java_library(name = 'protocol2',",
          "             srcs = [ 'Protocol2.java' ])");
      config.create("java/com/google/io/protocol/Protocol2.java");
    }

    config.create(
        "java/com/google/protobuf/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "proto_library(name = 'protobuf_proto_sources', srcs = [])");

    // RPC generator plugins.
    config.create("net/rpc/compiler/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_binary(name = 'proto2_py_plugin',",
        "          srcs = [ 'proto2_py_plugin.cc' ])",
        "cc_binary(name = 'proto2_java_plugin',",
        "          srcs = [ 'proto2_java_plugin.cc' ])");

    config.create("net/grpc/compiler/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_binary(name = 'composite_cc_plugin',",
        "          srcs = [ 'composite_cc_plugin.cc' ])");

    // Fake targets for proto API libs of all languages and versions.
    config.create(
        "net/proto2/public/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'cc_proto_library_blaze_internal_deps',",
        "           srcs = [ 'cc_proto_library_blaze_internal_deps.cc' ])");
    config.create("net/proto2/python/public/BUILD",
        "package(default_visibility=['//visibility:public'])",
         "py_library(name = 'public',",
         "           srcs = [ 'pyproto2.py' ])");
    config.create("net/proto2/bridge/public/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'compatibility_mode_support',",
        "           srcs = [ 'compatibility.cc' ])");
    config.create(
        "net/proto/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'proto',",
        "           srcs = [ 'proto.cc' ])",
        "py_library(name = 'pyproto',",
        "           srcs = [ 'pyproto.py' ])");
    config.create("net/proto/python/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "py_library(name = 'proto1',",
        "           srcs = [ 'pyproto.py' ])");
    config.create(
        "net/rpc/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'stubby12_proto_rpc_libs')",
        "cc_library(name = 'no_stubby_rpc_libs_please_dont_depend_on_this')");
    config.create("net/rpc4/public/core/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'stubby4_rpc_libs')");
    config.create("net/grpc/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_library(name = 'grpc++_codegen_lib')");
    config.create("net/rpc/python/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "py_library(name = 'proto_python_api_1_stub',",
        "           srcs = [ 'test_only_prefix_proto_python_api_1_stub.py' ])",
        "py_library(name = 'proto_python_api_2_stub',",
        "           srcs = [ 'test_only_prefix_proto_python_api_2_stub.py' ])");
    config.create("java/com/google/net/rpc/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(name = 'rpc',",
        "             srcs = [ 'Rpc.java' ])",
        "java_library(name = 'rpc_noloas_internal',",
        "             srcs = [ 'RpcNoloas.java' ])");
    config.create(
        "java/com/google/net/rpc3/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "java_library(name = 'rpc3_proto_runtime',",
        "             deps = [':rpc3_noloas_internal'],",
        "             srcs = [ 'Rpc3.java' ])",
        "java_library(name = 'rpc3_noloas_internal',",
        "             deps = ['//java/com/google/net/rpc:rpc_noloas_internal'],",
        "             srcs = [ 'Rpc3Noloas.java' ])");
    config.create(
        "net/proto2/proto/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "package(default_visibility=['//visibility:public'])",
        "genrule(name = 'go_internal_bootstrap_hack',",
        "        srcs = [ 'descriptor.pb.go-prebuilt' ],",
        "        outs = [ 'descriptor.pb.go' ],",
        "        cmd = '')",
        "proto_library(name='descriptor',",
        "              srcs=['descriptor.proto'])");
    config.create(
        "net/proto2/go/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "go_library(name = 'protodeps',",
        "           srcs = [ 'protodeps.go' ])",
        "go_library(name = 'proto',",
        "           srcs = [ 'proto.go' ])");
    config.create("net/proto2/compiler/go/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "go_binary(name = 'protoc-gen-go',",
        "          srcs = [ 'main.go' ])");
    config.create("net/rpc/go/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "go_library(name = 'rpc',",
        "           srcs = [ 'rpc.go' ])");
    config.create("go/context/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "go_library(name = 'context',",
        "           srcs = [ 'context.go' ])");
    config.create("third_party/py/six/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "py_library(name = 'six',",
        "           srcs = [ '__init__.py' ])");
    // TODO(b/77901188): remove once j_p_l migration is complete
    config.create(
        "third_party/java/jsr250_annotations/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "java_library(name = 'jsr250_source_annotations',",
        "           srcs = [ 'Generated.java' ])");
    config.create(
        "third_party/golang/grpc/metadata/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "go_library(name = 'metadata',",
        "           srcs = [ 'metadata.go' ])");
  }

  /** Create a dummy jspb support package. */
  private static void createJavascriptJspb(MockToolsConfig config) throws IOException {
    config.create(
        "net/proto2/compiler/js/internal/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "cc_binary(name = 'protoc-gen-js',",
        "    srcs = ['plugin.cc'])");
    config.create(
        "javascript/apps/jspb/BUILD",
        "load('//tools/build_defs/js:rules.bzl', 'js_library')",
        "package(default_visibility=['//visibility:public'])",
        "js_library(name = 'message',",
        "       srcs = ['message.js'],",
        "       deps_mgmt = 'legacy')");
    config.create(
        "javascript/closure/array/BUILD",
        "load('//tools/build_defs/js:rules.bzl', 'js_library')",
        "package(default_visibility=['//visibility:public'])",
        "js_library(name = 'array',",
        "       srcs = ['array.js'],",
        "       deps_mgmt = 'legacy')");
    config.create(
        "javascript/apps/xid/BUILD",
        "load('//tools/build_defs/js:rules.bzl', 'js_library')",
        "package(default_visibility=['//visibility:public'])",
        "js_library(name = 'xid',",
        "       srcs = ['xid.js'],",
        "       deps_mgmt = 'legacy')");
  }

  public static void setupWorkspace(Scratch scratch) throws Exception {
    scratch.appendFile(
        "WORKSPACE",
        "local_repository(",
        "    name = 'rules_proto',",
        "    path = 'third_party/rules_proto',",
        ")");
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
