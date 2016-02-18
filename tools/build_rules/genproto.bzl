# Copyright 2014 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a quick and dirty rule to make Bazel compile itself.  It
# only supports Java.

proto_filetype = FileType([".proto"])

def gensrcjar_impl(ctx):
  out = ctx.outputs.srcjar

  ctx.action(
    command=' '.join([
        "JAR='%s'" % ctx.executable._jar.path,
        "OUTPUT='%s'" % out.path,
        "PROTO_COMPILER='%s'" % ctx.executable._proto_compiler.path,
        "SOURCE='%s'" % ctx.file.src.path,
        ctx.executable._gensrcjar.path,
    ]),
    inputs=([ctx.file.src] + ctx.files._gensrcjar + ctx.files._jar +
            ctx.files._proto_compiler),
    outputs=[out],
    mnemonic="GenProtoSrcJar",
    use_default_shell_env=True)

gensrcjar = rule(
    gensrcjar_impl,
    attrs = {
        "src": attr.label(
            allow_files = proto_filetype,
            single_file = True,
        ),
        "_gensrcjar": attr.label(
            default = Label("@bazel_tools//tools/build_rules:gensrcjar"),
            allow_files = True,
            executable = True,
        ),
        # TODO(bazel-team): this should be a hidden attribute with a default
        # value, but Skylark needs to support select first.
        "_proto_compiler": attr.label(
            default = Label("@bazel_tools//third_party:protoc"),
            allow_files = True,
            executable = True,
            single_file = True,
        ),
        "_jar": attr.label(
            default = Label("@bazel_tools//tools/jdk:jar"),
            allow_files = True,
            executable = True,
            single_file = True,
        ),
    },
    outputs = {"srcjar": "lib%{name}.srcjar"},
)

# TODO(bazel-team): support proto => proto dependencies too
def java_proto_library(name, src):
  gensrcjar(name=name + "_srcjar", src=src)
  native.java_library(
    name=name,
    srcs=[name + "_srcjar"],
    deps=["@bazel_tools//third_party:protobuf"],
    # The generated code has lots of 'rawtypes' warnings.
    javacopts=["-Xlint:-rawtypes"],
)

def proto_java_library(name, src):
  print("Deprecated: use java_proto_library() instead, proto_java_library " +
        "will be removed in version 0.2.1")
  java_proto_library(name, src)
