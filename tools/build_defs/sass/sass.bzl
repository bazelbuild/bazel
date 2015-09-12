# Copyright 2015 Google Inc. All rights reserved.
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

SASS_FILETYPES = FileType([".sass", ".scss"])

def _sass_binary_impl(ctx):
    # Reference the sass compiler and define the default options
    # that sass_binary uses.
    sassc = ctx.file._sassc        
    options = [
        "--style={0}".format(ctx.attr.output_style),
        "--sourcemap",
    ]

    # Dynamically include all dependencies as part of the options.
    includes = set()
    for dep in ctx.attr.deps:
        for file in dep.files:
            includes = includes | [file]
    for include in includes:
        options = options + ["-I={0}".format(include)]
                             
    ctx.action(
        inputs = [sassc, ctx.file.src],
        executable = sassc,
        arguments = options + [ctx.file.src.path, ctx.outputs.css_file.path],
        mnemonic = "SassCompiler",
        outputs = [ctx.outputs.css_file, ctx.outputs.css_map_file],
    )

sass_binary = rule(
    implementation = _sass_binary_impl,
    attrs = {
        "src": attr.label(
            allow_files = SASS_FILETYPES,
            mandatory = True,
            single_file = True,
        ),
        "output_style": attr.string(default = "compressed"),
        "deps": attr.label_list(allow_files = SASS_FILETYPES),
        "_sassc": attr.label(
            default = Label("//tools/build_defs/sass:sassc"),
            executable = True,
            single_file = True,
        ),
    },
    outputs = {
        "css_file": "%{name}.css",
        "css_map_file": "%{name}.css.map",
    },
)
