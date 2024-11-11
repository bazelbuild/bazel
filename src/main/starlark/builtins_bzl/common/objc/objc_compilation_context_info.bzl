# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""A collection of compilation information gathered for a particular rule.

This is used to generate the compilation command line and to the supply information that goes
into the compilation info provider.
"""

load(":common/cc/cc_common.bzl", "cc_common")

def _objc_compilation_context_info_init(
        defines = [],
        public_hdrs = [],
        public_textual_hdrs = [],
        private_hdrs = [],
        includes = [],
        system_includes = [],
        quote_includes = [],
        direct_cc_compilation_contexts = [],
        cc_compilation_contexts = [],
        implementation_cc_compilation_contexts = [],
        providers = []):
    strict_dependency_includes = [
        path
        for objc_provider in providers
        for path in objc_provider.strict_include.to_list()
    ]
    return {
        "defines": defines,
        "public_hdrs": public_hdrs,
        "public_textual_hdrs": public_textual_hdrs,
        "private_hdrs": private_hdrs,
        "includes": includes,
        "system_includes": system_includes,
        "quote_includes": quote_includes,
        "strict_dependency_includes": strict_dependency_includes,
        "direct_cc_compilation_contexts": direct_cc_compilation_contexts,
        "cc_compilation_contexts": cc_compilation_contexts,
        "implementation_cc_compilation_contexts": implementation_cc_compilation_contexts,
    }

ObjcCompilationContextInfo, _new_objccompilationcontextinfo = provider(
    "Provider about ObjC compilation information gathered for a particular rule.",
    fields = {
        "defines": "",
        "public_hdrs": """The list of public headers. We expect this to contain both the headers
            from the src attribute, as well as any "additional" headers required for compilation.""",
        "public_textual_hdrs": "",
        "private_hdrs": "",
        "includes": "",
        "system_includes": "",
        "quote_includes": "",
        "strict_dependency_includes": "",
        "direct_cc_compilation_contexts": "",
        "cc_compilation_contexts": "",
        "implementation_cc_compilation_contexts": "",
    },
    init = _objc_compilation_context_info_init,
)

def create_cc_compilation_context(objc_compilation_context_info):
    return cc_common.create_compilation_context(
        defines = depset(objc_compilation_context_info.defines),
        headers = depset(
            objc_compilation_context_info.public_hdrs +
            objc_compilation_context_info.private_hdrs +
            objc_compilation_context_info.public_textual_hdrs,
        ),
        direct_public_headers = objc_compilation_context_info.public_hdrs,
        direct_private_headers = objc_compilation_context_info.private_hdrs,
        direct_textual_headers = objc_compilation_context_info.public_textual_hdrs,
        includes = depset(objc_compilation_context_info.includes),
        system_includes = depset(objc_compilation_context_info.system_includes),
        quote_includes = depset(objc_compilation_context_info.quote_includes),
        dependent_cc_compilation_contexts = objc_compilation_context_info.cc_compilation_contexts,
        exported_dependent_cc_compilation_contexts = objc_compilation_context_info.direct_cc_compilation_contexts,
    )
