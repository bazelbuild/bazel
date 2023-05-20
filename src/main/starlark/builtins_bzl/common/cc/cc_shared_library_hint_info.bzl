# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""CcSharedLibraryHintInfo provider.

Needed in a separate file to break cycle with cc_common. cc_common needs a
method to return this provider in order for users to check if the provider
is already present in the current Bazel version. It's not possible to check
for the presence of top level symbols in Starlark
"""

CcSharedLibraryHintInfo = provider(
    doc = """
    This provider should be used by rules that provide C++ linker inputs and
    want to guide what the cc_shared_library uses. The reason for this may be
    for example because the rule is not providing a standard provider like
    CcInfo or ProtoInfo or because the rule does not want certain attributes
    to be used for linking into shared libraries. It may also be needed if the
    rule is using non-standard linker_input.owner names.

    Propagation of the cc_shared_library aspect will always happen via all
    attributes that provide either CcInfo, ProtoInfo or
    CcSharedLibraryHintInfo, the hints control whether the result of that
    propagation actually gets used.
    """,
    fields = {
        "attributes": ("[String] - If not set, the aspect will use the result of every " +
                       "dependency that provides CcInfo, ProtoInfo or CcSharedLibraryHintInfo. " +
                       "If empty list, the aspect will not use the result of any dependency. If " +
                       "the list contains a list of attribute names, the aspect will only use the " +
                       "dependencies corresponding to those attributes as long as they provide CcInfo, " +
                       "ProtoInfo or CcSharedLibraryHintInfo"),
        "owners": ("[Label] - cc_shared_library will know which linker_inputs to link based on the owners " +
                   "field of each linker_input. Most rules will simply use the ctx.label but certain " +
                   "APIs like cc_common.create_linker_input(owner=) accept any label. " +
                   "cc_common.create_linking_context_from_compilation_outputs() accepts a `name` which " +
                   "will then be used to create the owner of the linker_input together with ctx.package." +
                   "For these cases, since the cc_shared_library cannot guess, the rule author should " +
                   "provide a hint with the owners of the linker inputs. If the value of owners is not set, then " +
                   "ctx.label will be used. If the rule author passes a list and they want ctx.label plus some other " +
                   "label then they will have to add ctx.label explicitly. If you want to use custom owners from C++ " +
                   "rules keep as close to the original ctx.label as possible, to avoid conflicts with linker_inputs " +
                   "created by other targets keep the original repository name, the original package name and re-use " +
                   "the original name as part of your new name, limiting your custom addition to a prefix or suffix."),
    },
)
