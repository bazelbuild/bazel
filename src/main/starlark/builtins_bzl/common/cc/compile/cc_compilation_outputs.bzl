# Copyright 2025 The Bazel Authors. All rights reserved.
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
"""
The CcCompilationOutputs provider.
"""

load(":common/cc/cc_helper_internal.bzl", "wrap_with_check_private_api", _PRIVATE_STARLARKIFICATION_ALLOWLIST = "PRIVATE_STARLARKIFICATION_ALLOWLIST")
load(":common/cc/compile/lto_compilation_context.bzl", "EMPTY_LTO_COMPILATION_CONTEXT", "merge_lto_compilation_contexts")

_cc_common_internal = _builtins.internal.cc_common
_cc_internal = _builtins.internal.cc_internal

# buildifier: disable=name-conventions
_UnboundValueProviderDoNotUse = provider("This provider is used as an unique symbol to distinguish between bound and unbound Starlark values, to avoid using kwargs.", fields = [])
_UNBOUND = _UnboundValueProviderDoNotUse()

CcCompilationOutputsInfo = provider(
    doc = "The outputs of a C++ compilation.",
    fields = {
        "objects": "(list[File]) All .o files built by the target.",
        "pic_objects": "(list[File]) All .pic.o files built by the target.",
        "_lto_compilation_context": "(LTOCompilationContext) Maps all .o bitcode files coming from a ThinLTO C(++) compilation under our control toinformation needed by the LTO indexing and backend steps.",
        "_dwo_files": "(list[File]) All .dwo files built by the target, corresponding to .o outputs.",
        "_pic_dwo_files": "(list[File]) All .pic.dwo files built by the target, corresponding to .pic.o outputs.",
        "_gcno_files": "(list[File]) All .gcno files built by the target, corresponding to .o outputs.",
        "_pic_gcno_files": "(list[File]) All .pic.gcno files built by the target, corresponding to .pic.gcno outputs.",
        "temps": '(() -> depset[File]) All artifacts that are created if "--save_temps" is true.',
        "_header_tokens": "(list[File]) All token .h.processed files created when preprocessing or parsing headers.",
        "_module_files": "(list[File]) All .pcm files built by the target.",
    },
)

def create_compilation_outputs_internal(
        *,
        objects = [],
        pic_objects = [],
        lto_compilation_context = None,
        dwo_files = [],
        pic_dwo_files = [],
        gcno_files = [],
        pic_gcno_files = [],
        temps = [],
        header_tokens = [],
        module_files = []):
    """Creates a CcCompilationOutputsInfo provider.

    Args:
        objects: A list of object files.
        pic_objects: A list of PIC object files.
        lto_compilation_context: The LTO compilation context.
        dwo_files: A list of dwo files.
        pic_dwo_files: A list of PIC dwo files.
        gcno_files: A list of gcno files.
        pic_gcno_files: A list of PIC gcno files.
        temps: A depset of temporary files.
        header_tokens: A list of header tokens.
        module_files: A list of module files.

    Returns:
        A CcCompilationOutputsInfo provider.
    """
    if lto_compilation_context == None:
        lto_compilation_context = EMPTY_LTO_COMPILATION_CONTEXT
    return CcCompilationOutputsInfo(
        objects = _cc_internal.freeze(objects),
        pic_objects = _cc_internal.freeze(pic_objects),
        temps = wrap_with_check_private_api(depset(temps)),
        _header_tokens = _cc_internal.freeze(header_tokens),
        _module_files = _cc_internal.freeze(module_files),
        _lto_compilation_context = lto_compilation_context,
        _gcno_files = _cc_internal.freeze(gcno_files),
        _pic_gcno_files = _cc_internal.freeze(pic_gcno_files),
        _dwo_files = _cc_internal.freeze(dwo_files),
        _pic_dwo_files = _cc_internal.freeze(pic_dwo_files),
    )

EMPTY_COMPILATION_OUTPUTS = create_compilation_outputs_internal()

def _validate_extensions(param_name, files, valid_extensions):
    """Validates that the files have the correct extensions.

    Args:
        param_name: The name of the parameter being validated.
        files: A list of files to validate.
        valid_extensions: A list of valid extensions.
    """
    for file in files:
        if type(file) != "File":
            fail("for '%s', got a depset of '%s', expected a depset of 'File'" % (param_name, type(file)))
        if file.is_directory:
            continue
        if file.extension not in valid_extensions:
            fail(
                "'%s' has wrong extension. The list of possible extensions for '%s' is: %s" % (
                    file.path,
                    param_name,
                    ", ".join(valid_extensions),
                ),
            )

def _to_list(data):
    if not data:
        return []
    if type(data) == type(depset()):
        return data.to_list()
    if type(data) == type([]):
        return data
    fail("Expected depset or list for artifacts, got " + type(data))

# IMPORTANT: This function is public API exposed on cc_common module!
def create_compilation_outputs(
        *,
        objects = None,
        pic_objects = None,
        lto_compilation_context = _UNBOUND,
        dwo_objects = _UNBOUND,
        pic_dwo_objects = _UNBOUND):
    """Creates a CcCompilationOutputsInfo provider from Starlark.

    Args:
        objects: A depset or list of object files.
        pic_objects: A depset or list of PIC object files.
        lto_compilation_context: The LTO compilation context.
        dwo_objects: A depset or list of dwo files.
        pic_dwo_objects: A depset or list of PIC dwo files.

    Returns:
        A CcCompilationOutputsInfo provider.
    """
    if lto_compilation_context != _UNBOUND or dwo_objects != _UNBOUND or pic_dwo_objects != _UNBOUND:
        _cc_common_internal.check_private_api(allowlist = _PRIVATE_STARLARKIFICATION_ALLOWLIST)
    if lto_compilation_context == _UNBOUND:
        lto_compilation_context = None
    if dwo_objects == _UNBOUND:
        dwo_objects = []
    if pic_dwo_objects == _UNBOUND:
        pic_dwo_objects = []

    objects = _to_list(objects)
    pic_objects = _to_list(pic_objects)
    dwo_objects = _to_list(dwo_objects)
    pic_dwo_objects = _to_list(pic_dwo_objects)

    obj_extensions = ["o", "obj", ".opb", ".bc"]
    _validate_extensions("objects", objects, obj_extensions)
    _validate_extensions("pic_objects", pic_objects, obj_extensions)

    return create_compilation_outputs_internal(
        objects = objects,
        pic_objects = pic_objects,
        lto_compilation_context = lto_compilation_context,
        dwo_files = dwo_objects,
        pic_dwo_files = pic_dwo_objects,
    )

# IMPORTANT: This function is public API exposed on cc_common module!
def merge_compilation_outputs(*, compilation_outputs):
    """Merges a list of CcCompilationOutputsInfo providers.

    Args:
        compilation_outputs: A list of CcCompilationOutputsInfo providers.

    Returns:
        A CcCompilationOutputsInfo provider.
    """
    objects = []
    pic_objects = []
    dwo_files = []
    pic_dwo_files = []
    gcno_files = []
    pic_gcno_files = []
    lto_compilation_contexts = []
    transitive_temps = []
    header_tokens = []
    module_files = []

    for co in compilation_outputs:
        objects.extend(co.objects)
        pic_objects.extend(co.pic_objects)
        dwo_files.extend(co._dwo_files)
        pic_dwo_files.extend(co._pic_dwo_files)
        gcno_files.extend(co._gcno_files)
        pic_gcno_files.extend(co._pic_gcno_files)
        transitive_temps.append(co.temps())
        header_tokens.extend(co._header_tokens)
        module_files.extend(co._module_files)
        if co._lto_compilation_context:
            lto_compilation_contexts.append(co._lto_compilation_context)

    return CcCompilationOutputsInfo(
        objects = objects,
        pic_objects = pic_objects,
        _lto_compilation_context = merge_lto_compilation_contexts(lto_compilation_contexts = lto_compilation_contexts),
        _dwo_files = dwo_files,
        _pic_dwo_files = pic_dwo_files,
        _gcno_files = gcno_files,
        _pic_gcno_files = pic_gcno_files,
        temps = wrap_with_check_private_api(depset(transitive = transitive_temps)),
        _header_tokens = header_tokens,
        _module_files = module_files,
    )
