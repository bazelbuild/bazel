# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""CSharp bazel rules"""

_MONO_UNIX_CSC = "/usr/local/bin/mcs"

# TODO(jeremy): Windows when it's available.

def _make_csc_flag(flag_start, flag_name, flag_value=None):
    return flag_start + flag_name + (":" + flag_value if flag_value else "")

def _make_csc_deps(deps, extra_files=[]):
    dlls = set()
    refs = set()
    transitive_dlls = set()
    for dep in deps:
        if hasattr(dep, "target_type"):
          dep_type = getattr(dep, "target_type")
          if dep_type == "exe":
              fail("You can't use a binary target as a dependency")
          if dep_type == "library":
              dlls += [dep.out]
              refs += [dep.name]
          if dep.transitive_dlls:
              transitive_dlls += dep.transitive_dlls
    return struct(
        dlls = dlls + set(extra_files),
        refs = refs,
        transitive_dlls = transitive_dlls)

def _get_libdirs(dlls, libdirs=[]):
    return [dep.dirname for dep in dlls] + libdirs

def _make_csc_arglist(ctx, output, depinfo, extra_refs=[]):
    flag_start = ctx.attr._flag_start
    args = [
         # /out:<file>
        _make_csc_flag(flag_start, "out", output.path),
         # /target (exe for binary, library for lib, module for module)
        _make_csc_flag(flag_start, "target", ctx.attr._target_type),
        # /fullpaths
        _make_csc_flag(flag_start, "fullpaths"),
        # /warn
        _make_csc_flag(flag_start, "warn", str(ctx.attr.warn)),
        # /nologo
        _make_csc_flag(flag_start, "nologo"),
    ]

    # /modulename:<string> only used for modules
    libdirs = _get_libdirs(depinfo.dlls)
    libdirs = _get_libdirs(depinfo.transitive_dlls, libdirs)
    # /lib:dir1,[dir1]
    args += [_make_csc_flag(flag_start, "lib", ",".join(list(libdirs)))] if libdirs else []
    # /reference:filename[,filename2]
    args += [_make_csc_flag(flag_start, "reference", ",".join(list(depinfo.refs + extra_refs)))] if depinfo.refs else extra_refs

    # /doc
    args += [_make_csc_flag(flag_start, "doc", ctx.outputs.doc_xml.path)] if hasattr(ctx.outputs, "doc_xml") else []
    # /debug
    debug = ctx.var.get("BINMODE", "") == "-dbg"
    args += [_make_csc_flag(flag_start, "debug")] if debug else []
    # /warnaserror
    # TODO(jeremy): /define:name[;name2]
    # TODO(jeremy): /resource:filename[,identifier[,accesibility-modifier]]
    # /main:class
    if hasattr(ctx.attr, "main_class") and ctx.attr.main_class:
        args += [_make_csc_flag(flag_start, "main", ctx.attr.main_class)]
    # TODO(jwall): /parallel
    return args

def _make_nunit_launcher(ctx, depinfo, output):
    content = """#!/bin/bash
cd $0.runfiles
# TODO(jeremy): This is a gross and fragile hack.
# We should be able to do better than this.
for l in {libs}; do
    ln -s -f $l $(basename $l)
done
/usr/local/bin/mono {nunit_exe} {libs} "$@"

"""
    libs = [d.short_path for d in depinfo.dlls]
    libs += [d.short_path for d in depinfo.transitive_dlls]

    content = content.format(
        nunit_exe=ctx.files._nunit_exe[0].path,
        libs=" ".join(libs))

    ctx.file_action(
        output=ctx.outputs.executable,
        content=content)

def _make_launcher(ctx, depinfo, output):
    content = """#!/bin/bash
cd $0.runfiles
# TODO(jeremy): This is a gross and fragile hack.
# We should be able to do better than this.
ln -s -f {exe} $(basename {exe})
for l in {libs}; do
    ln -s -f $l $(basename $l)
done
/usr/local/bin/mono $(basename {exe}) "$@"
"""
    libs = [d.short_path for d in depinfo.dlls]
    libs += [d.short_path for d in depinfo.transitive_dlls]

    content = content.format(
        exe=output.short_path,
        libs=" ".join(libs))

    ctx.file_action(
        output=ctx.outputs.executable,
        content=content)

def _csc_get_output(ctx):
    output = None
    if hasattr(ctx.outputs, "csc_lib"):
        output = ctx.outputs.csc_lib
    elif hasattr(ctx.outputs, "csc_exe"):
        output = ctx.outputs.csc_exe
    else:
        fail("You must supply one of csc_lib or csc_exe")
    return output

def _csc_collect_inputs(ctx, extra_files=[]):
    depinfo = _make_csc_deps(ctx.attr.deps, extra_files=extra_files)
    inputs = set(ctx.files.srcs) + depinfo.dlls + depinfo.transitive_dlls
    srcs = [src.path for src in ctx.files.srcs]
    return struct(
        depinfo=depinfo,
        inputs=inputs,
        srcs=srcs)

def _csc_compile_action(ctx, assembly, all_outputs, collected_inputs, extra_refs=[]):
    csc_args = _make_csc_arglist(ctx, assembly, collected_inputs.depinfo, extra_refs=extra_refs)
    command_script = " ".join([ctx.attr.csc] + csc_args + collected_inputs.srcs)

    ctx.action(
        inputs = list(collected_inputs.inputs),
        outputs = all_outputs,
        command = command_script,
        arguments = csc_args,
        progress_message = ("Compiling " +
                            ctx.label.package + ":" +
                            ctx.label.name))

def _cs_runfiles(ctx, outputs, depinfo):
    return ctx.runfiles(
        files = outputs,
        transitive_files = set(depinfo.dlls + depinfo.transitive_dlls) or None)


def _csc_compile_impl(ctx):
    if hasattr(ctx.outputs, "csc_lib") and hasattr(ctx.outputs, "csc_exe"):
        fail("exactly one of csc_lib and csc_exe must be defined")

    output = _csc_get_output(ctx)
    outputs = [output] + ([ctx.outputs.doc_xml] if hasattr(ctx.outputs, "doc_xml") else [])

    collected = _csc_collect_inputs(ctx)

    depinfo = collected.depinfo
    inputs = collected.inputs
    srcs = collected.srcs

    runfiles = _cs_runfiles(ctx, outputs, depinfo)

    _csc_compile_action(ctx, output, outputs, collected)

    if hasattr(ctx.outputs, "csc_exe"):
        _make_launcher(ctx, depinfo, output)

    return struct(name= ctx.label.name,
                  srcs = srcs,
                  target_type=ctx.attr._target_type,
                  out = output,
                  dlls = set([output]),
                  transitive_dlls = depinfo.dlls,
                  runfiles=runfiles)

def _cs_nunit_run_impl(ctx):
    if hasattr(ctx.outputs, "csc_lib") and hasattr(ctx.outputs, "csc_exe"):
        fail("exactly one of csc_lib and csc_exe must be defined")

    output = _csc_get_output(ctx)
    outputs = [output] + ([ctx.outputs.doc_xml] if hasattr(ctx.outputs, "doc_xml") else [])
    outputs = outputs

    collected_inputs = _csc_collect_inputs(ctx, ctx.files._nunit_framework)

    depinfo = collected_inputs.depinfo
    inputs = collected_inputs.inputs
    srcs = collected_inputs.srcs

    runfiles = _cs_runfiles(ctx, outputs + ctx.files._nunit_exe + ctx.files._nunit_exe_libs, depinfo)

    _csc_compile_action(ctx, output, outputs, collected_inputs, extra_refs=["Nunit.Framework"])

    _make_nunit_launcher(ctx, depinfo, output)

    return struct(name=ctx.label.name,
                  srcs=srcs,
                  target_type=ctx.attr._target_type,
                  out=output,
                  dlls = set([output]) if hasattr(ctx.outputs, "csc_lib") else None,
                  transitive_dlls = depinfo.dlls,
                  runfiles=runfiles)

_COMMON_ATTRS = {
    # configuration fragment that specifies
    "_flag_start": attr.string(default="-"),
    # where the csharp compiler is.
    "csc": attr.string(default=_MONO_UNIX_CSC),
    # code dependencies for this rule.
    # all dependencies must provide an out field.
    "deps": attr.label_list(providers=["out", "target_type"]),
    # source files for this target.
    "srcs": attr.label_list(allow_files = FileType([".cs", ".resx"])),
    # resources to use as dependencies.
    # TODO(jeremy): "resources_deps": attr.label_list(allow_files=True),
    #TODO(jeremy): # name of the module if you are creating a module.
    #TODO(jeremy): "modulename": attri.string(),
    # warn level to use
    "warn": attr.int(default=4),
    # define preprocessor symbols.
    #TODO(jeremy): "define": attr.string_list(),
}

_LIB_ATTRS={"_target_type": attr.string(default="library")}

_EXE_ATTRS={
    "_target_type": attr.string(default="exe"),
    # main class to use as entry point.
    "main_class": attr.string(),
}

_NUNIT_ATTRS={
    "_nunit_exe": attr.label(default=Label("@nunit//:nunit_exe"),
                             single_file=True),
    "_nunit_framework": attr.label(default=Label("@nunit//:nunit_framework")),
    "_nunit_exe_libs": attr.label(default=Label("@nunit//:nunit_exe_libs")),
}

_LIB_OUTPUTS={
    "csc_lib": "%{name}.dll",
    "doc_xml": "%{name}.xml",
}

_BIN_OUTPUTS={
    "csc_exe": "%{name}.exe",
}

csharp_library = rule(
    implementation=_csc_compile_impl,
    attrs=_COMMON_ATTRS + _LIB_ATTRS,
    outputs = _LIB_OUTPUTS,
)

csharp_binary = rule(
    implementation=_csc_compile_impl,
    attrs=_COMMON_ATTRS + _EXE_ATTRS,
    outputs = _BIN_OUTPUTS,
    executable=True)

csharp_nunit_test = rule(
    implementation=_cs_nunit_run_impl,
    executable=True,
    attrs=_COMMON_ATTRS + _LIB_ATTRS +_NUNIT_ATTRS,
    outputs = _LIB_OUTPUTS,
    test=True
)

NUNIT_BUILD_FILE = """
filegroup(
    name = "nunit_exe",
    srcs = ["NUnit-2.6.4/bin/nunit-console.exe"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nunit_exe_libs",
    srcs = glob(["NUnit-2.6.4/bin/lib/*.dll"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nunit_framework",
    srcs = glob(["NUnit-2.6.4/bin/framework/*.dll"]),
    visibility = ["//visibility:public"],
)
"""

def csharp_repositories():
  native.new_http_archive(
      name = "nunit",
      build_file_content = NUNIT_BUILD_FILE,
      sha256 = "1bd925514f31e7729ccde40a38a512c2accd86895f93465f3dfe6d0b593d7170",
      type = "zip",
      url = "https://github.com/nunit/nunitv2/releases/download/2.6.4/NUnit-2.6.4.zip",
  )
