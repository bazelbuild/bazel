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
"""FDO context describes how C++ FDO compilation should be done."""

load(":common/cc/fdo/fdo_prefetch_hints.bzl", "FdoPrefetchHintsInfo")
load(":common/cc/fdo/fdo_profile.bzl", "FdoProfileInfo")
load(":common/cc/fdo/memprof_profile.bzl", "MemProfProfileInfo")
load(":common/cc/fdo/propeller_optimize.bzl", "PropellerOptimizeInfo")
load(":common/paths.bzl", "paths")

cc_internal = _builtins.internal.cc_internal

def _create_fdo_context(
        ctx,
        *,
        llvm_profdata,
        all_files,
        zipper,
        cc_toolchain_config_info,
        coverage_enabled,
        _fdo_prefetch_hints,
        _propeller_optimize,
        _memprof_profile,
        _fdo_optimize,
        _fdo_profile,
        _xfdo_profile,
        _csfdo_profile,
        _proto_profile):
    """Creates FDO context when it should be available.

    When `-c opt` is used it parses values of FDO related flags, processes the
    input Files and return Files ready to be used for FDO.

    Args:
      ctx: (SubruleContext) used to create actions and obtain configuration
      llvm_profdata: (File) llvm-profdata executable
      all_files: (depset(File)) Files needed to run llvm-profdata
      zipper: (File) zip tool, used to unpact the profiles
      cc_toolchain_config_info: (CcToolchainConfigInfo) Used to check CPU value, should be removed
      coverage_enabled: (bool) Is code coverage enabled
      _fdo_prefetch_hints: (Target) Pointed to by --fdo_prefetch_hints
      _propeller_optimize: (Target) Pointed to by --propeller_optimize
      _memprof_profile: (Target) Pointed to by --memprof_profile
      _fdo_optimize: (Target) Pointed to by --fdo_optimize
      _fdo_profile: (Target) Pointed to by --fdo_profile
      _xfdo_profile: (Target) Pointed to by --xbinary_fdo
      _csfdo_profile: (Target) Pointed to by --cs_fdo_profile
      _proto_profile: (Target) Pointed to by --proto_profile_path
    Returns:
      (FDOContext) A structure with following fields:
      - branch_fdo_profile (struct|None)
         - branch_fdo_mode ("auto_fdo"|"xbinary_fdo"|"llvm_fdo"|llvm_cs_fdo")
         - profile_artifact (File)
      - proto_profile_artifact (File)
      - prefetch_hints_artifact (File),
      - propeller_optimize_info (PropellerOptimizeInfo)
      - memprof_profile_artifact (File)
    """
    cpp_config = ctx.fragments.cpp
    if cpp_config.compilation_mode() != "opt":
        return struct()

    # Propeller optimize cc and ld profiles
    cc_profile = _symlink_to(
        ctx,
        name_prefix = "fdo",
        absolute_path = cpp_config.propeller_optimize_absolute_cc_profile(),
        progress_message = "Symlinking LLVM Propeller Profile %{input}",
    )
    ld_profile = _symlink_to(
        ctx,
        name_prefix = "fdo",
        absolute_path = cpp_config.propeller_optimize_absolute_ld_profile(),
        progress_message = "Symlinking LLVM Propeller Profile %{input}",
    )
    if cc_profile or ld_profile:
        propeller_optimize_info = PropellerOptimizeInfo(
            cc_profile = cc_profile,
            ld_profile = ld_profile,
        )
    elif _propeller_optimize:
        propeller_optimize_info = _propeller_optimize[PropellerOptimizeInfo]
    else:
        propeller_optimize_info = None

    # Attempt to fetch the memprof profile input from an explicit flag or as part of the
    # fdo_profile rule. The former overrides the latter. Also handle the case where the
    # fdo_profile rule is specified using fdo_optimize.
    mem_prof_profile = None
    if _memprof_profile:
        mem_prof_profile = _memprof_profile[MemProfProfileInfo]
    elif _fdo_profile and _fdo_profile[FdoProfileInfo].memprof_artifact:
        mem_prof_profile = MemProfProfileInfo(artifact = _fdo_profile[FdoProfileInfo].memprof_artifact)
    elif _fdo_optimize and FdoProfileInfo in _fdo_optimize and _fdo_optimize[FdoProfileInfo].memprof_artifact:
        mem_prof_profile = MemProfProfileInfo(artifact = _fdo_optimize[FdoProfileInfo].memprof_artifact)

    fdo_inputs = None
    if cpp_config.fdo_path():
        # TODO(b/333997009): computation of cpp_config.fdo_path in CppConfiguration class is convoluted, simplify it
        # fdoZip should be set if the profile is a path, fdoInputFile if it is an artifact, but never both
        fdo_inputs = FdoProfileInfo(absolute_path = cpp_config.fdo_path())
    elif _fdo_optimize:
        if FdoProfileInfo in _fdo_optimize:
            fdo_inputs = _fdo_optimize[FdoProfileInfo]
        elif _fdo_optimize[DefaultInfo].files:
            if len(_fdo_optimize[DefaultInfo].files.to_list()) != 1:
                fail("--fdo_optimize does not point to a single target")
            [fdo_optimize_artifact] = _fdo_optimize[DefaultInfo].files.to_list()
            if fdo_optimize_artifact.short_path != _fdo_optimize.label.package + "/" + _fdo_optimize.label.name:
                fail("--fdo_optimize points to a target that is not an input file or an fdo_profile rule")
            fdo_inputs = FdoProfileInfo(artifact = fdo_optimize_artifact)
    elif _fdo_profile:
        fdo_inputs = _fdo_profile[FdoProfileInfo]
    elif _xfdo_profile:
        fdo_inputs = _xfdo_profile[FdoProfileInfo]

    cs_fdo_input = None
    if cpp_config.cs_fdo_path():
        cs_fdo_input = FdoProfileInfo(absolute_path = cpp_config.cs_fdo_path())
    elif _csfdo_profile:
        cs_fdo_input = _csfdo_profile[FdoProfileInfo]

    # If --noproto_profile is in effect, there is no proto profile.
    # If --proto_profile_path=<label> is passed, that profile is used.
    # If AutoFDO is not in effect, no profile is used.
    # If AutoFDO is in effect, a file called proto.profile next to the AutoFDO
    # profile is used, if it exists.
    proto_profile_artifact = None
    if not cpp_config.proto_profile() and _proto_profile:
        fail("--proto_profile_path cannot be set if --proto_profile is false")
    if _proto_profile:
        proto_profile_artifact = _symlink_to(
            ctx,
            name_prefix = "fdo",
            artifact = _proto_profile,
            progress_message = "Symlinking protobuf profile %{input}",
        )
    elif cpp_config.proto_profile():
        proto_profile_artifact = getattr(fdo_inputs, "proto_profile_artifact", None)

    branch_fdo_profile = None
    if fdo_inputs:
        branch_fdo_modes = {
            ".afdo": "auto_fdo",
            ".xfdo": "xbinary_fdo",
            ".profdata": "llvm_fdo",
            ".profraw": "llvm_fdo",
            ".zip": "llvm_fdo",
        }
        extension = paths.split_extension(_basename(fdo_inputs))[1]
        if extension not in branch_fdo_modes:
            fail("invalid extension for FDO profile file")
        branch_fdo_mode = branch_fdo_modes[extension]

        if branch_fdo_mode == "llvm_fdo":
            # Check if this is LLVM_CS_FDO
            if cs_fdo_input:
                branch_fdo_mode = "llvm_cs_fdo"

            if _xfdo_profile:
                fail("--xbinary_fdo only accepts *.xfdo and *.afdo")

        if coverage_enabled:
            fail("coverage mode is not compatible with FDO optimization")

        # This tries to convert LLVM profiles to the indexed format if necessary.
        if branch_fdo_mode == "llvm_fdo":
            profile_artifact = _convert_llvm_raw_profile_to_indexed(
                ctx,
                "fdo",
                fdo_inputs,
                llvm_profdata,
                all_files,
                zipper,
                cc_toolchain_config_info,
            )
        elif branch_fdo_mode in ["auto_fdo", "xbinary_fdo"]:
            profile_artifact = _symlink_input(
                ctx,
                "fdo",
                fdo_inputs,
                "Symlinking FDO profile %{input}",
            )
        else:  # branch_fdo_mode == "llvm_cs_fdo":
            non_cs_profile_artifact = _convert_llvm_raw_profile_to_indexed(
                ctx,
                "fdo",
                fdo_inputs,
                llvm_profdata,
                all_files,
                zipper,
                cc_toolchain_config_info,
            )
            cs_profile_artifact = _convert_llvm_raw_profile_to_indexed(
                ctx,
                "csfdo",
                cs_fdo_input,
                llvm_profdata,
                all_files,
                zipper,
                cc_toolchain_config_info,
            )
            profile_artifact = _merge_llvm_profiles(
                ctx,
                "mergedfdo",
                llvm_profdata,
                all_files,
                non_cs_profile_artifact,
                cs_profile_artifact,
                "MergedCS.profdata",
            )

        branch_fdo_profile = struct(
            branch_fdo_mode = branch_fdo_mode,
            profile_artifact = profile_artifact,
        )

    prefetch_hints_artifact = None
    if _fdo_prefetch_hints:
        prefetch_hints_artifact = _symlink_input(
            ctx,
            "fdo",
            _fdo_prefetch_hints[FdoPrefetchHintsInfo],
            "Symlinking LLVM Cache Prefetch Hints Profile %{input}",
        )

    memprof_profile_artifact = _get_mem_prof_profile_artifact(zipper, mem_prof_profile, ctx)

    return struct(
        branch_fdo_profile = branch_fdo_profile,
        prefetch_hints_artifact = prefetch_hints_artifact,
        propeller_optimize_info = propeller_optimize_info,
        memprof_profile_artifact = memprof_profile_artifact,
        proto_profile_artifact = proto_profile_artifact,
    )

def _convert_llvm_raw_profile_to_indexed(
        ctx,
        name_prefix,
        fdo_inputs,
        llvm_profdata,
        all_files,
        zipper,
        cc_toolchain_config_info):
    """This function checks the input profile format and converts it to the indexed format (.profdata) if necessary."""
    basename = _basename(fdo_inputs)
    if basename.endswith(".profdata"):
        return _symlink_input(ctx, name_prefix, fdo_inputs, "Symlinking LLVM Profile %{input}")

    if basename.endswith(".zip"):
        if not zipper:
            fail("Zipped profiles are not supported with platforms/toolchains before toolchain-transitions are implemented.")

        zip_profile_artifact = _symlink_input(ctx, name_prefix, fdo_inputs, "Symlinking LLVM ZIP Profile %{input}")

        # TODO(b/333997009): find a way to avoid hard-coding cpu architecture here
        cpu = cc_toolchain_config_info.target_cpu()
        if "k8" == cpu:
            raw_profile_file_name = name_prefix + "/" + ctx.label.name + "/" + "fdocontrolz_profile.profraw"
        else:
            raw_profile_file_name = name_prefix + "/" + ctx.label.name + "/" + "fdocontrolz_profile-" + cpu + ".profraw"

        raw_profile_artifact = ctx.actions.declare_file(raw_profile_file_name)

        # We invoke different binaries depending on whether the unzip_fdo tool
        # is available. When it isn't, unzip_fdo is aliased to the generic
        # zipper tool, which takes different command-line arguments.

        args = ctx.actions.args()
        if zipper.path.endswith("unzip_fdo"):
            args.add("--profile_zip", zip_profile_artifact)
            args.add("--cpu", cpu)
            args.add("--output_file", raw_profile_artifact)
        else:
            args.add("xf", zip_profile_artifact)
            args.add("-d", raw_profile_artifact.dirname)
        ctx.actions.run(
            mnemonic = "LLVMUnzipProfileAction",
            executable = zipper,
            arguments = [args],
            inputs = [zip_profile_artifact],
            outputs = [raw_profile_artifact],
            progress_message = "LLVMUnzipProfileAction: Generating %{output}",
        )
    else:  # .profraw
        raw_profile_artifact = _symlink_input(ctx, name_prefix, fdo_inputs, "Symlinking LLVM Raw Profile %{input}")

    if not llvm_profdata:
        fail("llvm-profdata not available with this crosstool, needed for profile conversion")

    name = name_prefix + "/" + ctx.label.name + "/" + paths.replace_extension(basename, ".profdata")
    profile_artifact = ctx.actions.declare_file(name)
    ctx.actions.run(
        mnemonic = "LLVMProfDataAction",
        executable = llvm_profdata,
        tools = [all_files],
        arguments = [ctx.actions.args().add("merge").add("-o").add(profile_artifact).add(raw_profile_artifact)],
        inputs = [raw_profile_artifact],
        outputs = [profile_artifact],
        use_default_shell_env = True,
        progress_message = "LLVMProfDataAction: Generating %{output}",
    )

    return profile_artifact

def _merge_llvm_profiles(
        ctx,
        name_prefix,
        llvm_profdata,
        all_files,
        profile1,
        profile2,
        merged_output_name):
    """This function merges profile1 and profile2 and generates merged_output."""
    profile_artifact = ctx.actions.declare_file(name_prefix + "/" + ctx.label.name + "/" + merged_output_name)

    # Merge LLVM profiles.
    ctx.actions.run(
        mnemonic = "LLVMProfDataMergeAction",
        executable = llvm_profdata,
        tools = [all_files],
        arguments = [ctx.actions.args().add("merge").add("-o").add(profile_artifact).add(profile1).add(profile2)],
        inputs = [profile1, profile2],
        outputs = [profile_artifact],
        use_default_shell_env = True,
        progress_message = "LLVMProfDataAction: Generating %{output}",
    )
    return profile_artifact

def _basename(inputs):
    if getattr(inputs, "artifact", None):
        return inputs.artifact.basename
    else:
        return paths.basename(inputs.absolute_path)

def _symlink_input(ctx, name_prefix, fdo_inputs, progress_message, basename = None):
    return _symlink_to(
        ctx,
        name_prefix,
        progress_message,
        getattr(fdo_inputs, "artifact", None),
        getattr(fdo_inputs, "absolute_path", None),
        basename,
    )

def _symlink_to(ctx, name_prefix, progress_message, artifact = None, absolute_path = None, basename = None):
    """Symlinks either an absolute path or file to a unique location."""
    if artifact:
        if not basename:
            basename = artifact.basename
        name = name_prefix + "/" + ctx.label.name + "/" + basename
        output = ctx.actions.declare_file(name)
        ctx.actions.symlink(
            output = output,
            target_file = artifact,
            progress_message = progress_message,
        )
        return output
    elif absolute_path:
        if not basename:
            basename = paths.basename(absolute_path)
        name = name_prefix + "/" + ctx.label.name + "/" + basename
        output = ctx.actions.declare_file(name)
        cc_internal.absolute_symlink(
            ctx = ctx,
            output = output,
            target_path = absolute_path,
            progress_message = progress_message,
        )
        return output
    else:
        return None

def _get_mem_prof_profile_artifact(zipper, memprof_profile, ctx):
    """This function symlinks the memprof profile (after unzipping as needed)."""
    if not memprof_profile:
        return None

    basename = _basename(memprof_profile)

    # If the profile file is already in the desired format, symlink to it and return.
    if basename.endswith(".profdata"):
        return _symlink_input(ctx, "memprof", memprof_profile, "Symlinking MemProf Profile %{input}", basename = "memprof.profdata")

    if not basename.endswith(".zip"):
        fail("Expected zipped memprof profile.")

    if not zipper:
        fail("Zipped profiles are not supported with platforms/toolchains before " +
             "toolchain-transitions are implemented.")

    zip_profile_artifact = _symlink_input(ctx, "memprof", memprof_profile, "Symlinking MemProf ZIP Profile %{input}")

    profile_artifact = ctx.actions.declare_file("memprof/" + ctx.label.name + "/memprof.profdata")

    # We invoke different binaries depending on whether the unzip_fdo tool
    # is available. When it isn't, unzip_fdo is aliased to the generic
    # zipper tool, which takes different command-line arguments.
    args = ctx.actions.args()
    if zipper.path.endswith("unzip_fdo"):
        args.add("--profile_zip", zip_profile_artifact)
        args.add("--memprof")
        args.add("--output_file", profile_artifact)
    else:
        args.add("xf", zip_profile_artifact)
        args.add("-d", profile_artifact.dirname)
    ctx.actions.run(
        mnemonic = "LLVMUnzipProfileAction",
        executable = zipper,
        arguments = [args],
        inputs = [zip_profile_artifact],
        outputs = [profile_artifact],
        progress_message = "MemProfUnzipProfileAction: Generating %{output}",
    )
    return profile_artifact

create_fdo_context = subrule(
    implementation = _create_fdo_context,
    fragments = ["cpp"],
    attrs = {
        "_fdo_optimize": attr.label(
            default = configuration_field(fragment = "cpp", name = "fdo_optimize"),
            allow_files = True,
            providers = [[DefaultInfo], [FdoProfileInfo]],
        ),
        "_xfdo_profile": attr.label(
            default = configuration_field(fragment = "cpp", name = "xbinary_fdo"),
            providers = [FdoProfileInfo],
        ),
        "_fdo_profile": attr.label(
            default = configuration_field(fragment = "cpp", name = "fdo_profile"),
            providers = [FdoProfileInfo],
        ),
        "_csfdo_profile": attr.label(
            default = configuration_field(fragment = "cpp", name = "cs_fdo_profile"),
            providers = [FdoProfileInfo],
        ),
        "_fdo_prefetch_hints": attr.label(
            default = configuration_field(fragment = "cpp", name = "fdo_prefetch_hints"),
            providers = [FdoPrefetchHintsInfo],
        ),
        "_propeller_optimize": attr.label(
            default = configuration_field(fragment = "cpp", name = "propeller_optimize"),
            providers = [PropellerOptimizeInfo],
        ),
        "_memprof_profile": attr.label(
            default = configuration_field(fragment = "cpp", name = "memprof_profile"),
            providers = [MemProfProfileInfo],
        ),
        "_proto_profile": attr.label(
            default = configuration_field(fragment = "cpp", name = "proto_profile_path"),
            allow_single_file = True,
        ),
    },
)
