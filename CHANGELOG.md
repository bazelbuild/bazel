## Release 0.4.3 (2016-12-22)

```
Baseline: c645a45

Cherry picks:
   + af878d0: Add coverage support for java test. (series 4/4 of
              open-sourcing coverage command for java test)
   + 09b92a8: Rollback of commit
              67b4d5250edcefa7220e928e529b1f385e2dc464.
   + b11dd48: Fix bad bug with the parallel implementation of
              BinaryOperatorExpression. Turns out that
              ForkJoinTask#adapt(Callable) returns a ForkJoinTask
              whose Future#get on error throws a ExecutionException
              wrapping a RuntimeException wrapping the thrown checked
              exception from the callable. This is documented
              behavior [1] that I incorrectly didn't know about.
   + 9012bf1: Fix scripts/packages/convert_changelog to read the
              changelog correctly
   + 55c97bc: Release script: if master branch does not exist, fall
              back on origin/master
   + 4fb378c: Debian repository: override section and priority fields
   + acbcbc2: Fix release notes in emails
   + 4975760: Fix PathFragment to not use Java8-only static hashCode
              methods.
   + 05fd076: Disable sandboxing for XibCompile actions.
```

Incompatible changes:

  - Skylark maven_jar and maven_aar settings attribute is now a label
    so it can be checked into your workspace.
  - --{no}experimental_use_rclass_generator is now a nop.

New features:

  - Coverage support (*experimental*) for pure Java target.
    Use `bazel coverage //my:target` to generate coverage information
    from a `java_test`.

Important changes:

  - Enable fallback URLs in Skylark http rules.
  - cc_proto_library generates C++ code from proto_library rules.
  - cc_library now supports the strip_prefix and strip_include_prefix
    attributes for control of include paths.
  - Skylark dicts internally don't rely on keys order anymore and
    accept any hashable values (i.e. structs with immutable values)
    as keys. Iteration order of dictionaries is no longer specified.

## Release 0.4.2 (2016-12-02)

```
Baseline: 6331a94

Cherry picks:
   + 7b835d9: Do not patch WORKSPACE in the release process
```

Incompatible changes:

  - Callback functions in Skylark no longer support the cfg
    parameter. This is a cleanup and only affects the signatures of
    callbacks, since the parameter hasn't been set since September
    2016.

Important changes:

  - Alias proto_library's produce a descriptor set that contains all
    srcs of its dependencies.
  - proto_library supports strict proto deps.
  - Top level @androidsdk support library targets have been replaced
    by @androidsdk//<group id>:<artifact id>-<version> for Android
    SDK Support and Google Play Services libraries.

## Release 0.4.1 (2016-11-21)

```
Baseline: 9a796de

Cherry picks:
   + 88bfe85: Description redacted. -- MOS_MIGRATED_REVID=139219934
   + b09ea94: Rollback of commit
              a3f5f576cd35798140ba3e81d03d919dd4ecb847.
```

New features:

  - android_library now has a "exported_plugins" attribute just like
    java_library
  - Use --strict_system_includes to apply hdrs_check=strict also to
        cc_library.includes, even if sandboxing is disabled.
  - Bazel on Windows: java_binary can now be the executable of
    Skylark rule actions (ctx.action's executable argument)
  - Packages are defined in BUILD.bazel as well as BUILD files.

Important changes:

  - getattr()'s 3-arg form no longer raises an error when the
    retrieved field is a built-in method.
  - --apk_signing_method default changed to v1. Android APKs are now
    signed with the new ApkSignerTool by default.
  - New rule: proto_lang_toolchain(), to support LANG_proto_library
    rules on multiple platforms.
  - Fix for Android clang++ std::stack segfault on 32bit x86. See
    https://code.google.com/p/android/issues/detail?id=220159
  - Default android_manifest_merger is now "android" which uses the
    official Android manifest merger.
    http://tools.android.com/tech-docs/new-build-system/user-guide/man
    ifest-merger
  - Do not propagate aspect to its own attributes when using '*'.
  - Comparing sets (`if set1 < set2:`) is not allowed anymore in
    Skylark because it didn't work correctly anyway.
  - When --experimental_extra_action_top_level_only, Bazel reports
    extra-actions for actions registered by Aspects injected by a
    top-level rule (approximately).
  - Blacklists for proto_lang_toolchain() no longer have to be
    proto_library's.
  - Extra actions now contain aspect-related information.
  - Fix slicing bug where "abc"[:-4:-1] would give wrong answer

## Release 0.4.0 (2016-10-26)

```
Baseline: 088bbc6

Cherry picks:
   + b01160c: Stamp Windows release.
   + 2d6736e: Add --no-tty for gpg signing
   + 9b1dfb8: Remove .sig file before gpg signing
   + 81aede1: Reimplement whole archive on Windows
```

Incompatible changes:

  - Skylark: updating list/dicts while they are being looped over is not
    allowed. Use an explicit copy if needed ("for x in list(mylist):").
  - Bazel now uses the --cpu flag to look up Jvms; it falls back
    to "default" if it can't find a Jvm matching the CPU value.
  - --command_port=-1 to use AF_UNIX for client/server communications
    is not supported anymore.
  - Sandboxed actions can access the network by default, unless their
    target has a "block-network" tag.

New features:

  - Files now have an "extension" property in Skylark.

Important changes:

  - Added a new flag --sandbox_tmpfs_path, which asks the sandbox to
    mount an empty, writable directory at a specified path when
    running actions. (Supported on Linux only for now.)
  - Update protoc-3.0.0-mingw.exe to a working (statically linked)
    binary
  - apple_static_library rule to create multi-architecture static
    archive files from Objc/C++/Swift dependencies on apple platforms
  - JS: Add support for localization with closure managed rules.
  - Create a flag --android_dynamic_mode to turn off dynamic mode
    during the Android split transition.
  - Darwin sandboxing is default.
  - Remove flag --experimental_zip_tree_artifact from j2objc Java
    annotation processing support.
  - A few functions are added to BUILD files for consistency (hash,
    dir,
      hasattr, getattr) with .bzl files, although they are not very
    useful.
  - --watchfs is now a command option; the startup option of the same
        name is deprecated. I.e., use bazel build --watchfs, not
    blaze --watchfs
        build.

## Release 0.3.2 (2016-10-07)

```
Baseline: 023a7bd

Cherry picks:
   + bebbbe5: Fix dependency on libtool's helper script
              make_hashed_objlist.py.
   + 8a0d45f: Add the version information to the bazel.exe file
   + 2bc0939: Allow new_ rules to overwrited BUILD files in
              downloaded repos
   + c5545fd: Rollback of commit
              96d46280bc5a4803ba2242a4ad16939f85a3b212.
   + eb87208: Make cc_configure on Windows more robust
   + c30432c: Fix cc_configure on Windows
   + 95b16a8: sandbox: Replace the error-prone lazy cleanup of
              sandbox directories by a simple synchronous cleanup.
   + e898023: Fix #1849: Sandboxing on OS X should be turned off by
              default for 0.3.2.
   + ffdc05d: Add action_config and feature for linking on Windows
```

Incompatible changes:

  - If you maintain a rule that uses persistent workers, you'll have
    to specify execution_requirements={"supports-workers": 1} in the
    ctx.action that intends to run a tool with workers. The
    WorkerSpawnStrategy will alert you with a warning message if you
    forget to make this change and fallback to non-worker based
    execution.
  - It is now an error to include a precompiled library (.a, .lo, .so)
    in a cc_library which would generate a library with the same name
    (e.g., libfoo.so in cc_library foo) if that library also contains
    other linkable
    sources.
  - The main repository's execution root is under the main
    repository's workspace name, not the source directory's basename.
    This shouldn't
    have any effect on most builds, but it's possible it could break
    someone doing
    weird things with paths in actions.
  - Blaze doesn't support Unix domain sockets for communication
    between its client and server anymore. Therefore, the
    --command_port command line argument doesn't accept -1 as a valid
    value anymore.
  - Skylark: It is an error to shadow a global variable with a local
    variable after the global has already been accessed in the
    function.
  - bin_dir and genfiles_dir are now properties of ctx, not
    configuration. That is, to access the bin or genfiles directory
    from a
    Skylark rule, use ctx.bin_dir or ctx.genfiles_dir (not
    ctx.configuration.{bin,genfiles}_dir).  At the moment, you can
    access
    {bin,genfiles}_dir from either, but the ctx.configuration version
    will
    stop working in a future release.
  - filegroup-based C++ toolchains are not supported anymore.
    --*_crosstool_top options must always point to a
    cc_toolchain_suite rule (or an alias of one).
  - repository_ctx.{download,download_and_extract,execute} API now use
                   named parameters for optional parameters and no
    longer uses argument
                   type to distinguished between arguments
    (executable attribute name
                   must be specified when preceding optional
    arguments are missing).

New features:

  - print and fail are now available in BUILD files.

Important changes:

  - Added @bazel_tools//tools/build_defs/repo/git.bzl as a Skylark
    rule for Git repositories.
  - Added @bazel_tools//tools/build_defs/repo/maven_rules.bzl as a
    Skylark rule for Maven repositories.
  - Add global hash() function for strings (only)
  - Improve Android split transition handling.
  - Removes exports_manifest attribute from android_binary rule.
  - java_proto_library: control strict-deps through a rule-level and
    a package-level attribute.
  - Persistent workers are now used by default for Java compilation
    in Bazel, which should speed up your Java builds by ~4x. You can
    switch back to the old behavior via --strategy=Javac=standalone.
    Check out http://www.bazel.io/blog/2015/12/10/java-workers.html
    for more details.
  - objc_* rules can now depend on any target that returns an "objc"
    provider.
  - Adds support for NDK12 to `android_ndk_repository` rule in Bazel.
  - Test targets can disable the JUnit4 test security manager via a
    property.
  - Disable the Android split transition if --android_cpu and
    fat_apk_cpu are both empty.
  - New sandboxing implementation for Linux in which all actions run
    in a separate execroot that contains input files as symlinks back
    to the originals in the workspace. The running action now has
    read-write access to its execroot and /tmp only and can no longer
    write in arbitrary other places in the file system.
  - Add worker support to single jar.
  - Invoke source jar action as a worker.
  - Sandboxed builds allow network access for builds by default.
    Tests will still be run without networking, unless
    "requires-network" is specified as a tag.
  - Add path.realpath() method for Skylark repositories.
  - On Mac devices, detect locally installed versions of xcode to:
     1. Use a sensible default if xcode is required but
    --xcode_version is unspecified.
     2. Use sensible default iOS SDK version for the targeted version
    of xcode if ios_sdk_version is unspecified.
  - Emacs' [C-x `], a.k.a. next-error, works again in emacsen >= 25.1
  - swift_library can be used to build watchOS apps.
  - Exposes the is_device field on Apple platform objects and adds
    the apple_common.platform_type(name) method to retrieve a
    platform_type value that can be passed to methods like the Apple
    fragment's multi_arch_platform.
  - Move Skylark git_repository rules to git.bzl
  - Add support for aspects to attr.label() attributes
  - Global varaiables HOST_CFG and DATA_CFG are deprecated in favor
    of strings "host"
    and "data.
    Argument `cfg = "host"` or `cfg = "data"` is mandatory if
    `executable = True` is provided for a label.
  - The deprecation attribute of all rules now causes warnings
    to be printed when other targets depend on a target with that
    attribute set.
  - Change default of --[no]instrument_test_targets to false, change
    default --instrumentation_filter (which previously tried to
    exclude test targets by heuristic) to only exclude targets in
    javatests.
  - Remove deprecated absolute paths in blaze IDE artifacts
  - When using android_binary.manifest_merger="android" the merger
    produces a summary log next to the merged manifest artifact.
  - Allow different default mallocs per configuration.

## Release 0.3.1 (2016-07-29)

```
Baseline: 792a9d6

Cherry picks:
   + 25e5995: Rollback of commit
              a2770334ea3f3111026eb3e1368586921468710c.
   + 2479405: Fix NPE with unset maven_jar sha1
   + 3cf2126: Rewrite the extra action info files if the data within
              them changes.
   + 5a9c6b4: JavaBuilder: Reintroduce the -extra_checks flag.
```

Incompatible changes:

  - Removed predefined Python variable "generic_cpu".
  - Skylark rules: if you set "outputs" or an attribute to a
    function, this function must now list its required attributes as
    parameters (instead of an attribute map).
  - The host_platform and target_platform entries are not written to
    the master log anymore.
  - Bazel requires Hazelcast 3.6 or higher now for remote execution
    support, because we upgraded our client library and the protocol
    it uses is incompatible with older versions.

New features:

  - LIPO context (--lipo_context) can now also be a cc_test (in
    addition to cc_binary)

Important changes:

  - If --android_crosstool_top is set, native code compiled for
    android will always use --android_compiler and not --compiler in
    choosing the crosstool toolchain, and will use --android_cpu if
    --fat_apk_cpu is not set.
  - Add --instrument_test_targets option.
  - apple_binary supports a new platform_type attribute, which, if
    set to "watchos", will build dependencies for Apple's watchOS2.
  - objc_binary now supports late-loaded dynamic frameworks.
  - Native Swift rules no longer pull in module maps unconditionally.
    Use --experimental_objc_enable_module_maps for that.
  - Merged manifests are guaranteed to have the application element
    as the last child of the manifest element as required by Android
    N.
  - The Android manifest merger is now available as an option for
    android_binary rules. The merger will honor tools annotations in
    AndroidManifest.xml and will perform placeholder substitutions
    using the values specified in android_binary.manifest_values. The
    merger may be selected by setting the manifest_merger attribute
    on android_binary.
  - The progress message would not clear packages that need to be
    loaded twice.
  - Remove warning for high value of --jobs.
  - Use the correct build configuration for shared native deps during
    Android split transitions.
  - When building ObjectiveC++, pass the flag -std=gnu++11.
  - use xcrun simctl instead of iossim to launch the app for "blaze
    run".
  - Glob arguments 'exclude' and 'exclude_directories' must be named
  - Bazel no longer regards an empty file as changed if its mtime has
    changed.

## Release 0.3.0 (2016-06-10)

```
Baseline: a9301fa

Cherry picks:
   + ff30a73: Turn --legacy_external_runfiles back on by default
   + aeee3b8: Fix delete[] warning on fsevents.cc
```

Incompatible changes:

  - The --cwarn command line option is not supported anymore. Use
    --copt instead.

New features:

  - On OSX, --watchfs now uses FsEvents to be notified of changes
    from the filesystem (previously, this flag had no effect on OS X).
  - add support for the '-=', '*=', '/=', and'%=' operators to
    skylark.  Notably, we do not support '|=' because the semantics
    of skylark sets are sufficiently different from python sets.

Important changes:

  - Use singular form when appropriate in blaze's test result summary
    message.
  - Added supported for Android NDK revision 11
  - --objc_generate_debug_symbols is now deprecated.
  - swift_library now generates an Objective-C header for its @objc
    interfaces.
  - new_objc_provider can now set the USES_SWIFT flag.
  - objc_framework now supports dynamic frameworks.
  - Symlinks in zip files are now unzipped correctly by http_archive,
    download_and_extract, etc.
  - swift_library is now able to import framework rules such as
    objc_framework.
  - Adds "jre_deps" attribute to j2objc_library.
  - Release apple_binary rule, for creating multi-architecture
    ("fat") objc/cc binaries and libraries, targeting ios platforms.
  - Aspects documentation added.
  - The --ues_isystem_for_includes command line option is not
    supported anymore.
  - global function 'provider' is removed from .bzl files. Providers
    can only be accessed through fields in a 'target' object.

## Release 0.2.3 (2016-05-10)

```
Baseline: 5a2dd7a
```

Incompatible changes:

  - All repositories are now directly under the x.runfiles directory
    in the runfiles tree (previously, external repositories were at
    x.runfiles/main-repo/external/other-repo. This simplifies
    handling remote repository runfiles considerably, but will break
    existing references to external repository runfiles.
    Furthermore, if a Bazel project does not provide a workspace name
    in the WORKSPACE file, Bazel will now default to using __main__
    as the workspace name (instead of "", as previously). The
    repository's runfiles will appear under x.runfiles/__main__/.
  - Bazel does not embed protocol buffer-related rules anymore.
  - It is now an error for a cc rule's includes attribute to point to
    the workspace root.
  - Bazel warns if a cc rule's includes attribute points out of
    third_party.
  - Removed cc_* attributes: abi / abi_deps. Use select() instead.

New features:

  - select({"//some:condition": None }) is now possible (this "unsets"
    the attribute).

Important changes:

  - java_import now allows its 'jars' attribute to be empty.
  - adds crunch_png attribute to android_binary
  - Replace --java_langtools, --javabuilder_top, --singlejar_top,
    --genclass_top, and --ijar_top with
    java_toolchain.{javac,javabuilder,singlejar,genclass,ijar}
  - External repository correctness fix: adding a new file/directory
    as a child of a new_local_repository is now noticed.
  - iOS apps are signed with get-task-allow=1 unless building with -c
    opt.
  - Generate debug symbols (-g) is enabled for all dbg builds of
    objc_ rules.
  - Bazel's workspace name is now io_bazel. If you are using Bazel's
    source as an external repository, then you may want to update the
    name you're referring to it as or you'll begin seeing warnings
    about name mismatches in your code.
  - Fixes integer overflow in J2ObjC sources to be Java-compatible.
  - A FlagPolicy specified via the --invocation_policy flag will now
    match the current command if any of its commands matches any of
    the commands the current command inherits from, as opposed to
    just the current command.
  - The key for the map to cc_toolchain_suite.toolchains is now a
    string of the form "cpu|compiler" (previously, it was just "cpu").
  - Fix interaction between LIPO builds and C++ header modules.
  - Ctrl-C will now interrupt a download, instead of waiting for it to
    finish.
  - Proxy settings can now be specified in http_proxy and https_proxy
    environment variables (not just HTTP_PROXY and HTTPS_PROXY).
  - Skylark targets can now read include directories from
    ObjcProvider.
  - Expose parameterized aspects to Skylark.
  - Support alwayslink cc_library dependencies in objc binaries.
  - Import cc_library dependencies in generated Xcode project.

## Release 0.2.2b (2016-04-22)

```
Baseline: 759bbfe

Cherry picks:
   + 1250fda: Rollback of commit
              351475627b9e94e5afdf472cbf465f49c433a25e.
   + ba8700e: Correctly set up build variables for the correct pic
              mode for fake_binary rules.
   + 386f242: Automated [] rollback of commit
              525fa71b0d6f096e9bfb180f688a4418c4974eb4.
   + 97e5ab0: Fix cc_configure include path for Frameworks on OS X.
   + a20352e: cc_configure: always add -B/usr/bin to the list of gcc
              option
   + 0b26f44: cc_configure: Add piii to the list of supported
              cpu_value
   + 3e4e416: cc_configure: uses which on the CC environment variable
   + aa3dbd3: cc_configure.bzl: strip end of line when looking for
              the cpu
   + 810d60a: cc_configure: Add -B to compiler flag too
```

Patch release, only includes fixes to C++ auto-configuration.

## Release 0.2.1 (2016-03-21)

```
Baseline: 19b5675
```

Incompatible changes:

  - Skylark rules that are available from their own repository will
    now issue a warning when accessed through @bazel_tools.
  - Set --legacy_bazel_java_test to off by default. java_test will
    now have a slightly different behaviour, correctly emitting XML
    file but, as a downside, it needs correct declaration of the
    test suite (see https://github.com/bazelbuild/bazel/issues/1017).
  - Labels in .bzl files in remote repositories will be resolved
    relative to their repository (instead of the repository the
    Skylark rule is used in).
  - Renamed proto_java_library to java_proto_library.  The former
    is now deprecated and will print out a warning when used.
  - android_sdk now compiles android_jack on the fly from
    android_jar, which means android_jar must be a jar and
    android_jack is now deprecated. The Jack tools (jack, jill,
    resource_extractor) must be specified.
  - Any project that depended on the objc_options rule will be
    broken. Can be fixed by adding attrs (infoplists,copts) directly
    to rules depending on the options.
  - .aidl files correctly require import statements for types
    defined in the same package and the same android_library.

New features:

  - Experimental Windows support is available.
  - Experimental support for writing remote repository rules in
    Skylark is available.
  - iOS ipa_post_processor attribute allows for user-defined IPA
    edits.
  - Adds a to_json method to Skylark structs, providing conversion to
    JSON format.
  - Native python rule can depend on skylark rule as long as skylark
    rule provides 'py' provider.
  - When using both --verbose_failures and --sandbox_debug, Bazel
    prints instructions how to spawn a debugging shell inside the
    sandbox.
  - add flag --sandbox_add_path, which takes a list of additional
    paths as argument and mount these paths to sandbox.

Important changes:

  - @androidsdk//:org_apache_http_legacy added for the legacy Apache
    classes for android sdk version 23 and above.
  - Genrules correctly work when used with bazel run.
  - When namespace-sandbox is run with the -D (debug) flag and
    inside a terminal, it spawns a shell inside the sandbox to aid in
    debugging when the sandboxed command fails.
  - Added --artifact to workspace generator for generating workspace
    and build file rules from artifact coodrinates.
  - Specifying --experimental_android_resource_shrinking on the
    command line will enable a resource shrinking pass on
    android_binary targets that already use Proguard.
  - J2ObjC updated to 1.0.1 release.
  - Added "root_symlinks" and "symlinks" parameters to Skylark
    runfiles() method.
  - You can no longer use objc_binary targets for the xctest_app
    attribute of an ios_test rule.
  - Enable overriding jsonnet binaries and stdlib for Jsonnet rules.
  - mount target of /etc/resolv.conf if it is a symlink.
  - Tests that failed to build because execution was halted no longer
    print their status.
  - Bazel warns if a cc rule's includes attribute contains up-level
    references that escape its package.
  - Add repository_ctx.download and repository_ctx.download_and_extract
    function.

## Release 0.2.0 (2016-02-18)

```
Baseline: 9e100ac
```

Incompatible changes:

  - ObjC compile actions for J2ObjC-translated code now only has
    access to headers from the java deps of the associated original
    java rule.
    These compile actions no longer takes the compiler options
    specified in "copts" attribute on objc_binary/ios_test rules.
    J2ObjC dead code removal (enabled through flag
    "--j2objc_dead_code_removal") now happens *after* ObjC
    compilation.
  - maven_jar no longer supports separate artifact_id, group_id, and
    verison fields. This information should be provided in the
    artifact field,
    instead.

New features:

  - Better support for toolchains that don't have a dynamic linker.
  - build_file_content attribute added to new_git_repository,
    new_http_archive, and new_local_repository.
  - Add support for .tar.bz2 archives to http_archive rules.

Important changes:

  - The --skyframe flag is no longer available for the build command.
  - The --artifacts flag was removed from the dump command.
  - The sha256 attribute is now optional (although recommended!) for
    remote repository rules.
  - Add instrumented file provider support to Skylark rules.
  - Add imports attribute to native Python rules.
  - Allow overriding -gsplit-dwarf from copts.
  - Improved sandbox performance on XFS filesystems.

## Release 0.1.5 (2016-02-05)

```
Baseline: 3a95f35
   + 8378cd8: Rollback of commit
              a9b84575a32476a5faf991da22b44661d75c19b6.
```

Incompatible changes:

  - Set stamping to false by default (i.e., --nostamp)
  - Removed --objc_dump_syms_binary.
  - Removes --objc_gcov_binary flag.
  - Remove JAVAC "Make" variable
  - The startup flag --blaze_cpu is removed,

New features:

  - A new java test runner that support XML output and test filtering
    is supported. It can be used by specifying --nolegacy_bazel_java_test
    or by specifying the test_class attribute on a java_test.
  - Skylark aspects can now specify configuration fragment
    dependencies with fragments and host_fragments like rules can.

Important changes:

  - Support for downloading remote resources through proxies by
    setting HTTP_PROXY (or HTTPS_PROXY).
  - Timestamps within Android apks are removed to make apks
    deterministic.
  - Support aggregation over existing rules in Skylark extensions
    through native.rules and native.rule.
  - A tools/bazel script in the workspace will be executed
    as an opportunity to use a fixed version of Bazel (not
    implemented for the homebrew recipe yet).
  - --noimplicit_deps and --nohost_deps work correctly for Aspect
    attributes.
  - JDK-related targets are now available via @local_jdk (instead of
    @local-jdk).
  - j2objc tools can now be accessed via @bazel_j2objc, not
    @bazel-j2objc.
  - Repository rules must use names that are valid workspace names.
  - [rust] Update to Rust 1.6
  - Add support for .tar.xz archives to http_archive rules.
  - Make C++ modules compatible with tools using
    --compilation_prerequisites_only
  - [d] Update to DMD 2.070.0

## Release 0.1.4 (2016-01-15)

```
Baseline: e933d5e
   + 3d796fe: Rollback of commit
              ac6ed79e1a3fa6b0ca91657b28e2a35f7e49758c.
   + 7a02e5d: Fix installer under OS X
   + 848740c: Fix bazel version for debian package
   + 7751d43: Add a method for getting the root of a rule workspace
              to the Label method
```

Important changes:

  - add loadfiles() query operator, to find skylark files loaded by
    targets.
  - Added ability to declare and use aspects in Skylark.
  - Skylark load statements may now reference .bzl files via build
    labels, in addition to paths. In particular, such labels can be
    used to reference Skylark files in external repositories; e.g.,
    load("@my_external_repo//some_pkg:some_file.bzl", ...).
    Path-based loads are now deprecated and may be disabled in the
    future. Caveats: Skylark files currently do not respect package
    visibility; i.e., all Skylark files are effectively public. Also,
    loads may not reference the special //external package.
  - Relative paths can now be used for 'path' with
    new_local_repository and local_repository.

## Release 0.1.3 (2016-01-07)

```
Baseline: 23ad8f6
   + de2183d: Only depend on the WORKSPACE file for external files
              that are under the external/ directory, i.e. were
              created by Bazel.
   + f8f855c: Rollback of commit
              12bad3af0eade9c4b79d76f9e1c950ad2e3214c2.
   + f627562: Stop parsing the WORKSPACE file when a parse error is
              detected
   + 763f139: Add -fno-canonical-system-headers to CROSSTOOL files so
              that gcc doesn't resolve symlinks in .d files, which
              would confuse Blaze.
   + b95995b: Use openjdk7 as dependency for debian package of jdk7
              flavor
```

New features:

  - Skylark macros are now enabled in WORKSPACE file.
  - .bazelrc allows workspace-relative imports as "import
    %workspace%/path/to/rcfile"
  - Evaluate the query expression in a file by passing
    --query_file=<file> to query

Important changes:

  - Remove obsolete --objc_per_proto_includes flag.
  - iOS apps and extensions now have launch_storyboard
  - Passing multiple JVM options via a single --host_jvm_args flag is
    now deprecated. Pass each JVM option behind its own
    --host_jvm_args flag.
  - Resources defined locally on an android_library rule will respect
    the neverlink attribute.
  - Update Rust to 1.4
  - Fix resource handling for exported android_library rules
  - Files in external repositories are now treated as mutable, which
    will make the correctness guarantees of using external
    repositories stronger (existent), but may cause performance
    penalties.

## Release 0.1.2 (2015-11-20)

```
Baseline: ee0ade3
   + 1e66ccd: RELNOTES: Symlink dirents of directories containing a
              file named
              "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA
              _A_RECURSIVE_TARGET_PATTERN" will *not* be traversed
              for transitive target patterns. The motivation here is
              to allow directories that intentionally contain wonky
              symlinks (e.g. foo/bar -> foo) to opt out of being
              consumed by Blaze. For example, given
   + f5773fc: Set the ijar MAX_BUFFER_SIZE to 256 MB
```

New features:

  - java_library now supports the proguard_specs attribute for
    passing Proguard configuration up to Android (not Java) binaries.
  - http_file can specify "executable" to make the downloaded file
    runnable.
  - Debian and tar packaging is now supported
    (see tools/build_defs/pkg/README.md).
  - cpxx_builtin_include_directory specifications allow more
    flexibility.
  - accept %crosstool_top% in cxx_builtin_include_directory
  - android_binary now supports proguard_apply_mapping to re-use a
    previously generated proguard mapping.

Important changes:

  - remove webstatusserver (--use_webstatusserver).
  - Add support for objc textual headers, which will not be compiled
    when modules are enabled.
  - actoolzip, momczip and swiftstdlibtoolzip have all been made into
    bash scripts and have been renamed to actoolwrapper, momcwrapper
    and swiftstdlibtoolwrapper respectively. The old versions will be
    deleted in a later change.
  - [rust] Add rust_bench_test and rust_doc_test rules and improve
    usability of rust_test tule.
  - Java rules now support a resource_strip_prefix attribute that
    allows the removal of path prefixes from Java resources.
  - [docker_build] incremental loading is default now.
    Specify explicitly //package:target.tar (with the .tar extension)
    to obtain the full image.
  - --ios_signing_cert_name allows specifying a cert for iOS app
    signing
  - Go rules for Bazel.
  - [jsonnet] Update to Jsonnet 0.8.1.
  - [jsonnet] Add vars and code_vars attributes to jsonnet_to_json to
    allow passing external variables to Jsonnet via --var and
    --code_var.
  - Adds --override_workspace_root blaze flag to hand-set
    workspace_root and mainGroup in xcodeproj.
  - Allow dots in package names.
  - When used as a forwarding rule (i.e., has no sources),
    android_library
    will also forward any exported_plugins in its dependencies.
  - Add support for Windows-created zip files with non-posix
    permissions.
  - [jsonnet] Add jsonnet_to_json_test rule for testing Jsonnet code.
  - C++ compile actions run in a sandbox now on systems that support
    sandboxed execution.
  - The names of the clang compilers in the Android NDK crosstool no
    longer reference gcc.
  - 420 dpi is now a valid density for andoid_binary.densities.
  - Bazel does strict validation of include files now to ensure
    correct incremental builds. If you see compilation errors when
    building C++ code, please make sure that you explicitly declare
    all header files in the srcs or hdrs attribute of your cc_*
    targets and that your cc_* targets have correct "deps" on
    cc_library's that they use.
  - [jsonnet] Fix jsonnet_to_json rule to read code_vars from
    code_vars instead of vars.
  - Tests, genrules, and Skylark actions without the
    "requires-network" tag will no longer be able to access the
    network.
  - C++ libraries no longer need includes = ["."] (or similar copts)
    to include paths relative to a remote repository's root.
  - Support exports attribute for android_library
  - Symlink dirents of directories containing a file named
    "DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSI
    VE_TARGET_PATTERN" will *not* be traversed for transitive target
    patterns. The motivation here is to allow directories that
    intentionally contain wonky symlinks (e.g. foo/bar -> foo) to opt
    out of being consumed by Blaze.

## Release 0.1.1 (2015-10-05)

```
Baseline: 22616ae
   + 1ef338f: Rollback of "Propagates cc_library linkopts attribute
              to dependent objc_libraries.": breaks certain
              objc_binary build targets.
   + 5fb1073: Reintroduce an inconsistency check (albeit, in a weaker
              form) removed by a previous change that was trying to
              optimize away a filesystem call.
   + 6d00468b2eb976866cfb814d562e0d53a580a46f: Add IdlClass to the embedded default android tools
              repository and rearrange BuildJar's JarHelper so that
              it too can be embedded.
   + a5199039934a2e399a7201adc0d74e2f2d2b0ff3: Fixes Android integration tests by wiring up idlclass
              rules in integration environment.
```

Incompatible changes:

  - Bazel requires JDK 8 to run.
  - Attribute "copts" is removed from j2objc_library.

New features:

  - a cc_binary rule may list '.s' and '.asm' files in the srcs
  - Support for build with libsass.
  - labels in "linkopts" may match any label in either "deps" or
    "srcs" to be considered valid.
  - Maven servers that require username & password authentication are
    now supported (see maven_server documentation).

Important changes:

  - Support empty plist files
  - The <compatible-screens> section of the AndroidManifest.xml will
    not be overwritten if it already contains a <screen> tag for each
    of the densities specified on the android_binary rule.
  - Add Jsonnet rules to Bazel
  - Remove deprecated xcode_options flag.
  - Workspace names are now restricted to being in their base
    directory
    (that is, the names cannot contain up-level references or /./).
  - j2objc_library on Bazel now transpiles transitive proto_library
    dependencies. (Note that java_* rules in Bazel do not yet support
    protos; currently they ignore proto dependencies.)
  - new_http_archive can specify a root directory.
  - Adds support for dylibs on devices for Xcode 7.
  - [d] d_docs rules now depend on a d_binary, a d_library or
    d_source_library.
  - [docker] docker_build now set the permission to 0555 to files
    added to the layer, use `mode = "0644"` to use the legacy behavior.
  - android_binary now has a main_dex_proguard_specs attribute to
    specify which classes should be in the main dex.
  - [rust] Add rust_docs rule for generating rustdoc.
## Release 0.1.0 (2015-09-08)

```
Baseline: a0881e8
   + 87374e6: Make android_binary use a constant, hard-coded,
              checked-in debug key.
   + 2984f1c: Adds some safety checks in the Bazel installer
   + 4e21d90: Remove BUILD.glob and incorporate the necessary
              filegroups into the android_{ndk,sdk}_repository rules
              themselves.
   + 1ee813e: Fix Groovy rules to work with sandboxing
   + 8741978: Add initial D rules to Bazel.
   + 2c2e70d: Fix the installer and fixing the package shiped into
              binary version of Bazel.
```

Initial release.
