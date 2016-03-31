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
