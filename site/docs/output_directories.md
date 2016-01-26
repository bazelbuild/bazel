---
layout: documentation
title: Output Directory Layout
---

# Output Directory Layout

## Requirements

Requirements for an output directory layout:

* Don't collide if multiple users are building on the same box.
* Support building in multiple workspaces at the same time.
* Support building for multiple target configurations in the same workspace.
* Don't collide with any other tools.
* Be easy to access.
* Be easy to clean, even selectively.
* Is unambiguous, even if the user relies on symbolic links when changing into
  his/her client directory.
* All the build state per user should be underneath one directory ("I'd like to
  clean all the .o files from all my clients.")

## Documentation of the current Bazel output directory layout

The solution that's currently implemented:

* Bazel must be invoked from a directory containing a WORKSPACE file. It reports
  an error if it is not. We call this the _workspace directory_.
* The _outputRoot_ directory is ~/.cache/bazel. (Unless `$TEST_TMPDIR` is
  set, as in a test of bazel itself, in which case this directory is used
  instead.)
* We stick the Bazel user's build state beneath `outputRoot/_bazel_$USER`. This
  is called the _outputUserRoot_ directory.
* Beneath the `outputUserRoot` directory, we create an `installBase` directory
  whose name is "install" plus the MD5 hash of the Bazel installation manifest.
* Beneath the `outputUserRoot` directory, we also create an `outputBase`
  directory whose name is the MD5 hash of the path name of the workspace
  directory. So, for example, if Bazel is running in the workspace directory
  `/home/user/src/my-project` (or in a directory symlinked to that one), then we
  create an output base directory called:
  `/home/.cache/bazel/_bazel_user/7ffd56a6e4cb724ea575aba15733d113`.
* Users can use Bazel's `--output_base` startup option to override the default
  output base directory. For example,
  `bazel --output_base=/tmp/bazel/output build x/y:z`.
* Users can also use Bazel's `--output_user_root` startup option to override the
  default install base and output base directories. For example:
  `bazel --output_user_root=/tmp/bazel build x/y:z`.

We put symlinks "bazel-&lt;workspace-name&gt;" and "bazel-out", as well as
"bazel-bin", "bazel-genfiles", and "bazel-includes" in the workspace directory;
these symlinks points to some directories inside a target-specific directory
inside the output directory. These symlinks are only for the user's convenience,
as Bazel itself does not use them. Also, we only do this if the workspace
directory is writable. The names of the "bazel-bin", "bazel-genfiles", and
"bazel-include" symlinks are affected by the `--symlink_prefix` option to bazel,
but "bazel-&lt;workspace-name&gt;" and "bazel-out" are not.

## Bazel internals: Directory layout

The directories are laid out as follows:

<pre>
&lt;workspace-name&gt;/                         <== The workspace directory
  bazel-my-project => <...my-project>     <== Symlink to execRoot
  bazel-out => <...bin>                   <== Convenience symlink to outputPath
  bazel-bin => <...bin>                   <== Convenience symlink to most recent written bin dir $(BINDIR)
  bazel-genfiles => <...genfiles>         <== Convenience symlink to most recent written genfiles dir $(GENDIR)

/home/user/.cache/bazel/                  <== Root for all Bazel output on a machine: outputRoot
  _bazel_$USER/                           <== Top level directory for a given user depends on the user name:
                                              outputUserRoot
    install/
      fba9a2c87ee9589d72889caf082f1029/   <== Hash of the Bazel install manifest: installBase
        _embedded_binaries/               <== Contains binaries and scripts unpacked from the data section of
                                              the bazel executable on first run (e.g. helper scripts and the
                                              main Java file BazelServer_deploy.jar)
    7ffd56a6e4cb724ea575aba15733d113/     <== Hash of the client's workspace directory (e.g.
                                              /home/some-user/src/my-project): outputBase
      action_cache/                       <== Action cache directory hierarchy
                                              This contains the persistent record of the file metadata
                                              (timestamps, and perhaps eventually also MD5 sums) used by the
                                              FilesystemValueChecker.
      action_outs/                        <== Action output directory. This contains a file with the
                                              stdout/stderr for every action from the most recent bazel run
                                              that produced output.
      command.log                         <== A copy of the stdout/stderr output from the most recent bazel
                                              command.
      server/                             <== The Bazel server puts all server-related files (such as socket
                                              file, logs, etc) here.
        server.socket                     <== Socket file for the server.
        server.log                        <== Server logs.
      &lt;workspace-name&gt;/                   <== Working tree for the Bazel build & root of symlink forest: execRoot
        _bin/                             <== Helper tools are linked from or copied to here.

        bazel-out/                        <== All actual output of the build is under here: outputPath
          local_linux-fastbuild/          <== one subdirectory per unique target BuildConfiguration instance;
                                              this is currently encoded
            bin/                          <== Bazel outputs binaries for target configuration here: $(BINDIR)
              foo/bar/_objs/baz/          <== Object files for a cc_* rule named //foo/bar:baz
                foo/bar/baz1.o            <== Object files from source //foo/bar:baz1.cc
                other_package/other.o     <== Object files from source //other_package:other.cc
              foo/bar/baz                 <== foo/bar/baz might be the artifact generated by a cc_binary named
                                              //foo/bar:baz
              foo/bar/baz.runfiles/       <== The runfiles symlink farm for the //foo/bar:baz executable.
                MANIFEST
                &lt;workspace-name&gt;/
                  ...
            genfiles/                     <== Bazel puts generated source for the target configuration here:
                                              $(GENDIR)
              foo/bar.h                       e.g. foo/bar.h might be a headerfile generated by //foo:bargen
            testlogs/                     <== Bazel internal test runner puts test log files here
              foo/bartest.log                 e.g. foo/bar.log might be an output of the //foo:bartest test with
              foo/bartest.status              foo/bartest.status containing exit status of the test (e.g.
                                              PASSED or FAILED (Exit 1), etc)
            include/                      <== a tree with include symlinks, generated as needed.  The
                                              bazel-include symlinks point to here. This is used for
                                              linkstamp stuff, etc.
          host/                           <== BuildConfiguration for build host (user's workstation), for
                                              building prerequisite tools, that will be used in later stages
                                              of the build (ex: Protocol Compiler)
        &lt;packages&gt;/                       <== Packages referenced in the build appear as if under a regular workspace
</pre>

The layout of the *.runfiles directories is documented in more detail in the places pointed to by RunfilesSupport.

## `bazel clean`

`bazel clean` does an `rm -rf` on the `outputPath` and the `action_cache`
directory. It also removes the workspace symlinks. The `--partial` option to
`bazel clean` will clean a configuration-specific `outputDir`, and the
`--expunge` option will clean the entire outputBase.
