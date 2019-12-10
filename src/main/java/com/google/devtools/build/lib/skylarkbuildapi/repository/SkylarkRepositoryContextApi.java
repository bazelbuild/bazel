// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skylarkbuildapi.repository;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Skylark API for the repository_rule's context. */
@SkylarkModule(
    name = "repository_ctx",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "The context of the repository rule containing"
            + " helper functions and information about attributes. You get a repository_ctx object"
            + " as an argument to the <code>implementation</code> function when you create a"
            + " repository rule.")
public interface SkylarkRepositoryContextApi<RepositoryFunctionExceptionT extends Throwable>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "name",
      structField = true,
      doc = "The name of the external repository created by this rule.")
  String getName();

  @SkylarkCallable(
      name = "attr",
      structField = true,
      doc =
          "A struct to access the values of the attributes. The values are provided by "
              + "the user (if not, a default value is used).")
  StructApi getAttr();

  @SkylarkCallable(
      name = "path",
      doc =
          "Returns a path from a string, label or path. If the path is relative, it will resolve "
              + "relative to the repository directory. If the path is a label, it will resolve to "
              + "the path of the corresponding file. Note that remote repositories are executed "
              + "during the analysis phase and thus cannot depends on a target result (the "
              + "label should point to a non-generated file). If path is a path, it will return "
              + "that path as is.",
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "string, label or path from which to create a path from")
      })
  RepositoryPathApi<?> path(Object path) throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "report_progress",
      doc = "Updates the progress status for the fetching of this repository",
      parameters = {
        @Param(
            name = "status",
            allowedTypes = {@ParamType(type = String.class)},
            doc = "string describing the current status of the fetch progress")
      })
  void reportProgress(String status);

  @SkylarkCallable(
      name = "symlink",
      doc = "Creates a symlink on the filesystem.",
      useLocation = true,
      parameters = {
        @Param(
            name = "from",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path to which the created symlink should point to."),
        @Param(
            name = "to",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path of the symlink to create, relative to the repository directory."),
      })
  void symlink(Object from, Object to, Location location)
      throws RepositoryFunctionExceptionT, EvalException, InterruptedException;

  @SkylarkCallable(
      name = "file",
      doc = "Generates a file in the repository directory with the provided content.",
      useLocation = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path of the file to create, relative to the repository directory."),
        @Param(
            name = "content",
            type = String.class,
            named = true,
            defaultValue = "''",
            doc = "the content of the file to create, empty by default."),
        @Param(
            name = "executable",
            named = true,
            type = Boolean.class,
            defaultValue = "True",
            doc = "set the executable flag on the created file, true by default."),
        @Param(
            name = "legacy_utf8",
            named = true,
            type = Boolean.class,
            defaultValue = "True",
            doc =
                "encode file content to UTF-8, true by default. Future versions will change"
                    + " the default and remove this parameter."),
      })
  void createFile(
      Object path, String content, Boolean executable, Boolean legacyUtf8, Location location)
      throws RepositoryFunctionExceptionT, EvalException, InterruptedException;

  @SkylarkCallable(
      name = "template",
      doc =
          "Generates a new file using a <code>template</code>. Every occurrence in "
              + "<code>template</code> of a key of <code>substitutions</code> will be replaced by "
              + "the corresponding value. The result is written in <code>path</code>. An optional"
              + "<code>executable</code> argument (default to true) can be set to turn on or off"
              + "the executable bit.",
      useLocation = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path of the file to create, relative to the repository directory."),
        @Param(
            name = "template",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path to the template file."),
        @Param(
            name = "substitutions",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "substitutions to make when expanding the template."),
        @Param(
            name = "executable",
            type = Boolean.class,
            defaultValue = "True",
            named = true,
            doc = "set the executable flag on the created file, true by default."),
      })
  void createFileFromTemplate(
      Object path,
      Object template,
      Dict<?, ?> substitutions, // <String, String> expected
      Boolean executable,
      Location location)
      throws RepositoryFunctionExceptionT, EvalException, InterruptedException;

  @SkylarkCallable(
      name = "read",
      doc = "Reads the content of a file on the filesystem.",
      useLocation = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc = "path of the file to read from."),
      })
  String readFile(Object path, Location location)
      throws RepositoryFunctionExceptionT, EvalException, InterruptedException;

  @SkylarkCallable(
      name = "os",
      structField = true,
      doc = "A struct to access information from the system.",
      useLocation = true)
  SkylarkOSApi getOS(Location location);

  @SkylarkCallable(
      name = "execute",
      doc =
          "Executes the command given by the list of arguments. The execution time of the command"
              + " is limited by <code>timeout</code> (in seconds, default 600 seconds). This method"
              + " returns an <code>exec_result</code> structure containing the output of the"
              + " command. The <code>environment</code> map can be used to override some"
              + " environment variables to be passed to the process.",
      useLocation = true,
      parameters = {
        @Param(
            name = "arguments",
            type = Sequence.class,
            doc =
                "List of arguments, the first element should be the path to the program to "
                    + "execute."),
        @Param(
            name = "timeout",
            type = Integer.class,
            named = true,
            defaultValue = "600",
            doc = "maximum duration of the command in seconds (default is 600 seconds)."),
        @Param(
            name = "environment",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "force some environment variables to be set to be passed to the process."),
        @Param(
            name = "quiet",
            type = Boolean.class,
            defaultValue = "True",
            named = true,
            doc = "If stdout and stderr should be printed to the terminal."),
        @Param(
            name = "working_directory",
            type = String.class,
            defaultValue = "\"\"",
            named = true,
            doc =
                "Working directory for command execution.\n"
                    + "Can be relative to the repository root or absolute."),
      })
  SkylarkExecutionResultApi execute(
      Sequence<?> arguments,
      Integer timeout,
      Dict<?, ?> environment, // <String, String> expected
      boolean quiet,
      String workingDirectory,
      Location location)
      throws EvalException, RepositoryFunctionExceptionT, InterruptedException;

  @SkylarkCallable(
      name = "delete",
      doc =
          "Deletes a file or a directory. Returns a bool, indicating whether the file or directory"
              + " was actually deleted by this call.",
      useLocation = true,
      parameters = {
        @Param(
            name = "path",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc =
                "Path of the file to delete, relative to the repository directory, or absolute."
                    + " Can be a path or a string."),
      })
  boolean delete(Object path, Location location)
      throws EvalException, RepositoryFunctionExceptionT, InterruptedException;

  @SkylarkCallable(
      name = "patch",
      doc =
          "Apply a patch file to the root directory of external repository. "
              + "The patch file should be a standard "
              + "<a href=\"https://en.wikipedia.org/wiki/Diff#Unified_format\">"
              + "unified diff format</a> file. "
              + "The Bazel-native patch implementation doesn't support fuzz match and binary patch "
              + "like the patch command line tool.",
      useLocation = true,
      parameters = {
        @Param(
            name = "patch_file",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            doc =
                "The patch file to apply, it can be label, relative path or absolute path. "
                    + "If it's a relative path, it will resolve to the repository directory."),
        @Param(
            name = "strip",
            type = Integer.class,
            named = true,
            defaultValue = "0",
            doc = "strip the specified number of leading components from file names."),
      })
  void patch(Object patchFile, Integer strip, Location location)
      throws EvalException, RepositoryFunctionExceptionT, InterruptedException;

  @SkylarkCallable(
      name = "which",
      doc =
          "Returns the path of the corresponding program or None "
              + "if there is no such program in the path.",
      allowReturnNones = true,
      useLocation = true,
      parameters = {
        @Param(
            name = "program",
            type = String.class,
            named = false,
            doc = "Program to find in the path."),
      })
  RepositoryPathApi<?> which(String program, Location location) throws EvalException;

  @SkylarkCallable(
      name = "download",
      doc =
          "Downloads a file to the output path for the provided url and returns a struct"
              + " containing a hash of the file with the fields <code>sha256</code> and"
              + " <code>integrity</code>.",
      useLocation = true,
      parameters = {
        @Param(
            name = "url",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            named = true,
            doc = "List of mirror URLs referencing the same file."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            defaultValue = "''",
            named = true,
            doc = "path to the output file, relative to the repository directory."),
        @Param(
            name = "sha256",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "the expected SHA-256 hash of the file downloaded."
                    + " This must match the SHA-256 hash of the file downloaded. It is a security"
                    + " risk to omit the SHA-256 as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
        @Param(
            name = "executable",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc = "set the executable flag on the created file, false by default."),
        @Param(
            name = "allow_fail",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc =
                "If set, indicate the error in the return value"
                    + " instead of raising an error for failed downloads"),
        @Param(
            name = "canonical_id",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "If set, restrict cache hits to those cases where the file was added to the cache"
                    + " with the same canonical id"),
        @Param(
            name = "auth",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "An optional dict specifying authentication information for some of the URLs."),
        @Param(
            name = "integrity",
            type = String.class,
            defaultValue = "''",
            named = true,
            positional = false,
            doc =
                "Expected checksum of the file downloaded, in Subresource Integrity format."
                    + " This must match the checksum of the file downloaded. It is a security"
                    + " risk to omit the checksum as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
      })
  StructApi download(
      Object url,
      Object output,
      String sha256,
      Boolean executable,
      Boolean allowFail,
      String canonicalId,
      Dict<?, ?> auth, // <String, Dict<?, ?>> expected
      String integrity,
      Location location)
      throws RepositoryFunctionExceptionT, EvalException, InterruptedException;

  @SkylarkCallable(
      name = "extract",
      doc = "Extract an archive to the repository directory.",
      useLocation = true,
      parameters = {
        @Param(
            name = "archive",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            named = true,
            doc =
                "path to the archive that will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            defaultValue = "''",
            named = true,
            doc =
                "path to the directory where the archive will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "stripPrefix",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "a directory prefix to strip from the extracted files."
                    + "\nMany archives contain a top-level directory that contains all files in the"
                    + " archive. Instead of needing to specify this prefix over and over in the"
                    + " <code>build_file</code>, this field can be used to strip it from extracted"
                    + " files."),
      })
  void extract(Object archive, Object output, String stripPrefix, Location location)
      throws RepositoryFunctionExceptionT, InterruptedException, EvalException;

  @SkylarkCallable(
      name = "download_and_extract",
      doc =
          "Downloads a file to the output path for the provided url, extracts it, and returns"
              + " a struct containing a hash of the downloaded file with the fields"
              + " <code>sha256</code> and <code>integrity</code>.",
      useLocation = true,
      parameters = {
        @Param(
            name = "url",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            named = true,
            doc = "List of mirror URLs referencing the same file."),
        @Param(
            name = "output",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Label.class),
              @ParamType(type = RepositoryPathApi.class)
            },
            defaultValue = "''",
            named = true,
            doc =
                "path to the directory where the archive will be unpacked,"
                    + " relative to the repository directory."),
        @Param(
            name = "sha256",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "the expected SHA-256 hash of the file downloaded."
                    + " This must match the SHA-256 hash of the file downloaded. It is a security"
                    + " risk to omit the SHA-256 as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."
                    + " If provided, the repository cache will first be checked for a file with the"
                    + " given hash; a download will only be attempted if the file was not found in"
                    + " the cache. After a successful download, the file will be added to the"
                    + " cache."),
        @Param(
            name = "type",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "the archive type of the downloaded file."
                    + " By default, the archive type is determined from the file extension of"
                    + " the URL."
                    + " If the file has no extension, you can explicitly specify either \"zip\","
                    + " \"jar\", \"war\", \"tar.gz\", \"tgz\", \"tar.bz2\", or \"tar.xz\" here."),
        @Param(
            name = "stripPrefix",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "a directory prefix to strip from the extracted files."
                    + "\nMany archives contain a top-level directory that contains all files in the"
                    + " archive. Instead of needing to specify this prefix over and over in the"
                    + " <code>build_file</code>, this field can be used to strip it from extracted"
                    + " files."),
        @Param(
            name = "allow_fail",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc =
                "If set, indicate the error in the return value"
                    + " instead of raising an error for failed downloads"),
        @Param(
            name = "canonical_id",
            type = String.class,
            defaultValue = "''",
            named = true,
            doc =
                "If set, restrict cache hits to those cases where the file was added to the cache"
                    + " with the same canonical id"),
        @Param(
            name = "auth",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "An optional dict specifying authentication information for some of the URLs."),
        @Param(
            name = "integrity",
            type = String.class,
            defaultValue = "''",
            named = true,
            positional = false,
            doc =
                "Expected checksum of the file downloaded, in Subresource Integrity format."
                    + " This must match the checksum of the file downloaded. It is a security"
                    + " risk to omit the checksum as remote files can change. At best omitting this"
                    + " field will make your build non-hermetic. It is optional to make development"
                    + " easier but should be set before shipping."),
      })
  StructApi downloadAndExtract(
      Object url,
      Object output,
      String sha256,
      String type,
      String stripPrefix,
      Boolean allowFail,
      String canonicalId,
      Dict<?, ?> auth, // <String, Dict<?, ?>> expected
      String integrity,
      Location location)
      throws RepositoryFunctionExceptionT, InterruptedException, EvalException;
}
