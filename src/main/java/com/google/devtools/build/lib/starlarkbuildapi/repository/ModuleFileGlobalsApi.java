// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.starlarkbuildapi.repository;

import com.google.devtools.build.docgen.annot.DocumentMethods;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkInt;

/** A collection of global Starlark build API functions that apply to MODULE.bazel files. */
@DocumentMethods
public interface ModuleFileGlobalsApi<ModuleFileFunctionExceptionT extends Exception> {

  @StarlarkMethod(
      name = "module",
      doc =
          "Declares certain properties of the Bazel module represented by the current Bazel repo."
              + " These properties are either essential metadata of the module (such as the name"
              + " and version), or affect behavior of the current module and its dependents.  <p>It"
              + " should be called at most once. It can be omitted only if this module is the root"
              + " module (as in, if it's not going to be depended on by another module).",
      parameters = {
        @Param(
            name = "name",
            // TODO(wyv): explain module name format
            doc =
                "The name of the module. Can be omitted only if this module is the root module (as"
                    + " in, if it's not going to be depended on by another module).",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "version",
            // TODO(wyv): explain version format
            doc =
                "The version of the module. Can be omitted only if this module is the root module"
                    + " (as in, if it's not going to be depended on by another module).",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "compatibility_level",
            // TODO(wyv): See X for more details; also mention multiple_version_override
            doc =
                "The compatibility level of the module; this should be changed every time a major"
                    + " incompatible change is introduced. This is essentially the \"major"
                    + " version\" of the module in terms of SemVer, except that it's not embedded"
                    + " in the version string itself, but exists as a separate field. Modules with"
                    + " different compatibility levels participate in version resolution as if"
                    + " they're modules with different names, but the final dependency graph cannot"
                    + " contain multiple modules with the same name but different compatibility"
                    + " levels.",
            named = true,
            positional = false,
            defaultValue = "0"),
        // TODO(wyv): bazel_compatibility, module_rule_exports, toolchains & platforms
      })
  void module(String name, String version, StarlarkInt compatibilityLevel)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "bazel_dep",
      doc = "Declares a direct dependency on another Bazel module.",
      parameters = {
        @Param(
            name = "name",
            doc = "The name of the module to be added as a direct dependency.",
            named = true,
            positional = false),
        @Param(
            name = "version",
            doc = "The version of the module to be added as a direct dependency.",
            named = true,
            positional = false),
        @Param(
            name = "repo_name",
            doc =
                "The name of the external repo representing this dependency. This is by default the"
                    + " name of the module.",
            named = true,
            positional = false,
            defaultValue = "''"),
      })
  void bazelDep(String name, String version, String repoName)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "single_version_override",
      doc =
          "Specifies that a dependency should still come from a registry, but its version should"
              + " be pinned, or its registry overridden, or a list of patches applied. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "version",
            doc =
                "Override the declared version of this module in the dependency graph. In other"
                    + " words, this module will be \"pinned\" to this override version. This"
                    + " attribute can be omitted if all one wants to override is the registry or"
                    + " the patches. ",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "registry",
            doc =
                "Overrides the registry for this module; instead of finding this module from the"
                    + " default list of registries, the given registry should be used.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  void singleVersionOverride(
      String moduleName,
      String version,
      String registry,
      Iterable<?> patches,
      StarlarkInt patchStrip)
      throws EvalException;

  @StarlarkMethod(
      name = "archive_override",
      doc =
          "Specifies that this dependency should come from an archive file (zip, gzip, etc) at a"
              + " certain location, instead of from a registry. This directive can only be used by"
              + " the root module; in other words, if a module specifies any overrides, it cannot"
              + " be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "urls",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Iterable.class, generic1 = String.class),
            },
            doc = "The URLs of the archive; can be http(s):// or file:// URLs.",
            named = true,
            positional = false),
        @Param(
            name = "integrity",
            doc = "The expected checksum of the archive file, in Subresource Integrity format.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "strip_prefix",
            doc = "A directory prefix to strip from the extracted files.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  void archiveOverride(
      String moduleName,
      Object urls,
      String integrity,
      String stripPrefix,
      Iterable<?> patches,
      StarlarkInt patchStrip)
      throws EvalException, ModuleFileFunctionExceptionT;

  @StarlarkMethod(
      name = "git_override",
      doc =
          "Specifies that a dependency should come from a certain commit of a Git repository. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "remote",
            doc = "The URL of the remote Git repository.",
            named = true,
            positional = false),
        @Param(
            name = "commit",
            doc = "The commit that should be checked out.",
            named = true,
            positional = false,
            defaultValue = "''"),
        @Param(
            name = "patches",
            doc =
                "A list of labels pointing to patch files to apply for this module. The patch files"
                    + " must exist in the source tree of the top level project. They are applied in"
                    + " the list order.",
            allowedTypes = {@ParamType(type = Iterable.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]"),
        @Param(
            name = "patch_strip",
            doc = "Same as the --strip argument of Unix patch.",
            named = true,
            positional = false,
            defaultValue = "0"),
      })
  void gitOverride(
      String moduleName, String remote, String commit, Iterable<?> patches, StarlarkInt patchStrip)
      throws EvalException, ModuleFileFunctionExceptionT;

  @StarlarkMethod(
      name = "local_path_override",
      doc =
          "Specifies that a dependency should come from a certain directory on local disk. This"
              + " directive can only be used by the root module; in other words, if a module"
              + " specifies any overrides, it cannot be used as a dependency by others.",
      parameters = {
        @Param(
            name = "module_name",
            doc = "The name of the Bazel module dependency to apply this override to.",
            named = true,
            positional = false),
        @Param(
            name = "path",
            doc = "The path to the directory where this module is.",
            named = true,
            positional = false),
      })
  void localPathOverride(String moduleName, String path) throws EvalException;

  // TODO(wyv): multiple_version_override
}
