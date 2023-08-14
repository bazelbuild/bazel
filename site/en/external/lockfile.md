Project: /_project.yaml
Book: /_book.yaml
keywords: product:Bazel,lockfile,Bzlmod

# Bazel Lockfile

{% include "_buttons.html" %}

The lockfile feature in Bazel enables the recording of specific versions or
dependencies of software libraries or packages required by a project. It
achieves this by storing the result of module resolution and extension
evaluation. The lockfile promotes reproducible builds, ensuring consistent
development environments. Additionally, it enhances build efficiency by allowing
Bazel to skip the resolution process when there are no changes in project
dependencies. Furthermore, the lockfile improves stability by preventing
unexpected updates or breaking changes in external libraries, thereby reducing
the risk of introducing bugs.

## Lockfile Generation {:#lockfile-generation}

The lockfile is generated under the workspace root with the name
`MODULE.bazel.lock`. It is created or updated during the build process,
specifically after module resolution and extension evaluation. The lockfile
captures the current state of the project, including the MODULE file, flags,
overrides, and other relevant information. Importantly, it only includes
dependencies that are included in the current invocation of the build.

When changes occur in the project that affect its dependencies, the lockfile is
automatically updated to reflect the new state. This ensures that the lockfile
remains focused on the specific set of dependencies required for the current
build, providing an accurate representation of the project's resolved
dependencies.

## Lockfile Usage {:#lockfile-usage}

The lockfile can be controlled by the flag
[`--lockfile_mode`](/reference/command-line-reference#flag--lockfile_mode) to
customize the behavior of Bazel when the project state differs from the
lockfile. The available modes are:

*   `update` (Default): If the project state matches the lockfile, the
    resolution result is immediately returned from the lockfile. Otherwise,
    resolution is executed, and the lockfile is updated to reflect the current
    state.
*   `error`: If the project state matches the lockfile, the resolution result is
    returned from the lockfile. Otherwise, Bazel throws an error indicating the
    variations between the project and the lockfile. This mode is particularly
    useful when you want to ensure that your project's dependencies remain
    unchanged, and any differences are treated as errors.
*   `off`: The lockfile is not checked at all.

## Lockfile Benefits {:#lockfile-benefits}

The lockfile offers several benefits and can be utilized in various ways:

-   **Reproducible builds.** By capturing the specific versions or dependencies
    of software libraries, the lockfile ensures that builds are reproducible
    across different environments and over time. Developers can rely on
    consistent and predictable results when building their projects.

-   **Efficient resolution skipping.** The lockfile enables Bazel to skip the
    resolution process if there are no changes in the project dependencies since
    the last build. This significantly improves build efficiency, especially in
    scenarios where resolution can be time-consuming.

-   **Stability and risk reduction.** The lockfile helps maintain stability by
    preventing unexpected updates or breaking changes in external libraries. By
    locking the dependencies to specific versions, the risk of introducing bugs
    due to incompatible or untested updates is reduced.

## Lockfile Contents {:#lockfile-contents}

The lockfile contains all the necessary information to determine whether the
project state has changed. It also includes the result of building the project
in the current state. The lockfile consists of two main parts:

1.  Inputs of the module resolution, such as `moduleFileHash`, `flags` and
    `localOverrideHashes`, as well as the output of the resolution, which is
    `moduleDepGraph`.
2.  For each module extension, the lockfile includes inputs that affect it,
    represented by `transitiveDigest`, and the output of running that extension
    referred to as `generatedRepoSpecs`

Here is an example that demonstrates the structure of the lockfile, along with
explanations for each section:

```json
{
  "lockFileVersion": 1,
  "moduleFileHash": "b0f47b98a67ee15f9.......8dff8721c66b721e370",
  "flags": {
    "cmdRegistries": [
      "https://bcr.bazel.build/"
    ],
    "cmdModuleOverrides": {},
    "allowedYankedVersions": [],
    "envVarAllowedYankedVersions": "",
    "ignoreDevDependency": false,
    "directDependenciesMode": "WARNING",
    "compatibilityMode": "ERROR"
  },
  "localOverrideHashes": {
    "bazel_tools": "b5ae1fa37632140aff8.......15c6fe84a1231d6af9"
  },
  "moduleDepGraph": {
    "<root>": {
      "name": "",
      "version": "",
      "executionPlatformsToRegister": [],
      "toolchainsToRegister": [],
      "extensionUsages": [
        {
          "extensionBzlFile": "extension.bzl",
          "extensionName": "lockfile_ext"
        }
      ],
      ...
    }
  },
  "moduleExtensions": {
    "//:extension.bzl%lockfile_ext": {
      "transitiveDigest": "oWDzxG/aLnyY6Ubrfy....+Jp6maQvEPxn0pBM=",
      "generatedRepoSpecs": {
        "hello": {
          "bzlFile": "@@//:extension.bzl",
          ...
        }
      }
    }
  }
}
```

### Module File Hash {:#module-file-hash}

The `moduleFileHash` represents the hash of the `MODULE.bazel` file contents. If
any changes occur in this file, the hash value differs.

### Flags {:#flags}

The `Flags` object stores all the flags that can affect the resolution result.

### Local Override Hashes {:#local-override-hashes}

If the root module includes `local_path_overrides`, this section stores the hash
of the `MODULE.bazel` file in the local repository. It allows tracking changes
to this dependency.

### Module Dependency Graph {:#module-dep-graph}

The `moduleDepGraph` represents the result of the resolution process using the
inputs mentioned above. It forms the dependency graph of all the modules
required to run the project.

### Module Extensions {:#module-extensions}

The `moduleExtensions` section is a map that includes only the extensions used
in the current invocation or previously invoked, while excluding any extensions
that are no longer utilized. In other words, if an extension is not being used
anymore across the dependency graph, it is removed from the `moduleExtensions`
map.

Each entry in this map corresponds to a used extension and is identified by its
containing file and name. The corresponding value for each entry contains the
relevant information associated with that extension:

1.  The `transitiveDigest` the digest of the extension implementation and its
    transitive .bzl files.
2.  The `generatedRepoSpecs` the result of running that extension with the
    current input.

An additional factor that can affect the extension results is their _usages_.
Although not stored in the lockfile, the usages are considered when comparing
the current state of the extension with the one in the lockfile.

## Best Practices {:#best-practices}

To maximize the benefits of the lockfile feature, consider the following best
practices:

*   Regularly update the lockfile to reflect changes in project dependencies or
    configuration. This ensures that subsequent builds are based on the most
    up-to-date and accurate set of dependencies.

*   Include the lockfile in version control to facilitate collaboration and
    ensure that all team members have access to the same lockfile, promoting
    consistent development environments across the project.

By following these best practices, you can effectively utilize the lockfile
feature in Bazel, leading to more efficient, reliable, and collaborative
software development workflows.