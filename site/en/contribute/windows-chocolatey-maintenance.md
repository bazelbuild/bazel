Project: /_project.yaml
Book: /_book.yaml

# Maintaining Bazel Chocolatey package on Windows

{% include "_buttons.html" %}

Note: The Chocolatey package is experimental; please provide feedback
(`@petemounce` in issue tracker).

## Prerequisites {:#prerequisites}

You need:

*    [chocolatey package manager](https://chocolatey.org) installed
*    (to publish) a chocolatey API key granting you permission to publish the
     `bazel` package
     * [@petemounce](https://github.com/petemounce){: .external} currently
     maintains this unofficial package.
*    (to publish) to have set up that API key for the chocolatey source locally
     via `choco apikey -k <your key here> -s https://chocolatey.org/`

## Build {:#build}

Compile bazel with msys2 shell and `compile.sh`.

```powershell
pushd scripts/packages/chocolatey
  ./build.ps1 -version 0.3.2 -mode local
popd
```

Should result in `scripts/packages/chocolatey/bazel.<version>.nupkg` being
created.

The `build.ps1` script supports `mode` values `local`, `rc` and `release`.

## Test {:#test}

0. Build the package (with `-mode local`)

    * run a webserver (`python -m SimpleHTTPServer` in
      `scripts/packages/chocolatey` is convenient and starts one on
      `http://localhost:8000`)

0. Test the install

    The `test.ps1` should install the package cleanly (and error if it did not
    install cleanly), then tell you what to do next.

0. Test the uninstall

    ```sh
    choco uninstall bazel
    # should remove bazel from the system
    ```

Chocolatey's moderation process automates checks here as well.

## Release {:#release}

Modify `tools/parameters.json` for the new release's URI and checksum once the
release has been published to github releases.

```powershell
./build.ps1 -version <version> -isRelease
./test.ps1 -version <version>
# if the test.ps1 passes
choco push bazel.x.y.z.nupkg --source https://chocolatey.org/
```

Chocolatey.org will then run automated checks and respond to the push via email
to the maintainers.
