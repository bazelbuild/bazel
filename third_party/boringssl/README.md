BoringSSL.

This is a clone of the repository `https://github.google.com/mdsteele/boringssl-bazel`
at `436432d8` which itself pulls in `https://boringssl.googlesource.com/boringssl`
at `82aa28fa` .

The `BUILD` file is replaced with one containing globs that is more readable.

To update:

1. Save the files `README.md` and `BUILD` from `third_party/boringssl`
2. `git clone --recurse-submodules https://github.com/mdsteele/boringssl-bazel`
3. Copy the git tree or BoringSSL to `third_party/boringssl`
4. Remove the `WORKSPACE` file from `third_party/boringssl`
5. Copy the saved `README.md` and `BUILD` files back
