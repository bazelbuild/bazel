# Updating protobuf

1) Fetch the desired version from https://github.com/google/protobuf,
   extract everything into a new directory and change into it.
2) Bazel uses upstream protobuf from source, except for java, as we
   currently don't build protobuf java when bootstrapping bazel. So
   instead we use already build jars. Go to `search.maven.org` and
   search for `com.google.protobuf` and download the jars for protobuf-java
   and protobuf-java-util and place them in the new directory.
3) Modify protobuf's `BUILD` file to not build java from source, but to use
   the jars instead. To do that, in the BUILD file delete the rules listed
   under `Java support`. Then, from the `third_party/protobuf/BUILD file`
   copy the rules under "Modifications made by bazel" to the new BUILD file.
   Those rules should have the same names as the ones you just deleted. You
   will need to update the names of the jars in the rules sources to the
   ones you just downloaded.
4) Copy `third_party/protobuf/com_google_protobuf_java.BUILD` to the new
   directory.
5) Copy this file to the new directory and update it if you found the
   instructions to be wrong or incomplete.
6) Replace the contents in `third_party/protobuf` with the contents in this
   directory.
7) Delete this directory.

# Current protobuf version

The current version of protobuf was obtained from @laszlocsomor's protobuf fork
`https://github.com/laszlocsomor/protobuf` at commit `421d90960d`. Once
`https://github.com/google/protobuf/pull/2969` is merged into upstream
protobuf, we no longer need to use @laszlocsomor's fork but can directly clone
upstream.
