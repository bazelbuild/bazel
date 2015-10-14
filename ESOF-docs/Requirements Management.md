# Requirements #
## TASK LIST ##
- [x] send email to bazel developpers
- [ ] writing the report
- [ ] use case diagram (see the notes at the end of this document)
- 
## Requirements Elicitation ##

## Requirements analysis and negotiation ##

## Requirements specification ##

## Requirements validation ##


Why would I want to use Bazel?

Bazel may give you faster build times because it can recompile only the files that need to be recompiled. Similarly, it can skip re-running tests that it knows haven't changed.

Bazel produces deterministic results. This eliminates skew between incremental and clean builds, laptop and CI system, etc.

Bazel can build different client and server apps with the same tool from the same workspace. For example, you can change a client/server protocol in a single commit, and test that the updated mobile app works with the updated server, building both with the same tool, reaping all the aforementioned benefits of Bazel.

Can I see examples?

Yes. For a simple example, see:

https://github.com/bazelbuild/bazel/blob/master/examples/cpp/BUILD

The Bazel source code itself provides more complex examples:

https://github.com/bazelbuild/bazel/blob/master/src/main/java/BUILD\ https://github.com/bazelbuild/bazel/blob/master/src/test/java/BUILD

What is Bazel best at?

Bazel shines at building and testing projects with the following properties:

Projects with a large codebase
Projects written in (multiple) compiled languages
Projects that deploy on multiple platforms
Projects that have extensive tests
