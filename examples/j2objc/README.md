# J2ObjC Examples

J2ObjC is an open-source tool that can transpile Java code to Objective-C code,
which can then be used by dependent Objective-C code. The J2ObjC repository can
be found at <https://github.com/google/j2objc>.

The example in this directory shows a simple use of J2Objc with a Java library
and an iOS app.
Because it builds an iOS application it can only be run on Mac OSX.
Here, a java_library is transpiled to Objective-C via j2objc_library.
We can then have an objc_library call upon this library.

Build the top-level application with
`bazel build examples/j2objc:J2ObjcExample`, which when finished emits the
path to a generated .ipa which you can then install to your test device. The
same build will also emits the path to an Xcode project directory which you can
open to continue working with the application in Xcode.
