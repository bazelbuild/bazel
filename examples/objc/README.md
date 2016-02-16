# Objective C Examples

The example in this directory show typical use of Objective C libraries,
binaries and imports. Because they build an iOS application they can only be run
on Mac OSX.

Build the top-level application with
`bazel build examples/objc:PrenotCalculator`, which when finished prints the
path to the generated .ipa. which you can then install to your test device. The
same build will also print the path to an Xcode project directory which you can
open to continue working with the application in Xcode.

Running `bazel build examples/objc:PrenotCalculatorInstruments` will build and
run the application to obtain a screenshot, the path to which it then prints.


