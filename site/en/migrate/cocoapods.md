Project: /_project.yaml
Book: /_book.yaml

# Converting CocoaPods Dependencies

{% include "_buttons.html" %}

CocoaPods is a third-party dependency management system for Apple application
development.

[PodToBUILD](https://github.com/pinterest/PodToBUILD){: .external} provides a
`repository_rule` to automatically generate [CocoaPods](https://cocoapods.org/)
Bazel packages that are compatible with [Tulsi](https://tulsi.bazel.build/).

[BazelPods](https://github.com/sergeykhliustin/BazelPods){: .external} provides a 
tool to automatically generate [CocoaPods](https://cocoapods.org/) Bazel packages
using [rules_ios](https://github.com/bazel-ios/rules_ios) to resolve Swift + ObjC
mixed code pods.
