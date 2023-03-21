Project: /_project.yaml
Book: /_book.yaml
# Bazel roadmap

{% include "_buttons.html" %}

## Overview

Happy new year to our Bazel community. With the new year, we plan to bring details of our 2023 roadmap. Last year, we published our 2022 year roadmap with our Bazel 6.0 plans. We hope that the roadmap provided informed your build tooling needs. As the Bazel project continues to evolve in response to your needs, we want to share our 2023 update.

With these changes, we’re looking to keep our open-source community informed and included. This roadmap describes current initiatives and predictions for the future of Bazel development, giving you visibility into current priorities and ongoing projects.

## Bazel 7.0 Release
We plan to bring Bazel 7.0 [long term support (LTS)](https://bazel.build/release/versioning) to you in late 2023. With Bazel 7.0 we aim to deliver many of the in progress items and continue to work through feature improvements that our users have been asking for.

### Better cross-platform cache sharing
Enables [cached artifacts](https://docs.google.com/document/d/1o0mrl2DanfV_6kB_Kf_jUdge13CQ8CvCiqeni2o-rvA/edit#heading=h.mvuo768l4ja2) to be shared across different build local (Mac) and remote (Linux) build platforms primarily for Java/Kotlin and Android development, resulting in better performance and efficient cache usage.

### Android app build with Bazel
Manifest & Resource Merger updates to v30.1.3 so Android app developers can use newer manifest merging features like tools:node="merge".

### Remote execution improvements
Bazel 7.0 provides support for asynchronous execution, speeding up remote execution via increased parallelism with flag --jobs.

### Bzlmod: external dependency management system
[Bzlmod](https://bazel.build/docs/bzlmod) automatically resolves transitive dependencies, allowing projects to scale while staying fast and resource-efficient. Bazel 7.0 contains a number of enhancements to [Bazel's external dependency management](https://docs.google.com/document/d/1moQfNcEIttsk6vYanNKIy3ZuK53hQUFq1b1r0rmsYVg/edit#heading=h.lgyp7ubwxmjc) functionality, including:

-   Bzlmod turned on by default for external dependency management in Bazel
-   Lock file support — enables hermetic build with Bzlmod
-   Vendor/offline mode support — allows users to run builds with pre-downloaded dependencies
-   Complete repository cache support (caching not only downloads artifacts, but also the final repository content)
-   [Bazel Central Registry](https://registry.bazel.build/) includes regular community contribution and adoption of key Bazel rules & projects

### Build analysis metrics
Bazel 7.0 provides analysis-phase time metrics, letting developers optimize their own build performance.

### Build without the Bytes turned on by default
[Builds without the Bytes](https://github.com/bazelbuild/bazel/issues/6862) optimizes performance by avoiding the download of intermediate artifacts and preventing builds from bottlenecking on network bandwidth. Features added include:

-   [Support for remote cache eviction with a lease service](https://docs.google.com/document/d/1wM61xufcMS5W0LQ0ar5JBREiN9zKfgu6AnHVD7BSkR4/edit#heading=h.mflzzzunlhlz), so that users don’t run into errors when artifacts are evicted prematurely

-   Address feature gaps in symlink support
-   Provide options to retrieve intermediate outputs from remote actions

### Build Productivity with Skymeld
Bazel 7.0 introduces Skymeld — an evaluation mode that reduces the wall time of your multi-target builds. Skymeld eliminates the barrier between analysis and execution phases to improve build speeds, especially for builds with multiple top-level targets. However, for single-target builds, no significant difference is expected.

## Bazel Ecosystem & Tooling

### Android app build with Bazel
-   Migrate Android native rules to Starlark: For Bazel 7.0 the Android rules migrate to Starlark to decouple development from Bazel itself and to better enable community contributions. Additionally, we have made these rules independent of the core Bazel binary, allowing us to release more frequently.
-   [Migration of Android rules to Starlark](https://bazel.build/reference/be/android)
-   R8 support: Allows Android app developers to use R8 updated optimizations.
-   Mobile Install: Allows Android app developers to develop, test, deploy any Android app changes quickly through an updated version of [Mobile Install](https://bazel.build/docs/mobile-install).

### Software Bill of Materials data generation (SBOMs) & OSS license compliance tools
With Bazel, developers can generate data to help produce [SBOMs](https://security.googleblog.com/2022/06/sbom-in-action-finding-vulnerabilities.html). This data outputs in text or JSON format, and can be easily formatted to meet [SPDX](https://spdx.dev/specifications/) or [CycloneDX](https://cyclonedx.org/specification/overview/) specifications. Additionally, the process provides rules to declare the licenses Bazel modules are made available under, and tools to build processes around those declarations. See the in-progress [rules_license implementation](https://github.com/bazelbuild/rules_license) on GitHub.

### Signed builds
Bazel provides trusted binaries for Windows and Mac signed with Google keys. This feature enables multi-platform developers/dev-ops to identify the source of Bazel binaries and protect their systems from potentially malicious, unverified binaries.

### Migration of Java, C++, and Python rules to Starlark
Complete migration of Java, C++, and Python rulesets to Starlark. This effort allows Bazel users to fork only rulesets and not Bazel binary codebase, allowing users to

-   Update and customize rules as needed
-   Update rules independently of Bazel

### Bazel-JetBrains* IntelliJ IDEA support
Incremental IntelliJ plugin updates to support the latest JetBrains plugin release.

*This roadmap snapshots targets, and should not be taken as guarantees. Priorities are subject to change in response to developer and customer feedback, or new market opportunities.*

*To be notified of new features — including updates to this roadmap — join the [Google Group](https://groups.google.com/g/bazel-discuss) community.*

*Copyright © 2022 JetBrains s.r.o. JetBrains and IntelliJ are registered trademarks of JetBrains s.r.o