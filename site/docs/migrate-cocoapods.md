---
layout: documentation
title: Converting CocoaPods dependencies
---

<div style="background-color: #EFCBCB; color: #AE2B2B;  border: 1px solid #AE2B2B; border-radius: 5px; border-left: 10px solid #AE2B2B; padding: 0.5em;">
<b>IMPORTANT:</b> The Bazel docs have moved! Please update your bookmark to <a href="https://bazel.build/migrate/cocoapods" style="color: #0000EE;">https://bazel.build/migrate/cocoapods</a>
<p/>
You can <a href="https://blog.bazel.build/2022/02/17/Launching-new-Bazel-site.html" style="color: #0000EE;">read about</a> the migration, and let us <a href="https://forms.gle/onkAkr2ZwBmcbWXj7" style="color: #0000EE;">know what you think</a>.
</div>


# Converting CocoaPods Dependencies

CocoaPods is a third-party dependency management system for Apple application
development.

[PodToBUILD](https://github.com/pinterest/PodToBUILD) provides a
`repository_rule` to automatically generate [CocoaPods](https://cocoapods.org/)
Bazel packages that are compatible with [Tulsi](https://tulsi.bazel.build/).

