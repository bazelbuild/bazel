// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.starlarkbuildapi.apple;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** An interface for an object representing an Apple platform. */
@StarlarkBuiltin(
    name = "apple_platform",
    category = DocCategory.BUILTIN,
    doc =
        "Corresponds to Xcode's notion of a platform as would be found in"
            + " <code>Xcode.app/Contents/Developer/Platforms</code>. Each platform represents an"
            + " Apple platform type (such as iOS or tvOS) combined with one or more related CPU"
            + " architectures. For example, the iOS simulator platform supports <code>x86_64</code>"
            + " and <code>i386</code> architectures.<p>Specific instances of this type can be"
            + " retrieved from the fields of the <a"
            + " href='../toplevel/apple_common.html#platform'>apple_common.platform</a>"
            + " struct:<br><ul><li><code>apple_common.platform.ios_device</code></li>"
            + "<li><code>apple_common.platform.ios_simulator</code></li>"
            + "<li><code>apple_common.platform.macos</code></li>"
            + "<li><code>apple_common.platform.tvos_device</code></li>"
            + "<li><code>apple_common.platform.tvos_simulator</code></li>"
            + "<li><code>apple_common.platform.watchos_device</code></li><li><code>apple_common.platform.watchos_simulator</code></li></ul><p>More"
            + " commonly, however, the <a href='../fragments/apple.html'>apple</a> configuration"
            + " fragment has fields/methods that allow rules to determine the platform for which a"
            + " target is being built.<p>Example:<br><pre class='language-python'>\n"
            + "p = apple_common.platform.ios_device\n"
            + "print(p.name_in_plist)  # 'iPhoneOS'\n"
            + "</pre>")
public interface ApplePlatformApi extends StarlarkValue {

  /** Returns the platform type of this platform. */
  @StarlarkMethod(
      name = "platform_type",
      doc = "Returns the platform type of this platform.",
      structField = true)
  String getType();

  /**
   * Returns true if this platform is a device platform, or false if this is a simulator platform.
   */
  @StarlarkMethod(
      name = "is_device",
      doc =
          "Returns <code>True</code> if this platform is a device platform or <code>False</code> "
              + "if it is a simulator platform.",
      structField = true)
  boolean isDevice();

  /**
   * Returns the name of the "platform" as it appears in the CFBundleSupportedPlatforms plist
   * setting.
   */
  @StarlarkMethod(
      name = "name_in_plist",
      structField = true,
      doc =
          "The name of the platform as it appears in the <code>CFBundleSupportedPlatforms</code>"
              + " entry of an Info.plist file and in Xcode's platforms directory, without the"
              + " extension (for example, <code>iPhoneOS</code> or"
              + " <code>iPhoneSimulator</code>).<br>This name, when converted to lowercase (e.g.,"
              + " <code>iphoneos</code>, <code>iphonesimulator</code>), can be passed to Xcode's"
              + " command-line tools like <code>ibtool</code> and <code>actool</code> when they"
              + " expect a platform name.")
  String getNameInPlist();
}
