package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

@StarlarkBuiltin(
    name = "IncompatiblePlatformProvider",
    doc = "An interface for targets that are incompatible with the target platform. See "
            + "<a href='../../platforms.html#detecting-incompatible-targets-using-bazel-cquery'>"
            + "Detecting incompatible targets using <code>bazel cquery</code></a> for more "
            + "information.",
    category = DocCategory.PROVIDER)
public interface IncompatiblePlatformProviderApi extends StarlarkValue {}
