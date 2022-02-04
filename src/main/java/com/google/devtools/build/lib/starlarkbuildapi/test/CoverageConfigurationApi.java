package com.google.devtools.build.lib.starlarkbuildapi.test;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

@StarlarkBuiltin(
    name = "coverage",
    category = DocCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment representing the coverage configuration.")
public interface CoverageConfigurationApi extends StarlarkValue {

  @StarlarkMethod(
      name = "output_generator",
      allowReturnNones = true,
      structField = true,
      doc =
          "Returns the label pointed to by the"
              + " <a href=\"../../user-manual.html#flag--coverage_output_generator\">"
              + "<code>--coverage_output_generator</code></a> option if coverage collection is"
              + " enabled, otherwise returns <code>None</code>. Can be accessed with"
              + " <a href=\"globals.html#configuration_field\"><code>configuration_field"
              + "</code></a>:<br/>"
              + "<pre>attr.label(<br/>"
              + "    default = configuration_field(<br/>"
              + "        fragment = \"coverage\",<br/>"
              + "        name = \"output_generator\"<br/>"
              + "    )<br/>"
              + ")</pre>")
  @Nullable
  Label outputGenerator();
}
