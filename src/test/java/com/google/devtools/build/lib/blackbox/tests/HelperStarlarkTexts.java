package com.google.devtools.build.lib.blackbox.tests;

import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import java.io.IOException;
import java.nio.file.Path;

public class HelperStarlarkTexts {
  // todo create helper object??
  private static String WRITE_TEXT_TO_FILE =
      "def _impl(ctx):\n"
          + "  out = ctx.actions.declare_file(ctx.attr.filename)\n"
          + "  ctx.actions.write(out, ctx.attr.text)\n"
          + "  return [DefaultInfo(files = depset([out]))]\n"
          + "\n"
          + "write_to_file = rule(\n"
          + "    implementation = _impl,\n"
          + "    attrs = {\n"
          + "        \"filename\": attr.string(default = \"out.txt\"),\n"
          + "        \"text\": attr.string()\n"
          + "    }\n"
          + ")";

  static String getLoadForRuleWritingTextToFile(String repoName) {
    return String.format("load('%s//:helper.bzl', 'write_to_file')\n", repoName);
  }

  static String callRuleWritingTextToFile(String name, String filename, String text) {
    return String.format("write_to_file(name = '%s', filename = '%s', text ='%s')",
        name, filename, text);
  }

  static Path setupRepositoryWithRuleWritingTextToFile(Path root, String subdir, String text)
      throws IOException {
    Path workspace = PathUtils.writeFileInDir(root, subdir + "/WORKSPACE");
    PathUtils.writeFileInDir(root, subdir + "/helper.bzl", WRITE_TEXT_TO_FILE);
    PathUtils.writeFileInDir(root, subdir + "/BUILD",
        getLoadForRuleWritingTextToFile(""),
        callRuleWritingTextToFile("x", "out", text));
    return workspace.getParent();
  }
}
