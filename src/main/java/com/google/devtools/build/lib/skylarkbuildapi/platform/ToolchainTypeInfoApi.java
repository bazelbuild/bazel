package com.google.devtools.build.lib.skylarkbuildapi.platform;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Info object representing data about a specific platform. */
@SkylarkModule(
    name = "ToolchainTypeInfo",
    doc =
        "Provides access to data about a specific toolchain type. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = SkylarkModuleCategory.PROVIDER)
public interface ToolchainTypeInfoApi extends StructApi {

  @SkylarkCallable(
      name = "type_label",
      doc = "The label uniquely identifying this toolchain type.",
      structField = true)
  Label typeLabel();
}
