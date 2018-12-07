package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.vfs.PathFragment;

public class LabelConstants {

  public static final PathFragment EXTERNAL_PACKAGE_NAME = PathFragment.create("external");
  public static final PackageIdentifier EXTERNAL_PACKAGE_IDENTIFIER =
      PackageIdentifier.createInMainRepo(EXTERNAL_PACKAGE_NAME);


  public static final PathFragment EXTERNAL_PATH_PREFIX = PathFragment.create("external");
}
