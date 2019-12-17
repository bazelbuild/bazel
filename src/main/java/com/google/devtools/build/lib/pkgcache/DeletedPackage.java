package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;

public class DeletedPackage {
  private PackageIdentifier packageIdentifier;
  private boolean matchSubpackages;
  DeletedPackage(PackageIdentifier packageIdentifier, boolean matchSubpackages) {
    this.packageIdentifier = packageIdentifier;
    this.matchSubpackages = matchSubpackages;
  }

  public boolean match(PackageIdentifier otherPackage) {
    if (this.packageIdentifier.equals(otherPackage)) {
      return true;
    }
    return matchSubpackages
        && this.packageIdentifier.getRepository().equals(otherPackage.getRepository())
        && otherPackage.getPackageFragment().startsWith(this.packageIdentifier.getPackageFragment());
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof DeletedPackage) {
      return this.packageIdentifier.equals(((DeletedPackage) o).packageIdentifier)
          && this.matchSubpackages == ((DeletedPackage) o).matchSubpackages;
    }
    return false;
  }
}
