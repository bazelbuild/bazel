choco uninstall bazel --force
rm *.nupkg
choco pack bazel.nuspec
$pkg = get-childitem bazel*.nupkg
choco install $pkg.FullName --verbose --debug --force -y
