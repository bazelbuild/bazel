param(
  [string] $version = "0.3.1"
)

choco uninstall bazel --force -y

choco install "./bazel.$($version).nupkg" --verbose --debug --force -y

if ($LASTEXITCODE -ne 0)
{
  write-error "`$LASTEXITCODE was not zero. Inspect the output from choco install above."
  exit 1
}

write-host @"
The package should have installed without errors.

Now:
* open an msys2 shell
* Make sure your environment is accurate (see ``./tools/chocolateyinstall.ps1`` output)
* run ``bazel version`` in that msys2 shell
* ... and you should get a version number back
"@
