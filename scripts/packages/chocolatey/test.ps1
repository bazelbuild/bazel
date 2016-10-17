param(
  [string] $version = "0.3.2"
)

choco uninstall bazel --force -y

choco install bazel --verbose --debug --force -y -s ".;https://chocolatey.org/api/v2/"

if ($LASTEXITCODE -ne 0)
{
  write-error "`$LASTEXITCODE was not zero. Inspect the output from choco install above."
  exit 1
}

write-host @"
The package should have installed without errors.

Now:
* open a new shell (this should work in msys2, cmd, powershell)
* Make sure your environment is accurate (see ``./tools/chocolateyinstall.ps1`` output)
* run ``bazel version`` in that shell
* ... and you should get a version number back
"@
