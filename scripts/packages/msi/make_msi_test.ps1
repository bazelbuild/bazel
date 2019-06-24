Import-Module .\scripts\packages\msi\make_msi_lib.ps1

function Assert-Equal {
    param (
        [Parameter(Position = 0)][string]$x,
        [Parameter(Position = 1)][string]$y
    )
    if ($x -ne $y) {
        throw "Expected equality of ($x) and ($y)"
    }
}

function Assert-NotEqual {
    param (
        [Parameter(Position = 0)][string]$x,
        [Parameter(Position = 1)][string]$y
    )
    if ($x -eq $y) {
        throw "Expected non-equality of ($x) and ($y)"
    }
}

# Tests for Replace-Slashes
Assert-Equal $(Replace-Slashes "") ""
Assert-Equal $(Replace-Slashes 'foo') 'foo'
Assert-Equal $(Replace-Slashes 'foo/bar/baz\qux') 'foo\bar\baz\qux'

# Test for Compute-RelaseNameAndVersion
$rel, $ver = Compute-RelaseNameAndVersion 'bazel-1.2.3-windows-x86_64.exe'
Assert-Equal $rel '1.2.3'
Assert-Equal $ver '1.2.3'
$rel, $ver = Compute-RelaseNameAndVersion 'bazel-0.99.5rc3-windows-x86_64.exe'
Assert-Equal $rel '0.99.5rc3'
Assert-Equal $ver '0.99.5'

# Test for Get-UpgradeGuid
Assert-Equal $(Get-UpgradeGuid 0.28.0) 'B7864F52-FA13-402E-8334-5CF8FE168728'
Assert-Equal $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.0.5)
Assert-Equal $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.0.5rc2)
Assert-Equal $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.3.3)
Assert-Equal $(Get-UpgradeGuid 2.1.0) $(Get-UpgradeGuid 2.0.3rc1)
Assert-NotEqual $(Get-UpgradeGuid 0.28.0) $(Get-UpgradeGuid 0.29.0)
Assert-NotEqual $(Get-UpgradeGuid 1.5.0) $(Get-UpgradeGuid 2.0.3rc1)


Write-Host 'PASSED'
