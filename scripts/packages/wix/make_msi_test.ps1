Import-Module .\scripts\packages\wix\make_msi_lib.ps1

function AssertEq {
    Param(
        [Parameter(Position = 0)][string]$x,
        [Parameter(Position = 1)][string]$y
    )
    if ($x -ne $y) {
        Throw "Expected equality of ($x) and ($y)"
    }
}

function AssertNe {
    Param(
        [Parameter(Position = 0)][string]$x,
        [Parameter(Position = 1)][string]$y
    )
    if ($x -eq $y) {
        Throw "Expected non-equality of ($x) and ($y)"
    }
}

# Tests for Replace-Slashes
AssertEq $(Replace-Slashes "") ""
AssertEq $(Replace-Slashes "foo") "foo"
AssertEq $(Replace-Slashes "foo/bar/baz\qux") "foo\bar\baz\qux"

# Test for Compute-RelaseNameAndVersion
$rel, $ver = Compute-RelaseNameAndVersion "bazel-1.2.3-windows-x86_64.exe"
AssertEq "$rel" "1.2.3"
AssertEq "$ver" "1.2.3"
$rel, $ver = Compute-RelaseNameAndVersion "bazel-0.99.5rc3-windows-x86_64.exe"
AssertEq "$rel" "0.99.5rc3"
AssertEq "$ver" "0.99.5"

# Test for Get-UpgradeGuid
AssertEq $(Get-UpgradeGuid 0.28.0) "B7864F52-FA13-402E-8334-5CF8FE168728"
AssertEq $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.0.5)
AssertEq $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.0.5rc2)
AssertEq $(Get-UpgradeGuid 1.0.0) $(Get-UpgradeGuid 1.3.3)
AssertEq $(Get-UpgradeGuid 2.1.0) $(Get-UpgradeGuid 2.0.3rc1)
AssertNe $(Get-UpgradeGuid 0.28.0) $(Get-UpgradeGuid 0.29.0)
AssertNe $(Get-UpgradeGuid 1.5.0) $(Get-UpgradeGuid 2.0.3rc1)


Write-Host "PASSED"