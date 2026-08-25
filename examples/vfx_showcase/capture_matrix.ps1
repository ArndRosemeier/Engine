param(
    [string]$OutputDirectory = "target-vfx-qa/polished"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$captures = @(
    @{ Kind = "fire"; Delivery = "single"; Time = "1.00"; Name = "fire-single-travel.png" },
    @{ Kind = "fire"; Delivery = "single"; Time = "1.72"; Name = "fire-single-impact.png" },
    @{ Kind = "fire"; Delivery = "single"; Time = "2.48"; Name = "fire-single-dissipation.png" },
    @{ Kind = "fire"; Delivery = "aoe"; Time = "1.08"; Name = "fire-aoe.png" },
    @{ Kind = "frost"; Delivery = "aoe"; Time = "1.08"; Name = "frost-aoe.png" },
    @{ Kind = "lightning"; Delivery = "single"; Time = "0.88"; Name = "lightning-single.png" },
    @{ Kind = "poison"; Delivery = "aoe"; Time = "1.18"; Name = "poison-aoe.png" },
    @{ Kind = "root"; Delivery = "aoe"; Time = "0.78"; Name = "root-aoe.png" },
    @{ Kind = "hold"; Delivery = "single"; Time = "1.02"; Name = "hold-single.png" },
    @{ Kind = "snare"; Delivery = "aoe"; Time = "0.92"; Name = "snare-aoe.png" },
    @{ Kind = "charm"; Delivery = "single"; Time = "1.10"; Name = "charm-single.png" },
    @{ Kind = "frost"; Delivery = "cone"; Time = "1.05"; Name = "frost-cone.png" },
    @{ Kind = "hold"; Delivery = "pbaoe"; Time = "1.18"; Name = "hold-pbaoe.png" }
)

foreach ($capture in $captures) {
    $env:VFX_KIND = $capture.Kind
    $env:VFX_DELIVERY = $capture.Delivery
    $env:VFX_SCREENSHOT_TIME = $capture.Time
    $env:ENGINE_SCREENSHOT = Join-Path $OutputDirectory $capture.Name
    $env:ENGINE_SCREENSHOT_WAIT = "1"
    cargo run -q -p vfx_showcase
    if ($LASTEXITCODE -ne 0) {
        throw "VFX capture failed: $($capture.Name)"
    }
}

Remove-Item Env:VFX_KIND, Env:VFX_DELIVERY, Env:VFX_SCREENSHOT_TIME, Env:ENGINE_SCREENSHOT, Env:ENGINE_SCREENSHOT_WAIT