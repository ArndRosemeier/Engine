//! Which graphics card and driver the renderer runs on, and a warning for
//! drivers known to draw wrongly.
//!
//! NVIDIA driver 528.76 (early 2023) on an RTX 4060 laptop drew far things
//! over near ones in about half of all starts (depth test passing where it
//! must fail; every depth value itself was right). Driver 621.14 drew
//! correctly. Engine cannot repair a driver, so the game should tell the
//! player to update it.

/// The adapter the renderer chose.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    /// PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel).
    pub vendor: u32,
    /// Driver name and version as the backend reports them.
    pub driver: String,
    pub driver_info: String,
    /// "Vulkan", "Dx12", "Metal", ...
    pub backend: String,
}

const NVIDIA: u32 = 0x10DE;

/// NVIDIA drivers below this are known to be unsafe (see the module notes).
/// 528.76 failed, 621.14 worked; 550 (early 2024) is the first line that
/// is not known to fail, chosen between the two.
pub const NVIDIA_MIN_DRIVER: f32 = 550.0;

impl GpuInfo {
    /// NVIDIA's driver version as players know it (e.g. 621.14), if the
    /// backend tells it. Vulkan reports "621.14"; DirectX the Windows form
    /// "31.0.15.2876", whose last five digits are 528.76.
    pub fn nvidia_driver(&self) -> Option<f32> {
        if self.vendor != NVIDIA {
            return None;
        }
        for text in [&self.driver_info, &self.driver] {
            let parts: Vec<&str> = text.trim().split('.').collect();
            match parts.as_slice() {
                [major, minor] if major.len() == 3 => {
                    if let Ok(v) = format!("{major}.{minor}").parse() {
                        return Some(v);
                    }
                }
                [_, _, a, b] => {
                    let digits = format!("{a}{b}");
                    if digits.len() >= 5 && digits.chars().all(|c| c.is_ascii_digit()) {
                        let last = &digits[digits.len() - 5..];
                        if let Ok(v) = format!("{}.{}", &last[..3], &last[3..]).parse() {
                            return Some(v);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// A sentence for the player when the driver is known to draw wrongly.
    pub fn driver_warning(&self) -> Option<String> {
        let v = self.nvidia_driver()?;
        (v < NVIDIA_MIN_DRIVER).then(|| {
            format!(
                "Your NVIDIA driver ({v:.2}) is old and can draw far things in front of near ones. \
                 Please update it from nvidia.com (version {NVIDIA_MIN_DRIVER:.0} or newer)."
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvidia(driver: &str, info: &str) -> GpuInfo {
        GpuInfo {
            name: "NVIDIA GeForce RTX 4060 Laptop GPU".into(),
            vendor: NVIDIA,
            driver: driver.into(),
            driver_info: info.into(),
            backend: "Vulkan".into(),
        }
    }

    #[test]
    fn reads_nvidia_versions_from_vulkan_and_directx() {
        assert_eq!(nvidia("NVIDIA", "621.14").nvidia_driver(), Some(621.14));
        assert_eq!(nvidia("31.0.15.2876", "").nvidia_driver(), Some(528.76));
    }

    #[test]
    fn warns_for_the_driver_that_failed_only() {
        assert!(nvidia("31.0.15.2876", "").driver_warning().is_some());
        assert!(nvidia("NVIDIA", "528.76").driver_warning().is_some());
        assert!(nvidia("NVIDIA", "621.14").driver_warning().is_none());
        let amd = GpuInfo {
            vendor: 0x1002,
            ..nvidia("AMD", "24.1")
        };
        assert!(amd.driver_warning().is_none());
    }
}
