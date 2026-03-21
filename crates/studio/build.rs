use std::{env, path::PathBuf, time::Duration};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let asset_dir = out_dir.join("surfpool-studio-ui");

    println!("cargo:warning=------------ Studio Build Script ------------");
    // Skip if already extracted
    if !asset_dir.join("_next").exists() {
        println!(
            "cargo:warning=Extracting Surfpool Studio UI assets to {}",
            asset_dir.display()
        );
        let url = "https://txtx-public.s3.amazonaws.com/surfpool-studio-ui/latest.zip";
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client for studio asset download");
        let resp = client
            .get(url)
            .send()
            .and_then(|response| response.error_for_status())
            .expect("Failed to download dist zip");
        let reader = std::io::Cursor::new(
            resp.bytes()
                .expect("Failed to read downloaded studio asset archive"),
        );
        let mut zip = zip::ZipArchive::new(reader)
            .expect("Failed to open downloaded studio asset archive as zip");

        zip.extract(&asset_dir).expect("Failed to extract zip");
    } else {
        println!(
            "cargo:warning=Studio assets already found at {}",
            asset_dir.display()
        );
    }
}
