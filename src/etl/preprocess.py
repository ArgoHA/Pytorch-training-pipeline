import multiprocessing as mp
from pathlib import Path

import hydra
import loguru
import pypdfium2 as pdfium
from omegaconf import DictConfig
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm


def convert_pdf_to_jpg(pdf_path: Path) -> None:
    """
    Convert pdf to .jpg format
    """
    pages = pdfium.PdfDocument(pdf_path)
    for i, _ in enumerate(pages):
        image = pages.get_page(i)
        pil_image = image.render().to_pil().convert("RGB")
        pil_image.save(pdf_path.with_suffix(f".{i}.jpg"))

    pdf_path.unlink()


def convert_image_to_jpg(filepath: Path) -> None:
    """
    Convert a single image file to .jpg format
    """
    if filepath.suffix.lower() in [".tif", ".jpeg", ".png", ".tiff", ".heic"]:
        try:
            image = Image.open(filepath).convert("RGB")
        except OSError:
            print("Can't open:", filepath.name)
            filepath.unlink()
            return

        image.save(filepath.with_suffix(".jpg"))
        filepath.unlink()

    elif filepath.suffix.lower() == ".pdf":
        convert_pdf_to_jpg(filepath)

    elif filepath.suffix.lower() != ".jpg":
        print("NOT converted:", filepath.stem)
        filepath.unlink()


def convert_images_to_jpg(dir_path: Path, num_threads: int) -> None:
    """
    Convert all images in a directory to .jpg format
    """
    all_files = [f.stem for f in dir_path.iterdir() if not f.name.startswith(".")]

    with mp.Pool(processes=num_threads) as pool:
        filepaths = [
            filepath
            for filepath in dir_path.glob("*")
            if not filepath.name.startswith(".")
        ]

        for _ in tqdm(pool.imap_unordered(convert_image_to_jpg, filepaths)):
            pass

    jpg_files = [f.stem for f in dir_path.iterdir() if f.suffix.lower() == ".jpg"]
    lost_files = set(all_files) - set(jpg_files)

    if not lost_files:
        loguru.logger.info(
            f"All files were converted to .jpg, total amount: {len(jpg_files)}"
        )
    else:
        loguru.logger.warning(
            f"Not converted to .jpg, amount: {len(lost_files)}, files: {lost_files}"
        )


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    root_path = Path(cfg.train.data_path)
    for images_dir in root_path.iterdir():
        if images_dir.is_dir() and not images_dir.name.startswith("."):
            convert_images_to_jpg(images_dir, cfg.train.threads_to_use)


if __name__ == "__main__":
    register_heif_opener()
    main()
